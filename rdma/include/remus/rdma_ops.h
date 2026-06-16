#pragma once

#include <atomic>
#include <cstdint>

#include "connection.h"
#include "rdma_ptr.h"

// [mfs]  Could we have async versions of the one-sided operations?
//
// [mfs]  We might need a better way of polling the completion queue
namespace remus::internal {

/// Internal utility function for performing a one-sided write over RDMA
template <typename T>
void Write(rdma_ptr<T> ptr, const T &val, uint8_t *seg, int32_t rkey,
           int32_t lkey, Connection *conn, std::atomic<int> *ack) {
  T *local = (T *)seg;

  REMUS_ASSERT((uint64_t)local != ptr.address(), "WTF");
  // [mfs] Is this memset really necessary?  Maybe for gaps in structs?
  std::memset(local, 0, sizeof(T));
  *local = val;
  ibv_sge sge; // only 3 fields, so no need to memset it to 0
  sge.addr = reinterpret_cast<uint64_t>(local);
  sge.length = sizeof(T);
  sge.lkey = lkey;

  // [mfs] Check the args
  ibv_send_wr send_wr_{}; // [mfs] Should we memset this?  It's large.
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_RDMA_WRITE;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.rdma.remote_addr = ptr.address();
  send_wr_.wr.rdma.rkey = rkey;

  // set the counter to the # of completions we expect, then make request
  *ack = 1;
  conn->send_onesided(&send_wr_);
  // Poll until we get an ack for this event, and assert it's good
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    if (wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   (std::stringstream() << ptr).str());
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }
}

/// Do a 64-bit CAS over RDMA
template <typename T>
T CompareAndSwap(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap,
                 uint8_t *seg, int32_t rkey, int32_t lkey, Connection *conn,
                 std::atomic<int> *ack) {
  static_assert(sizeof(T) == 8);

  // [mfs] Why are we casting to volatile?
  volatile uint64_t *prev_ = reinterpret_cast<volatile uint64_t *>(seg);

  ibv_sge sge; // only 3 fields, so no need to memset it to 0
  sge.addr = reinterpret_cast<uint64_t>(prev_);
  sge.length = sizeof(uint64_t);
  sge.lkey = lkey;

  ibv_send_wr send_wr_{}; // [mfs] Should we memset this? It's large.
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.atomic.remote_addr = ptr.address();
  send_wr_.wr.atomic.rkey = rkey;
  send_wr_.wr.atomic.compare_add = expected;
  send_wr_.wr.atomic.swap = swap;

  // set the counter to the number of work completions we expect
  *ack = 1;
  conn->send_onesided(&send_wr_);

  // Poll until we match on the condition
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    // Assert a good result
    if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                   "ibv_poll_cq(): {} @ {}",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   format_rdma_ptr(ptr));
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }

  T ret = T(*prev_);
  return ret;
}

template <typename T>
T FetchAndAdd(rdma_ptr<T> ptr, uint64_t add, uint8_t *seg, int32_t rkey,
              int32_t lkey, Connection *conn, std::atomic<int> *ack) {
  static_assert(sizeof(T) == 8);
  // [mfs] Why are we casting it to volatile, just to uncast it here?  Isn't seg
  // thread-local?
  volatile uint64_t *prev = reinterpret_cast<volatile uint64_t *>(seg);

  ibv_sge sge; // only 3 fields, so no need to memset it to 0
  sge.addr = reinterpret_cast<uint64_t>(prev);
  sge.length = sizeof(uint64_t);
  sge.lkey = lkey;

  ibv_send_wr send_wr_{}; // [mfs] Should we memset this? It's large.
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.atomic.remote_addr = ptr.address();
  send_wr_.wr.atomic.rkey = rkey;
  send_wr_.wr.atomic.compare_add = add;

  // set the counter to the number of work completions we expect
  *ack = 1;
  conn->send_onesided(&send_wr_);

  // Poll until we match on the condition
  //
  // [mfs] Can we move polling into a helper function and reuse it broadly?
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    // Assert a good result
    if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                   "ibv_poll_cq(): {} @ {}",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   format_rdma_ptr(ptr));
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }

  T ret = T(*prev);
  return ret;
}

#if 0
    
    /// Write a block of data to a remote RDMA address from a preallocated local
    /// buffer.
    ///
    /// @tparam T The data type of the RDMA pointers.
    ///
    /// @param ptr      The remote RDMA pointer where data will be written.
    /// @param prealloc The local RDMA pointer containing the data to write.
    /// @param size     The number of bytes to write from `prealloc` to `ptr`.
    ///
    /// @throws std::invalid_argument If the RDMA pointers are invalid or
    /// overlapping.
    /// @throws std::runtime_error     If RDMA operations fail.
    template <typename T>
    void ExtendedWrite(rdma_ptr<T> ptr, rdma_ptr<T> prealloc, uint64_t size,
                       MemoryPool &mp) {
    
      // Validate the RDMA pointers
      auto info = cm_.get_conn(ptr.raw(), 0); // [mfs] Using 0 for now
    
      // Retrieve the thread's index for tracking completions
      auto thread_id = std::this_thread::get_id();
      auto thread_it = cm_.thread_ids.find(thread_id);
      assert(thread_it != cm_.thread_ids.end());
      uint64_t index_as_id = thread_it->second;
    
      // Ensure that the prealloc address does not overlap with the remote pointer
      assert(reinterpret_cast<uintptr_t>(prealloc.raw()) != ptr.raw());
    
      // Prepare the scatter-gather entry (SGE) for the RDMA write
      ibv_sge sge{};
      sge.addr = reinterpret_cast<uint64_t>(std::to_address(prealloc));
      sge.length = size;
      // Ensure lkey corresponds to prealloc's memory region
      sge.lkey = mp.rdma_memory_->mr()->lkey;
    
      // Prepare the send work request (WR)
      ibv_send_wr send_wr{};
      send_wr.wr_id = index_as_id;
      send_wr.num_sge = 1;
      send_wr.sg_list = &sge;
      send_wr.opcode = IBV_WR_RDMA_WRITE;
      send_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
      send_wr.wr.rdma.remote_addr = ptr.address();
      send_wr.wr.rdma.rkey = info.rkey;
    
      // Set the counter to expect one work completion
      *ack = 1;
    
      // Send the RDMA write request
      info.conn->send_onesided(&send_wr);
    
      // Poll the completion queue until the write is acknowledged
      ibv_wc wc;
      // [mfs] These counters could have awful cache thrashing
      while (*ack != 0) {
        int poll = info.conn->poll_cq(1, &wc);
        if (poll == 0 || (poll < 0 && errno == EAGAIN)) {
          // No completions yet, continue polling
          //
          // [mfs] Why?  What if someone ack'd me?
          continue;
        }
    
        if (poll < 0) {
          throw std::runtime_error("Failed to poll completion queue: " +
                                   std::string(strerror(errno)));
        }
    
        // Check for successful completion
        if (wc.status != IBV_WC_SUCCESS) {
          throw std::runtime_error("RDMA Write failed with status: " +
                                   std::string(ibv_wc_status_str(wc.status)));
        }
    
        // Decrement the completion counter atomically
        int old = cm_.reordering_counters[wc.wr_id].fetch_sub(1);
        if (old < 1) {
          throw std::runtime_error("Completion counter underflow detected.");
        }
      }
    }
    
    /// Do a 64-bit swap over RDMA
    template <typename T>
    T AtomicSwap(rdma_ptr<T> ptr, uint64_t swap, MemoryPool &mp,
                 uint64_t hint = 0) {
      static_assert(sizeof(T) == 8);
      auto info = cm_.get_conn(ptr.raw(), 0); // [mfs] Using 0 for now
    
      // [esl] Getting the thread's index to determine it's owned flag
      uint64_t index_as_id = cm_.thread_ids.at(std::this_thread::get_id());
    
      auto alloc = mp.rdma_memory_.get();
      // [esl]  There is probably a better way to avoid allocating every time we
      //        do this call (maybe be preallocating the space thread_local)
      volatile uint64_t *prev_ = alloc->template allocateT<uint64_t>();
    
      ibv_sge sge{.addr = reinterpret_cast<uint64_t>(prev_),
                  .length = sizeof(uint64_t),
                  .lkey = mp.rdma_memory_->mr()->lkey};
    
      ibv_send_wr send_wr_{};
      send_wr_.wr_id = index_as_id;
      send_wr_.num_sge = 1;
      send_wr_.sg_list = &sge;
      send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
      send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
      send_wr_.wr.atomic.remote_addr = ptr.address();
      send_wr_.wr.atomic.rkey = info.rkey;
      send_wr_.wr.atomic.compare_add = hint;
      send_wr_.wr.atomic.swap = swap;
    
      while (true) {
        // set the counter to the number of work completions we expect
        *ack = 1;
        info.conn->send_onesided(&send_wr_);
    
        // Poll until we match on the condition
        ibv_wc wc;
        while (*ack != 0) {
          // int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
          int poll = info.conn->poll_cq(1, &wc);
          if (poll == 0 || (poll < 0 && errno == EAGAIN))
            continue;
          // Assert a good result
          if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
            REMUS_ASSERT(
                poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
                (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
          }
          int old = cm_.reordering_counters[wc.wr_id].fetch_sub(1);
          REMUS_ASSERT(old >= 1, "Broken synchronization");
        }
    
        if (*prev_ == send_wr_.wr.atomic.compare_add)
          break;
        send_wr_.wr.atomic.compare_add = *prev_;
      };
      T ret = T(*prev_);
      alloc->deallocateT((uint64_t *)prev_, 1);
      return ret;
    }
#endif

/// Common code for RDMA read
template <typename T>
void ReadInternal(rdma_ptr<T> ptr, size_t offset, size_t bytes,
                  size_t chunk_size, uint8_t *seg, int32_t rkey, int32_t lkey,
                  Connection *conn, std::atomic<int> *ack) {
  // [mfs]  We have support for rather sizeable chunks.  Do we really need to do
  //        explicit chunking?  Is that advantageous for performance?
  const int num_chunks =
      bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
  const size_t remainder = bytes % chunk_size;
  const bool is_multiple = remainder == 0;

  T *local = (T *)seg;
  // NB: This use of variable-length arrays is acceptable
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  ibv_sge sges[num_chunks];
  ibv_send_wr wrs[num_chunks];
#pragma clang diagnostic pop

  for (int i = 0; i < num_chunks; ++i) {
    auto chunk_offset = offset + i * chunk_size;
    sges[i].addr = reinterpret_cast<uint64_t>(local) + chunk_offset;
    if (is_multiple) {
      sges[i].length = chunk_size;
    } else {
      sges[i].length = (i == num_chunks - 1 ? remainder : chunk_size);
    }
    sges[i].lkey = lkey;

    wrs[i].wr_id = (uint64_t)ack;
    wrs[i].num_sge = 1;
    wrs[i].sg_list = &sges[i];
    wrs[i].opcode = IBV_WR_RDMA_READ;
    wrs[i].send_flags = IBV_SEND_FENCE;
    if (i == num_chunks - 1)
      wrs[i].send_flags |= IBV_SEND_SIGNALED;
    wrs[i].wr.rdma.remote_addr = ptr.address() + chunk_offset;
    wrs[i].wr.rdma.rkey = rkey;
    wrs[i].next = (i != num_chunks - 1 ? &wrs[i + 1] : nullptr);
  }

  // set the counter to the # of completions we expect, then make request
  *ack = 1;
  conn->send_onesided(wrs);

  // Poll until we match on the condition
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    // [mfs] Why doesn't write look for poll != 1, but this does?
    if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                   "ibv_poll_cq(): {} @ {}",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   format_rdma_ptr(ptr));
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }
}

/// Fast async read for single-threaded polling (no atomics needed)
template <typename T>
void ReadInternalAsyncFast(rdma_ptr<T> ptr, size_t offset, size_t bytes,
                           size_t chunk_size, uint8_t *seg, int32_t rkey, 
                           int32_t lkey, Connection *conn, int *pending) {
  const int num_chunks = bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
  const size_t remainder = bytes % chunk_size;
  const bool is_multiple = remainder == 0;

  T *local = (T *)seg;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  ibv_sge sges[num_chunks];
  ibv_send_wr wrs[num_chunks];
#pragma clang diagnostic pop

  for (int i = 0; i < num_chunks; ++i) {
    auto chunk_offset = offset + i * chunk_size;
    sges[i].addr = reinterpret_cast<uint64_t>(local) + chunk_offset;
    sges[i].length = is_multiple ? chunk_size : (i == num_chunks - 1 ? remainder : chunk_size);
    sges[i].lkey = lkey;

    wrs[i].wr_id = (uint64_t)pending;  // Store pointer to counter
    wrs[i].num_sge = 1;
    wrs[i].sg_list = &sges[i];
    wrs[i].opcode = IBV_WR_RDMA_READ;
    wrs[i].send_flags = IBV_SEND_FENCE;
    //Test with removed fence to see if it helps with latency
    // Only signal the last chunk
    if (i == num_chunks - 1)
      wrs[i].send_flags |= IBV_SEND_SIGNALED;
      
    wrs[i].wr.rdma.remote_addr = ptr.address() + chunk_offset;
    wrs[i].wr.rdma.rkey = rkey;
    wrs[i].next = (i != num_chunks - 1 ? &wrs[i + 1] : nullptr);
  }

  // Increment counter and post
  (*pending)++;
  conn->send_onesided(wrs);
}

/// Fast async CAS for single-threaded polling (no atomics needed)
template <typename T>
void CompareAndSwapAsyncFast(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap,
                              uint8_t *seg, int32_t rkey, int32_t lkey, 
                              Connection *conn, int *pending) {
  static_assert(sizeof(T) == 8);
  
  volatile uint64_t *prev_ = reinterpret_cast<volatile uint64_t *>(seg);
  
  ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(prev_);
  sge.length = sizeof(uint64_t);
  sge.lkey = lkey;
  
  ibv_send_wr send_wr_{};
  send_wr_.wr_id = (uint64_t)pending;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.atomic.remote_addr = ptr.address();
  send_wr_.wr.atomic.rkey = rkey;
  send_wr_.wr.atomic.compare_add = expected;
  send_wr_.wr.atomic.swap = swap;
  
  // Increment counter and post
  (*pending)++;
  conn->send_onesided(&send_wr_);
}

/// Fast async write for single-threaded polling (no atomics needed)
template <typename T>
void WriteAsyncFast(rdma_ptr<T> ptr, const T &val, uint8_t *seg, int32_t rkey,
                    int32_t lkey, Connection *conn, int *pending) {
  T *local = (T *)seg;
  *local = val;
  
  ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(local);
  sge.length = sizeof(T);
  sge.lkey = lkey;

  ibv_send_wr send_wr_{};
  send_wr_.wr_id = (uint64_t)pending;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_RDMA_WRITE;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.rdma.remote_addr = ptr.address();
  send_wr_.wr.rdma.rkey = rkey;

  (*pending)++;
  conn->send_onesided(&send_wr_);
}

/// Helper to poll completions from a specific connection
/// Returns number of completions processed
inline int PollCompletionsFast(Connection* conn, int max_polls = 16) {
  ibv_wc wc[16];
  int got = conn->poll_cq(max_polls, wc);
  
  if (got > 0) {
    for (int i = 0; i < got; ++i) {
      if (wc[i].status == IBV_WC_SUCCESS) {
        int* counter = reinterpret_cast<int*>(wc[i].wr_id);
        (*counter)--;  // Simple decrement, no atomic needed
      } else {
        REMUS_FATAL("RDMA operation failed: {}", ibv_wc_status_str(wc[i].status));
      }
    }
  }
  
  return got;
}

/// Internal utility to post a read but NOT wait for completion
template <typename T>
void ReadInternalAsync(rdma_ptr<T> ptr, size_t offset, size_t bytes,
                       size_t chunk_size, uint8_t *seg, int32_t rkey, int32_t lkey,
                       Connection *conn, std::atomic<int> *ack) {
  // 1. Calculate chunks (same as synchronous version)
  const int num_chunks = bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
  const size_t remainder = bytes % chunk_size;
  const bool is_multiple = remainder == 0;

  T *local = (T *)seg;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  ibv_sge sges[num_chunks];
  ibv_send_wr wrs[num_chunks];
#pragma clang diagnostic pop

  for (int i = 0; i < num_chunks; ++i) {
    auto chunk_offset = offset + i * chunk_size;
    sges[i].addr = reinterpret_cast<uint64_t>(local) + chunk_offset;
    sges[i].length = is_multiple ? chunk_size : (i == num_chunks - 1 ? remainder : chunk_size);
    sges[i].lkey = lkey;

    // We pass the atomic pointer as the wr_id so the poller knows which counter to decrement
    wrs[i].wr_id = (uint64_t)ack;
    wrs[i].num_sge = 1;
    wrs[i].sg_list = &sges[i];
    wrs[i].opcode = IBV_WR_RDMA_READ;
    wrs[i].send_flags = IBV_SEND_FENCE;
    
    // Only signal the LAST chunk to avoid spamming the Completion Queue
    if (i == num_chunks - 1)
      wrs[i].send_flags |= IBV_SEND_SIGNALED;
      
    wrs[i].wr.rdma.remote_addr = ptr.address() + chunk_offset;
    wrs[i].wr.rdma.rkey = rkey;
    wrs[i].next = (i != num_chunks - 1 ? &wrs[i + 1] : nullptr);
  }

  // 2. Increment the counter BEFORE sending.
  // Note: If you are doing multiple Async calls, you should fetch_add(1)
  // instead of a direct assignment to allow multiple outstanding calls.
  ack->fetch_add(1); 

  // 3. Post to the NIC and return immediately
  conn->send_onesided(wrs);
}

/// Async CAS - post but don't wait for completion
template <typename T>
void CompareAndSwapAsync(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap,
                         uint8_t *seg, int32_t rkey, int32_t lkey, 
                         Connection *conn, std::atomic<int> *ack) {
  static_assert(sizeof(T) == 8);
  
  volatile uint64_t *prev_ = reinterpret_cast<volatile uint64_t *>(seg);
  
  ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(prev_);
  sge.length = sizeof(uint64_t);
  sge.lkey = lkey;
  
  ibv_send_wr send_wr_{};
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.atomic.remote_addr = ptr.address();
  send_wr_.wr.atomic.rkey = rkey;
  send_wr_.wr.atomic.compare_add = expected;
  send_wr_.wr.atomic.swap = swap;
  
  // Increment counter and post (don't poll)
  ack->fetch_add(1);
  conn->send_onesided(&send_wr_);
}

/// Batch async read - posts multiple reads without polling
/// Returns the number of operations posted for later polling
template <typename T>
size_t BatchReadAsync(const std::vector<rdma_ptr<T>>& ptrs, 
                      const std::vector<uint8_t*>& segs,
                      const std::vector<int32_t>& rkeys,
                      int32_t lkey,
                      Connection* conn,
                      uint64_t wr_id_base = 0) {
  size_t num_ops = ptrs.size();
  
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  ibv_sge sges[num_ops];
  ibv_send_wr wrs[num_ops];
#pragma clang diagnostic pop

  for (size_t i = 0; i < num_ops; ++i) {
    sges[i].addr = reinterpret_cast<uint64_t>(segs[i]);
    sges[i].length = sizeof(T);
    sges[i].lkey = lkey;

    wrs[i].wr_id = wr_id_base + i;  // Simple identifier, not atomic pointer
    wrs[i].num_sge = 1;
    wrs[i].sg_list = &sges[i];
    wrs[i].opcode = IBV_WR_RDMA_READ;
    wrs[i].send_flags = IBV_SEND_FENCE | IBV_SEND_SIGNALED;  // Signal each one
    wrs[i].wr.rdma.remote_addr = ptrs[i].address();
    wrs[i].wr.rdma.rkey = rkeys[i];
    wrs[i].next = (i != num_ops - 1 ? &wrs[i + 1] : nullptr);
  }

  conn->send_onesided(wrs);
  return num_ops;
}

/// Batch async CAS - posts multiple CAS without polling
template <typename T>
size_t BatchCasAsync(const std::vector<rdma_ptr<T>>& ptrs,
                     const std::vector<uint64_t>& expected,
                     const std::vector<uint64_t>& swaps,
                     const std::vector<uint8_t*>& segs,
                     const std::vector<int32_t>& rkeys,
                     int32_t lkey,
                     Connection* conn,
                     uint64_t wr_id_base = 0) {
  static_assert(sizeof(T) == 8);
  size_t num_ops = ptrs.size();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  ibv_sge sges[num_ops];
  ibv_send_wr wrs[num_ops];
#pragma clang diagnostic pop

  for (size_t i = 0; i < num_ops; ++i) {
    volatile uint64_t* prev = reinterpret_cast<volatile uint64_t*>(segs[i]);
    
    sges[i].addr = reinterpret_cast<uint64_t>(prev);
    sges[i].length = sizeof(uint64_t);
    sges[i].lkey = lkey;

    wrs[i].wr_id = wr_id_base + i;  // Simple identifier
    wrs[i].num_sge = 1;
    wrs[i].sg_list = &sges[i];
    wrs[i].opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wrs[i].send_flags = IBV_SEND_FENCE | IBV_SEND_SIGNALED;
    wrs[i].wr.atomic.remote_addr = ptrs[i].address();
    wrs[i].wr.atomic.rkey = rkeys[i];
    wrs[i].wr.atomic.compare_add = expected[i];
    wrs[i].wr.atomic.swap = swaps[i];
    wrs[i].next = (i != num_ops - 1 ? &wrs[i + 1] : nullptr);
  }

  conn->send_onesided(wrs);
  return num_ops;
}

/// Poll for exactly N completions from a connection
/// Returns number of completions received (should equal expected)
inline size_t PollCompletions(Connection* conn, size_t expected) {
  static constexpr int MAX_BATCH = 16;
  ibv_wc wc[MAX_BATCH];
  size_t received = 0;
  
  while (received < expected) {
    int got = conn->poll_cq(MAX_BATCH, wc);
    if (got > 0) {
      for (int i = 0; i < got; ++i) {
        if (wc[i].status == IBV_WC_SUCCESS) {
          received++;
        }
      }
    }
  }
  
  return received;
}

#if 0
    template <typename T>
    std::vector<T> BatchRead(std::vector<rdma_ptr<T>> &group, rdma_ptr<T> prealloc,
                             MemoryPool &mp) {
      // Get thread ID for completion tracking
      auto thread_id = std::this_thread::get_id();
      auto thread_it = cm_.thread_ids.find(thread_id);
      assert(thread_it != cm_.thread_ids.end());
      uint64_t index_as_id = thread_it->second;
      // Calculate the number of work requests and the number of requests per
      // batch
      const size_t num_requests = group.size();
      const size_t num_wrs_per_batch = kMaxWr;
    
      // Get connection info for the remote memory pool
      // [mfs] I don't understand the role of group yet.
      auto info = cm_.get_conn(group[0].raw(), 0); // [mfs] Using 0 for now
    
      // Allocate local memory if prealloc is nullptr
      T *local;
      bool allocated = false;
      auto alloc = mp.rdma_memory_.get();
      if (prealloc == nullptr) {
        local = alloc->template allocateT<T>(num_wrs_per_batch);
        allocated = true;
      } else {
        local = std::to_address(prealloc);
      }
    
      // Create work requests and SGEs for each remote address
      ibv_send_wr wrs[num_wrs_per_batch];
      ibv_sge sges[num_wrs_per_batch];
      // Construct return vector from local buffer
      std::vector<T> result;
      result.reserve(num_requests);
      // Set up work requests - one WR per remote address in batches
      for (size_t i = 0;
           i < (num_requests + num_wrs_per_batch - 1) / num_wrs_per_batch; i++) {
        size_t j = 0;
        for (; j < num_wrs_per_batch && i * num_wrs_per_batch + j < num_requests;
             j++) {
          sges[j].addr = reinterpret_cast<uint64_t>(local) + i * sizeof(T);
          sges[j].length = sizeof(T);
          sges[j].lkey = mp.rdma_memory_->mr()->lkey;
    
          // Set up work request
          wrs[j].wr_id = index_as_id;
          wrs[j].opcode = IBV_WR_RDMA_READ;
          wrs[j].send_flags = IBV_SEND_FENCE;
          if (j == num_wrs_per_batch - 1 ||
              i * num_wrs_per_batch + j == num_requests - 1)
            wrs[j].send_flags |= IBV_SEND_SIGNALED;
          wrs[j].num_sge = 1;
          wrs[j].sg_list = &sges[j];
          wrs[j].next = (j < num_wrs_per_batch - 1 &&
                                 i * num_wrs_per_batch + j < num_requests - 1
                             ? &wrs[j + 1]
                             : nullptr);
          wrs[j].wr.rdma.remote_addr = group[i * num_wrs_per_batch + j].address();
          wrs[j].wr.rdma.rkey = info.rkey;
        }
    
        // Set the counter to the expected completion (one per batch)
        *ack = 1;
        info.conn->send_onesided(wrs);
    
        // Poll until the work completion is received
        ibv_wc wc;
        while (*ack != 0) {
          int poll = info.conn->poll_cq(1, &wc);
          if (poll == 0 || (poll < 0 && errno == EAGAIN))
            continue;
          if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
            REMUS_ASSERT(
                poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
                (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                format_rdma_ptr(group[0]));
          }
          int old = cm_.reordering_counters[wc.wr_id].fetch_sub(1);
          REMUS_ASSERT(old >= 1, "Broken synchronization");
        }
    
        for (size_t k = 0; k < j; k++) {
          result.push_back(local[k]);
        }
      }
    
      // Cleanup if we allocated memory
      if (allocated) {
        REMUS_DEBUG("false", num_wrs_per_batch);
        alloc->template deallocateT<T>(local, num_wrs_per_batch);
      }
      return result;
    }
    
    /// to simplify the logic and allow user do optimization, all group's remote
    /// pointers should be on the same node
    ///
    /// @tparam T
    /// @param group
    template <typename T>
    void BatchWrite(std::vector<rdma_ptr<T>> &group, rdma_ptr<T> local_ptr,
                    bool use_one, MemoryPool &mp) {
      // Get thread ID for completion tracking
      auto thread_id = std::this_thread::get_id();
      auto thread_it = cm_.thread_ids.find(thread_id);
      assert(thread_it != cm_.thread_ids.end());
      uint64_t index_as_id = thread_it->second;
    
      // Get connection info for the remote memory pool using the first remote
      // pointer
      //
      // [mfs] I don't understand the role of group yet
      auto info = cm_.get_conn(group[0].raw(), 0); // [mfs] Using 0 for now
    
      const size_t num_requests = group.size();
      const size_t num_wrs_per_batch = kMaxWr;
    
      // Create work requests and SGEs for each write operation
      ibv_send_wr wrs[num_wrs_per_batch];
      ibv_sge sges[num_wrs_per_batch];
    
      // Set up work requests in batches similar to BatchRead
      for (size_t i = 0;
           i < (num_requests + num_wrs_per_batch - 1) / num_wrs_per_batch; i++) {
        size_t j = 0;
        for (; j < num_wrs_per_batch && i * num_wrs_per_batch + j < num_requests;
             j++) {
          auto remote_ptr = group[i * num_wrs_per_batch + j];
          if (use_one) {
            sges[j].addr = reinterpret_cast<uint64_t>(std::to_address(local_ptr));
          } else {
            sges[j].addr = reinterpret_cast<uint64_t>(std::to_address(local_ptr) +
                                                      (i * num_wrs_per_batch + j) *
                                                          sizeof(uint64_t));
          }
          sges[j].length = sizeof(T);
          sges[j].lkey = mp.rdma_memory_->mr()->lkey;
    
          wrs[j].wr_id = index_as_id;
          wrs[j].opcode = IBV_WR_RDMA_WRITE;
          wrs[j].send_flags = IBV_SEND_FENCE;
          if (j == num_wrs_per_batch - 1 ||
              i * num_wrs_per_batch + j == num_requests - 1)
            wrs[j].send_flags |= IBV_SEND_SIGNALED;
          wrs[j].num_sge = 1;
          wrs[j].sg_list = &sges[j];
          wrs[j].next = (j < num_wrs_per_batch - 1 &&
                                 i * num_wrs_per_batch + j < num_requests - 1
                             ? &wrs[j + 1]
                             : nullptr);
          wrs[j].wr.rdma.remote_addr = remote_ptr.address();
          wrs[j].wr.rdma.rkey = info.rkey;
        }
    
        // Post the batch: set the counter to 1 since only the last WR is signaled
        *ack = 1;
        info.conn->send_onesided(wrs);
    
        // Poll until the work completion is received
        ibv_wc wc;
        while (*ack != 0) {
          int poll = info.conn->poll_cq(1, &wc);
          if (poll == 0 || (poll < 0 && errno == EAGAIN))
            continue;
          if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
            REMUS_ASSERT(
                poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
                (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                format_rdma_ptr(group[0]));
          }
          int old = cm_.reordering_counters[wc.wr_id].fetch_sub(1);
          REMUS_ASSERT(old >= 1, "Broken synchronization");
        }
      }
    }
#endif
} // namespace remus::internal
