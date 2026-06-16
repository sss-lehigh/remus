#pragma once

#include <array>
#include <atomic>
#include <list>
#include <memory>
#include <thread>
#include <unordered_map>

#include "cfg.h"
#include "compute_node.h"
#include "connection.h"
#include "logging.h"
#include "mn_alloc_pol.h"
#include "qp_sched_pol.h"
#include "rdma_ops.h"
#include "rdma_ptr.h"
#include "util.h"

namespace remus::internal {
/// A simple distributed bump allocator.  This is a size-preserving allocator,
/// because it does not coalesce adjacent free blocks into larger free blocks.
/// It is somewhat amenable to type-preservation, because there is some room in
/// object headers that a synchronization mechanism can use for metadata.
class BumpAllocator {
  const uint64_t seg_size_; // Size of Segments at each MemoryNode

  /// Size-segregated collections of free memory blocks
  std::unordered_map<uint64_t, std::vector<uint64_t>> freelists_;

  /// A list of "really big" free blocks of memory, as <size, address> pairs
  std::vector<std::pair<uint64_t, uintptr_t>> free_blocks_;

  /// Requests at or below this size will be rounded to the nearest 64 bytes
  static constexpr uint64_t ALLOC_SMALL_THRESH = 1024;

  /// Requests at or below this size will be rounded to the nearest 1024 bytes
  static constexpr uint64_t ALLOC_MED_THRESH = 8192;

  /// When we allocate a block of memory, it will have this header
  struct header_t {
    std::atomic<uint64_t> size_;    // The size of the block: write-once
    std::atomic<uint64_t> padding_; // Padding to 16B; can be used as a lock
  };

  /// Compute the slabclass for a given size.  Assumes size includes
  /// HEADER_SIZE.
  inline uint64_t calculate_slabclass(uint64_t size) {
    // [mfs]  Check this math... do we need more parenthesis to get it right?
    //        Are we sure it's doing what we want?  Why not use roundup()
    //        instead?
    if (size <= ALLOC_SMALL_THRESH) {
      return ((size + 63) >> 6 << 6);
    } else if (size <= ALLOC_MED_THRESH) {
      return ((size + 1023) >> 10 << 10);
    } else {
      return ((size + 63) >> 6 << 6);
    }
  }

public:
  /// The size of the header for allocated memory blocks
  static constexpr uint64_t HEADER_SIZE = sizeof(header_t);

  /// The policy for deciding with Segment to use when performing an Alloc
  internal::MnAllocPolicy mn_alloc_pol_;

  /// Compute the desired size for an allocation
  ///
  /// @warning  Sizeof(T) is insufficient for variable-sized objects, such as
  ///           skip list nodes!
  ///
  /// @tparam T The type of the item to allocate, used for getting its size
  /// @param n  The number of elements of size T
  ///
  /// @return A size to allocate.  Guaranteed to be at least n *sizeof(T) +
  ///         HEADER_SIZE.  Typically larger (i.e., rounded up).
  template <typename T> uint64_t compute_size(std::size_t n) {
    return calculate_slabclass(sizeof(T) * n + HEADER_SIZE);
  }

  /// Try to get a fresh region of memory from one of the slabs
  ///
  /// @warning  Out of memory errors can manifest as infinite loops, and some
  ///           policies will lead to out of memory errors even when there is
  ///           available memory, if it's in Segments that the policy won't
  ///           access.
  ///
  /// @param size The desired size of the region.  This should be computed via
  ///             compute_size(), so it includes the header
  /// @param seg_locator  A lambda for getting the base of a Segment
  /// @param hint_locator A lambda for getting the hint for a Segment
  /// @param faa          A lambda for doing an RDMA fetch-and-add
  /// @param writer       A lambda for doing an RDMA write
  ///
  /// @return A region of memory
  uintptr_t try_allocate_global(
      std::size_t size, std::function<uint64_t(uint64_t, uint64_t)> seg_locator,
      std::function<std::atomic<uint64_t> &(uint64_t, uint64_t)> hint_locator,
      std::function<uint64_t(rdma_ptr<uint64_t>, uint64_t)> faa,
      std::function<void(rdma_ptr<uint64_t>, uint64_t)> writer) {
    while (true) {
      // Get a MemoryNode and Segment on which to try to allocate
      //
      // NB:  On subsequent iterations of the loop, if these values do not
      //      change, we'll have an infinite loop.
      auto [mn_id, seg_id] = mn_alloc_pol_.get_mn_seg();
      auto base = seg_locator(mn_id, seg_id);
      // Since our bump allocator doesn't coalesce, if this machine has ever
      // seen its counter exceed what we need for this alloc to work, then try
      // to get another mn_id/seg_id.
      auto &hint = hint_locator(mn_id, seg_id);
      if (hint + size > seg_size_) {
        continue;
      }
      // Try to FAA to allocate
      //
      // NB:  The above check was just a hint, so failure is still possible
      auto bump_counter = rdma_ptr<uint64_t>(
          base + offsetof(internal::ControlBlock, allocated_));
      uint64_t offset = faa(bump_counter, size);
      if (offset + size > seg_size_) {
        // NB: Due to concurrency, there's no decrementing to try to recover!
        continue;
      }
      // Update the hint unless someone else's subsequent update finishes first
      uint64_t curr_hint = hint;
      uint64_t new_hint = offset + size;
      do {
      } while ((curr_hint <= new_hint) &&
               !hint.compare_exchange_strong(curr_hint, new_hint));
      // This is a fresh allocation, so set the size and zero the padding
      uint64_t ptr = base + offset;
      writer(remus::rdma_ptr<uint64_t>(ptr + offsetof(header_t, size_)), size);
      writer(remus::rdma_ptr<uint64_t>(ptr + offsetof(header_t, padding_)), 0);
      return ptr + HEADER_SIZE;
    }
    REMUS_FATAL("Out of memory"); // This is actually unreachable
  }

  /// Try to allocate from a freelist
  ///
  /// @param size The desired size of the region.  This should be computed via
  ///             compute_size(), so it includes the header
  ///
  /// @return A raw pointer (uintptr_t) or none, if no freelist can satisfy the
  ///         request.
  std::optional<uintptr_t> try_allocate_local(std::size_t size) {
    // If we can satisfy it via the freelist, do so, and we're done
    ///
    /// [mfs] The common case is not "big allocations", so we should probably
    ///       have the 'else' code come first?
    if (size > ALLOC_MED_THRESH) {
      // For large allocations (>8192), use best-fit allocation from the
      // best_fit_freelist Find the first chunk in the freelist that can fit
      // this allocation
      //
      // [mfs]  This is first-fit, not best-fit.  That's probably unwise?  Also,
      //        reverse iteration is probably smarter.
      auto it = std::find_if(
          free_blocks_.begin(), free_blocks_.end(),
          [size](const auto &chunk) { return chunk.first >= size; });
      if (it != free_blocks_.end()) {
        uint64_t ptr = it->second;
        // [mfs]  Erasing from a vector is costly.  Consider using a deque?  Or
        //        does reverse iteration reduce the risk?
        free_blocks_.erase(it);
        return ptr + HEADER_SIZE;
      }
    } else {
      auto &freelist = freelists_[size];
      if (!freelist.empty()) {
        auto ptr = freelist.back();
        freelist.pop_back();
        return ptr + HEADER_SIZE;
      }
    }
    return {};
  }

  /// Reclaim by putting the pointer into the appropriate freelist
  ///
  /// @warning This assumes the caller read the size from ptr
  ///
  /// @tparam T The type of object referenced.  Immaterial to this function
  ///
  /// @param ptr    An rdma_ptr
  /// @param size   The size that was read from ptr / the allocation size (not
  ///               sizeof(T))
  template <typename T> void reclaim(rdma_ptr<T> ptr, uint64_t size) {
    uint64_t slabclass = calculate_slabclass(size);
    // [mfs] Again, small blocks are the common case, so they should come first
    if (slabclass > ALLOC_MED_THRESH) {
      free_blocks_.push_back({slabclass, ptr.raw() - HEADER_SIZE});
    } else {
      // [mfs]  See constructor.  I think we pre-insert a vector, so .end() is
      //        not possible?
      if (freelists_.find(slabclass) == freelists_.end()) {
        freelists_[slabclass] = std::vector<uint64_t>();
      }
      freelists_[slabclass].push_back(ptr.raw() - HEADER_SIZE);
    }
  }

  /// Construct a simple Bump allocator
  ///
  /// @param args The arguments to the program
  BumpAllocator(std::shared_ptr<ArgMap> args)
      : seg_size_(1ULL << args->uget(remus::SEG_SIZE)), mn_alloc_pol_(args) {
    // initialize freelists
    //
    // [mfs]  Reclaim() assumes they're not initialized, and initializes them
    //        lazily, but try_alloc_local() assumes they are initialized.
    //        Something is out of sync.
    for (uint64_t i = 64; i < ALLOC_SMALL_THRESH; i += 64) {
      freelists_[i] = std::vector<uint64_t>();
    }
    for (uint64_t i = 1024; i <= ALLOC_MED_THRESH; i += 1024) {
      freelists_[i] = std::vector<uint64_t>();
    }
  }
};
} // namespace remus::internal

namespace remus {
/// ComputeThread is a per-node object that, once configured, provides all of
/// the underlying features needed by threads wishing to interact with RDMA
/// memory.
///
/// [mfs] Need to implement the metrics
///
/// [mfs] Check destructors and documentation
class ComputeThread {
protected:
  uint64_t node_id;    // The ComputeNode id
  uint64_t id_;        // This thread's Id
  uint8_t *seg_slice_; // This thread's memory region for doing one-sided ops
  std::shared_ptr<ComputeNode> compute_node_; // The ComputeNode
  std::shared_ptr<ArgMap> args_; // The command-line args to the program

  /// Counters for receiving ibv completion events
  ///
  /// [mfs] A vector is overkill until we support async one-sided ops
  std::vector<std::atomic<int>> counters_;

  /// The various events that we count
  enum RDMA_METRICS {
    NUM_READ = 0,
    NUM_WRITE = 1,
    NUM_CAS = 2,
    NUM_FAA = 3,
    NUM_SWAP = 4,
    READ_BYTES = 5,
    WRITE_BYTES = 6,
    COUNT = 7,
  };

  /// A count of metrics.  For now, we keep it really simple, but we probably
  /// want histograms and other fine-grained details at some point.
  ///
  /// [mfs] This needs to be initialized and used.
  ///
  /// [mfs] Would an unordered map be easier in the long run?  IDK.
  std::vector<uint64_t> metrics_;

  /// The policy for deciding which QP to use when connecting to a MemoryNode
  internal::QpSchedPolicy qp_sched_pol_;

  internal::BumpAllocator allocator; // The allocator
public:
  /// Construct a ComputeThread
  ///
  /// @param id   The thread's zero-based numerical Id
  /// @param cn   The ComputeNode context for this machine
  /// @param args The command-line arguments to the program
  explicit ComputeThread(uint64_t id, std::shared_ptr<ComputeNode> cn,
                         std::shared_ptr<ArgMap> args)
      : node_id(id), compute_node_(cn), args_(args),
        counters_(args->uget(remus::CN_THREAD_MSGS)), qp_sched_pol_(args),
        allocator(args) {
    // [mfs]  This would be much simpler if we could extract id_ from an
    //        initializer.  Consider switching to a factory?
    auto registration = compute_node_->register_thread();
    id_ = registration.first;
    seg_slice_ = registration.second;
    REMUS_INFO("Created thread #{}", id_);

    // Select the scheduling policies to use
    qp_sched_pol_.set_policy(
        internal::QpSchedPolicy::to_policy(args_->sget(QP_SCHED_POL)), id_);
    allocator.mn_alloc_pol_.set_policy(
        internal::MnAllocPolicy::to_policy(args->sget(ALLOC_POL)), args_, id_);
  }

  /// [mfs] Make sure we don't need a custom destructor?
  ~ComputeThread() { REMUS_INFO("Terminating thread #{}", id_); }

  uint8_t* get_seg_slice() { return seg_slice_; }

  uint64_t get_tid() { return id_; }

   // Helper for getting connection lane index
  uint32_t get_lane_idx(uint16_t node_id) {
    return qp_sched_pol_.get_lane_idx(node_id);
  }
  
  // Helper for getting connection info
  struct ConnInfo {
    internal::Connection* conn;
    uint32_t lkey;
    uint32_t rkey;
  };
  
  ConnInfo get_conn_info(uint64_t ptr_raw, uint32_t conn_idx) {
    auto& ci = compute_node_->get_conn(ptr_raw, conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr_raw);
    return {ci.conn_.get(), ci.lkey, rkey};
  }

  /// Write to an RDMA heap
  template <typename T> void Write(rdma_ptr<T> ptr, const T &val) {
    // Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    internal::Write(ptr, val, seg_slice_, rkey, ci.lkey, ci.conn_.get(),
                    &counters_[0]);
  }

  /// Perform a CAS on the RDMA heap
  template <typename T>
  T CompareAndSwap(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    // Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    return internal::CompareAndSwap(ptr, expected, swap, seg_slice_, rkey,
                                    ci.lkey, ci.conn_.get(), &counters_[0]);
  }

  /// Perform a FetchAndAdd on the RDMA heap
  template <typename T> T FetchAndAdd(rdma_ptr<T> ptr, uint64_t add) {
    // Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    return internal::FetchAndAdd(ptr, add, seg_slice_, rkey, ci.lkey,
                                 ci.conn_.get(), &counters_[0]);
  }

#if 1
  /// Write to an RDMA heap
  ///
  /// [mfs] This might need to pass the ComputeThread, to get the right qp
  ///       when there's >1.  Also, we're going to need the right rkey.
  template <typename T>
  void ExtendedWrite(rdma_ptr<T> ptr, rdma_ptr<T> prealloc, uint64_t size) {
    ExtendedWrite(ptr, prealloc, size, seg_slice_);
  }

  /// Perform a CAS on the RDMA heap that loops until it succeeds
  ///
  /// [mfs] This might need to pass the ComputeThread, to get the right qp
  ///       when there's >1.  Also, we're going to need the right rkey.
  template <typename T>
  T AtomicSwap(rdma_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    return AtomicSwap<T>(ptr, swap, hint, seg_slice_);
  }

  /// Read a variable-sized object from the RDMA heap
  ///
  /// [mfs] This might need to pass the ComputeThread, to get the right qp
  ///       when there's >1.  Also, we're going to need the right rkey.
  template <typename T>
  rdma_ptr<T> ExtendedRead(rdma_ptr<T> ptr, int size, rdma_ptr<T> local_dest = nullptr) {
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());

    size_t total_bytes = sizeof(T) * size;

    if (local_dest != nullptr) {
      internal::ReadInternal<T>(
          ptr, 0, total_bytes, total_bytes,
          reinterpret_cast<uint8_t*>(local_dest.raw()), // Destination
          rkey,
          ci.lkey,
          ci.conn_.get(),
          &counters_[0]);
      return local_dest;
    } else {
      internal::ReadInternal<T>(
          ptr, 0, total_bytes, total_bytes, 
          seg_slice_, // Destination
          rkey, 
          ci.lkey,
          ci.conn_.get(), 
          &counters_[0]
      );
      return rdma_ptr<T>(reinterpret_cast<uintptr_t>(seg_slice_));
    }
}

/// Async CAS that returns immediately without polling
template <typename T>
void CompareAndSwapAsync(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    CompareAndSwapAsync(ptr, expected, swap, &counters_[0]);
}

/// Async CAS that tracks completion using a custom atomic integer
template <typename T>
void CompareAndSwapAsync(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap, std::atomic<int>* custom_ack) {
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    
    internal::CompareAndSwapAsync(ptr, expected, swap, seg_slice_, rkey,
                                  ci.lkey, ci.conn_.get(), custom_ack);
}


#endif

  /// Read a fixed-sized object from the RDMA heap
  template <typename T> T Read(rdma_ptr<T> ptr) {
    /// Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    internal::ReadInternal(ptr, 0, sizeof(T), sizeof(T), seg_slice_, rkey,
                           ci.lkey, ci.conn_.get(), &counters_[0]);
    return *(T *)seg_slice_;
  }


#if 0

  /// [mfs] This might need to pass the ComputeThread, to get the right qp
  ///       when there's >1.  Also, we're going to need the right rkey.
  template <typename T>
  std::vector<T> BatchRead(std::vector<rdma_ptr<T>> &group,
                           rdma_ptr<T> prealloc) {
    return BatchRead(group, prealloc);
  }

  /// [mfs] This might need to pass the ComputeThread, to get the right qp
  ///       when there's >1.  Also, we're going to need the right rkey.
  template <typename T>
  void BatchWrite(std::vector<rdma_ptr<T>> &group, rdma_ptr<T> local_ptr,
                  bool use_one) {
    return BatchWrite(group, local_ptr, use_one);
  }
#endif

  /// Determine if a rdma_ptr is local to the machine
  ///
  /// [mfs] We need to figure out if we still need this
  template <class T> bool is_local(rdma_ptr<T> ptr) {
    return ptr.id() == node_id;
  }

  /// Arrive at the global barrier in Segment 0 of MemoryNode 0
  void arrive_control_barrier(int total_threads) {
    auto barrier =
        rdma_ptr<uint64_t>(compute_node_->get_seg_start(0, 0) +
                           offsetof(internal::ControlBlock, barrier_));
    // Arrive via a simple increment, use low bit to get the next "sense"
    auto was = this->FetchAndAdd(barrier, 2);
    uint64_t new_sense = 1 - (was & 1);
    // If the preceding FAA was the last one, reset the barrier, otherwise spin
    if ((was >> 1) == (total_threads - 1)) {
      this->Write(barrier, new_sense);
      return;
    }
    while ((this->Read(barrier) & 1) != new_sense) {
    }
  }

  /// Allocate a region of n * sizeof(T) bytes.  This will use the memory
  /// allocation policy to choose a memory node from which to allocate.
  template <typename T> rdma_ptr<T> allocate(std::size_t n = 1) {
    auto size = allocator.compute_size<T>(n);
    auto local = allocator.try_allocate_local(size);
    if (local.has_value())
      return rdma_ptr<T>(local.value());
    // [mfs]  The use of four lambdas here is really icky, this should be
    //        refactored at some point.
    auto global = allocator.try_allocate_global(
        size,
        [&](uint64_t mn_id, uint64_t seg_id) {
          return compute_node_->get_seg_start(mn_id, seg_id);
        },
        [&](uint64_t mn_id, uint64_t seg_id) -> std::atomic<uint64_t> & {
          return compute_node_->get_alloc_hint(mn_id, seg_id);
        },
        [&](rdma_ptr<uint64_t> ptr, uint64_t val) {
          return FetchAndAdd(ptr, val);
        },
        [&](rdma_ptr<uint64_t> ptr, uint64_t val) { return Write(ptr, val); });
    return rdma_ptr<T>(global);
  }

  /// Deallocate a region of memory, so it can be used again.
  template <typename T> void deallocate(rdma_ptr<T> ptr) {
    // REMUS_INFO("Deallocating ptr: {}", ptr.raw());
    auto size = Read<uint64_t>(remus::rdma_ptr<uint64_t>(
        ptr.raw() - internal::BumpAllocator::HEADER_SIZE));
    allocator.reclaim(ptr, size);
  }

  /// Set the root pointer in MemoryNode 0, Segment 0, to `root`
  ///
  /// [mfs] Should we pass in the memory node and segment id?
  template <typename T> void set_root(rdma_ptr<T> root) {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    Write(root_ptr, root.raw());
  }

  /// Read the root pointer in MemoryNode 0, Segment 0
  ///
  /// [mfs] Should we pass in the memory node and segment id?
  template <typename T> rdma_ptr<T> get_root() {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    return rdma_ptr<T>(Read<uint64_t>(root_ptr));
  }

//   /// Polls the completion queue once. If a completion is found, 
//   /// the wr_id (which is our atomic counter) is decremented.
//   void PollOnce(uint16_t node_id, uint64_t lane_idx) {
//       ibv_wc wc;
//       // Now this call will succeed
//       auto &ci = compute_node_->get_conn_by_index(node_id, lane_idx); 
      
//       int poll = ci.conn_->poll_cq(1, &wc);
//       if (poll > 0) {
//           if (wc.status != IBV_WC_SUCCESS) {
//               REMUS_FATAL("RDMA Op Failed: {}", ibv_wc_status_str(wc.status));
//           }
//           std::atomic<int>* counter = reinterpret_cast<std::atomic<int>*>(wc.wr_id);
//           counter->fetch_sub(1);
//       }
//   }

//   /// Helper to wait for a specific counter to hit zero
//   void Wait(std::atomic<int>* counter, Connection* conn) {
//     while (counter->load(std::memory_order_acquire) > 0) {
//         ibv_wc wc;
//         int poll = conn->poll_cq(1, &wc);
//         if (poll > 0) {
//             std::atomic<int>* ack = reinterpret_cast<std::atomic<int>*>(wc.wr_id);
//             ack->fetch_sub(1);
//         }
//     }
// }

};
} // namespace remus
