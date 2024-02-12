#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <experimental/memory_resource>
#include <fstream>
#include <infiniband/verbs.h>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <variant>
#include <vector>

#include <protos/metrics.pb.h> // TODO should be a part of matrics
#include <protos/rdma.pb.h> // TODO should be replaced with a JSON object

#include <spdlog/fmt/fmt.h> // [mfs] Used in remote_ptr... factor away?
#include <vector>

#include "rome/util/status.h"
#include "rome/logging/logging.h"
#include "rome/metrics/summary.h"

#include "connection.h"
#include "peer.h"
#include "remote_ptr.h"
#include "segment.h"

#define THREAD_MAX 10

// [mfs]  The entire dependency on fmt boils down to this template, used in one
//        assertion?
template <typename T> struct fmt::formatter<::rome::rdma::remote_ptr<T>> {
  typedef ::rome::rdma::remote_ptr<T> remote_ptr;
  constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
    return ctx.end();
  }

  template <typename FormatContext>
  auto format(const remote_ptr &input, FormatContext &ctx)
      -> decltype(ctx.out()) {
    return format_to(ctx.out(), "(id={}, address=0x{:x})", input.id(),
                     input.address());
  }
};

template <typename T> struct std::hash<rome::rdma::remote_ptr<T>> {
  std::size_t operator()(const rome::rdma::remote_ptr<T> &ptr) const {
    return std::hash<uint64_t>()(static_cast<uint64_t>(ptr));
  }
};

namespace rome::rdma::internal {

/// TODO: Document this
class MemoryPool {

  /// Specialization of a `memory_resource` that wraps RDMA accessible memory.
  ///
  /// TODO: This is really just a freelist-based allocator without true
  ///       recycling.  Can it be simplified?  Or could we just use a bona fide
  ///       std::memory_resource?
  class rdma_memory_resource : public std::experimental::pmr::memory_resource {
    rdma_memory_resource(const rdma_memory_resource &) = delete;
    rdma_memory_resource &operator=(const rdma_memory_resource &) = delete;

    static constexpr uint8_t kMinSlabClass = 3;
    static constexpr uint8_t kMaxSlabClass = 20;
    static constexpr uint8_t kNumSlabClasses =
        kMaxSlabClass - kMinSlabClass + 1;
    static constexpr size_t kMaxAlignment = 1 << 8;

    // [mfs]  base 2 log is really cheap these days... can't we just use an
    //        instruction, and avoid the cache impact of this table?
    //
    // see:
    // https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
    static constexpr char kLogTable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
        -1,    0,     1,     1,     2,     2,     2,     2,
        3,     3,     3,     3,     3,     3,     3,     3,
        LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6), LT(7),
        LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7),
    };

    std::unique_ptr<Segment> rdma_memory_;
    std::atomic<uint8_t *> head_;

    // Stores addresses of freed memory for a given slab class.
    inline static thread_local std::array<uint8_t, kNumSlabClasses> alignments_;
    inline static thread_local std::array<std::list<std::pair<size_t, void *>>,
                                          kNumSlabClasses>
        freelists_;

    inline unsigned int UpperLog2(size_t x) {
      size_t r;
      size_t p, q;
      if ((q = x >> 16)) {
        r = (p = q >> 8) ? 24 + kLogTable[p] : 16 + kLogTable[q];
      } else {
        r = (p = x >> 8) ? 8 + kLogTable[p] : kLogTable[x];
      }
      return ((1ul << r) < x) ? r + 1 : r;
    }

    // Returns a region of RDMA-accessible memory that satisfies the given
    // memory allocation request of `bytes` with `alignment`. First, it checks
    // whether there exists a region in `freelists_` that satisfies the request,
    // then it attempts to allocate a new region. If the request cannot be
    // satisfied, then `nullptr` is returned.
    void *do_allocate(size_t bytes, size_t alignment) override {
      if (alignment > bytes)
        bytes = alignment;
      auto slabclass = UpperLog2(bytes);
      slabclass = std::max(kMinSlabClass, static_cast<uint8_t>(slabclass));
      auto slabclass_idx = slabclass - kMinSlabClass;
      ROME_ASSERT(slabclass_idx >= 0 && slabclass_idx < kNumSlabClasses,
                  "Invalid allocation requested: {} bytes", bytes);
      ROME_ASSERT(alignment <= kMaxAlignment, "Invalid alignment: {} bytes",
                  alignment);

      if (alignments_[slabclass_idx] & alignment) {
        auto *freelist = &freelists_[slabclass_idx];
        ROME_ASSERT_DEBUG(!freelist->empty(), "Freelist should not be empty");
        for (auto f = freelist->begin(); f != freelist->end(); ++f) {
          if (f->first >= alignment) {
            auto *ptr = f->second;
            f = freelist->erase(f);
            if (f == freelist->end())
              alignments_[slabclass_idx] &= ~alignment;
            std::memset(ptr, 0, 1 << slabclass);
            ROME_TRACE("(Re)allocated {} bytes @ {}", bytes, fmt::ptr(ptr));
            return ptr;
          }
        }
      }

      uint8_t *__e = head_, *__d;
      do {
        __d = (uint8_t *)((uint64_t)__e & ~(alignment - 1)) - bytes;
        if ((void *)(__d) < rdma_memory_->raw()) {
          ROME_CRITICAL("OOM!");
          return nullptr;
        }
      } while (!head_.compare_exchange_strong(__e, __d));

      ROME_TRACE("Allocated {} bytes @ {}", bytes, fmt::ptr(__d));
      return reinterpret_cast<void *>(__d);
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
      ROME_TRACE("Deallocating {} bytes @ {}", bytes, fmt::ptr(p));
      auto slabclass = UpperLog2(bytes);
      slabclass = std::max(kMinSlabClass, static_cast<uint8_t>(slabclass));
      auto slabclass_idx = slabclass - kMinSlabClass;

      alignments_[slabclass_idx] |= alignment;
      freelists_[slabclass_idx].emplace_back(alignment, p);
    }

    // Only equal if they are the same object.
    bool do_is_equal(const std::experimental::pmr::memory_resource &other)
        const noexcept override {
      return this == &other;
    }

  public:
    virtual ~rdma_memory_resource() {}

    explicit rdma_memory_resource(size_t bytes, ibv_pd *pd)
        : rdma_memory_(std::make_unique<Segment>(bytes, pd)),
          head_(rdma_memory_->raw() + bytes) {
      std::memset(alignments_.data(), 0, sizeof(alignments_));
      ROME_TRACE("rdma_memory_resource: {} to {} (length={})",
                 fmt::ptr(rdma_memory_->raw()), fmt::ptr(head_.load()), bytes);
    }

    /// [mfs] If we cached this once, we wouldn't need the call?
    ibv_mr *mr() const { return rdma_memory_->mr(); }

    template <typename T>
    [[nodiscard]] constexpr T *allocateT(std::size_t n = 1) {
      return reinterpret_cast<T *>(allocate(sizeof(T) * n, 64));
    }

    template <typename T> constexpr void deallocateT(T *p, std::size_t n = 1) {
      deallocate(reinterpret_cast<T *>(p), sizeof(T) * n, 64);
    }
  };

  // TODO: Document this
  //
  // TODO: Now that we can have multiple connections, this needs a redesign
  struct conn_info_t {
    Connection *conn;
    uint32_t rkey;
    uint32_t lkey;
  };

  /// Used to protect the id generator and the thread_ids
  std::mutex control_lock_;
  /// A generator for thread ids
  uint64_t id_gen = 0;
  /// A mapping of thread id to an index into the reordering_semaphores array.
  /// Passed as the wr_id in work requests.
  std::unordered_map<std::thread::id, uint64_t> thread_ids;

  /// Used to protect the rdma_per_read_ summary statistics thing
  std::mutex rdma_per_read_lock_;
  rome::metrics::Summary<size_t> rdma_per_read_;

  /// a vector of semaphores, one for each thread that can send an operation.
  /// Threads will use this to recover from polling another thread's wr_id
  ///
  /// TODO:  the size should be a run-time value
  ///
  /// TODO:  threads should have descriptors and not use this, as it's going to
  ///        cause cache thrashing.
  std::array<std::atomic<int>, 20> reordering_counters;

  // TODO: The rest of this file needs a lot more documentation

  Peer self_;
  std::unique_ptr<rdma_memory_resource> rdma_memory_;

  std::unordered_map<uint16_t, conn_info_t> conn_info_;

  static inline void cpu_relax() { asm volatile("pause\n" ::: "memory"); }

public:
  MemoryPool(const Peer &self)
      : self_(self), rdma_per_read_("rdma_per_read", "ops", 10000) {}

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;

  rdma_memory_resource *init_memory(uint32_t capacity, ibv_pd *pd) {
    // Create a memory region (mr) in the current protection domain (pd)
    rdma_memory_ =
        std::make_unique<rdma_memory_resource>(capacity + sizeof(uint64_t), pd);
    return rdma_memory_.get();
  }

  void receive_conn(uint16_t id, Connection *conn, uint32_t rkey,
                    uint32_t lkey) {
    conn_info_.emplace(id, conn_info_t{conn, rkey, lkey});
  }

  rome::metrics::MetricProto rdma_per_read_proto() {
    return rdma_per_read_.ToProto();
  }
  conn_info_t conn_info(uint16_t id) const { return conn_info_.at(id); }

  // This seems to be giving each thread a unique Id, solely so that it can have
  // a slot in the reordering_counters array.  Per-thread descriptors would be a
  // better approach.
  void RegisterThread() {
    control_lock_.lock();
    std::thread::id mid = std::this_thread::get_id();
    if (this->thread_ids.find(mid) != this->thread_ids.end()) {
      ROME_FATAL("Cannot register the same thread twice");
    }
    if (this->id_gen >= THREAD_MAX) {
      ROME_FATAL("Hit upper limit on THREAD_MAX. todo: fix this condition");
    }
    this->thread_ids.insert(std::make_pair(mid, this->id_gen));
    this->reordering_counters[this->id_gen] = 0;
    this->id_gen++;
    control_lock_.unlock();
  }

  /// Allocate some memory from the local RDMA heap
  template <typename T> remote_ptr<T> Allocate(size_t size = 1) {
    auto ret =
        remote_ptr<T>(self_.id, rdma_memory_->template allocateT<T>(size));
    return ret;
  }

  /// Deallocate some memory to the local RDMA heap (must be from this node)
  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1) {
    ROME_ASSERT(p.id() == self_.id,
                "Alloc/dealloc on remote node not implemented...");
    rdma_memory_->template deallocateT<T>(std::to_address(p), size);
  }

  /// Read from RDMA, store the result in prealloc (may allocate)
  template <typename T>
  remote_ptr<T> Read(remote_ptr<T> ptr,
                     remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>();
    ReadInternal(ptr, 0, sizeof(T), sizeof(T), prealloc);
    return prealloc;
  }

  /// Read from RDMA, store the result in prealloc (may allocate)
  ///
  /// This version takes a `size` argument, for variable-length objects
  template <typename T>
  remote_ptr<T> ExtendedRead(remote_ptr<T> ptr, int size,
                             remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>(size);
    // TODO: What happens if I decrease chunk size (sizeT * size --> sizeT)
    ReadInternal(ptr, 0, sizeof(T) * size, sizeof(T) * size, prealloc);
    return prealloc;
  }

  /// Read from RDMA, store the result in prealloc (may allocate)
  ///
  /// This version does not read the entire object
  ///
  /// TODO: Does this require bytes < sizeof(T)?
  template <typename T>
  remote_ptr<T> PartialRead(remote_ptr<T> ptr, size_t offset, size_t bytes,
                            remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>();
    ReadInternal(ptr, offset, bytes, sizeof(T), prealloc);
    return prealloc;
  }

  /// Write to RDMA
  template <typename T>
  void Write(remote_ptr<T> ptr, const T &val,
             remote_ptr<T> prealloc = remote_nullptr) {
    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    // [mfs] I have a few concerns about this code:
    // -  Why do we need to allocate?  Why can't we just use `val`?  Is it
    //    because `val` might be shared?  If so, the read of `val` is racy.
    // -  If we do need to allocate, why can't we allocate on the stack?
    // -  Why are we memsetting `local` if we're using the copy constructor to
    //    set `local` to val?
    // -  Does the use of the copy constructor imply that we are assuming T is
    //    trivially copyable?
    T *local;
    if (prealloc == remote_nullptr) {
      auto alloc = rdma_memory_.get();
      local = alloc->template allocateT<T>();
      ROME_TRACE("Allocated memory for Write: {} bytes @ 0x{:x}", sizeof(T),
                 (uint64_t)local);
    } else {
      local = std::to_address(prealloc);
      ROME_TRACE("Preallocated memory for Write: {} bytes @ 0x{:x}", sizeof(T),
                 (uint64_t)local);
    }

    ROME_ASSERT((uint64_t)local != ptr.address(), "WTF");
    std::memset(local, 0, sizeof(T));
    *local = val;
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local);
    sge.length = sizeof(T);
    sge.lkey = rdma_memory_->mr()->lkey;

    ibv_send_wr send_wr_{};
    send_wr_.wr_id = index_as_id;
    send_wr_.num_sge = 1;
    send_wr_.sg_list = &sge;
    send_wr_.opcode = IBV_WR_RDMA_WRITE;
    send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    send_wr_.wr.rdma.remote_addr = ptr.address();
    send_wr_.wr.rdma.rkey = info.rkey;

    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = 1;
    // Send the request
    info.conn->send_onesided(&send_wr_);
    // TODO: [esl] poll for more than 1
    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = info.conn->poll_cq(1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})",
                  (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                  (std::stringstream() << ptr).str());
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    // [mfs]  It is odd that we have metrics for read (albeit with a bottleneck)
    //        but we don't have metrics for write/swap/cas?

    if (prealloc == remote_nullptr) {
      auto alloc = rdma_memory_.get();
      alloc->template deallocateT<T>(local);
    }
  }

  /// Do a 64-bit swap over RDMA
  ///
  /// TODO: This is really just a CAS with a loop if it fails, because RDMA
  ///       doesn't support a true "swap" operation.  It's still good to have,
  ///       because of the overhead of trying from scratch, but we can certainly
  ///       optimize it a bit.
  template <typename T>
  T AtomicSwap(remote_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    auto alloc = rdma_memory_.get();
    // [esl]  There is probably a better way to avoid allocating every time we
    //        do this call (maybe be preallocating the space thread_local)
    volatile uint64_t *prev_ = alloc->template allocateT<uint64_t>();

    ibv_sge sge{.addr = reinterpret_cast<uint64_t>(prev_),
                .length = sizeof(uint64_t),
                .lkey = rdma_memory_->mr()->lkey};

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
      reordering_counters[index_as_id] = 1;
      info.conn->send_onesided(&send_wr_);

      // Poll until we match on the condition
      ibv_wc wc;
      while (reordering_counters[index_as_id] != 0) {
        int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
        if (poll == 0 || (poll < 0 && errno == EAGAIN))
          continue;
        // Assert a good result
        ROME_ASSERT(
            poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
            (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
        int old = reordering_counters[wc.wr_id].fetch_sub(1);
        ROME_ASSERT(old >= 1, "Broken synchronization");
      }

      if (*prev_ == send_wr_.wr.atomic.compare_add)
        break;
      send_wr_.wr.atomic.compare_add = *prev_;
    };
    T ret = T(*prev_);
    alloc->deallocateT((uint64_t *)prev_, 8);
    return ret;
  }

  /// Do a 64-bit CAS over RDMA
  ///
  /// [mfs] If the swap value is always a uint64_t, why is this templated on T?
  ///       Or perhaps the question is "how does this work if the field to CAS
  ///       isn't the first field of the T?"
  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    auto alloc = rdma_memory_.get();
    // [esl] There is probably a better way to avoid allocating every time we do
    // this call (maybe be preallocating the space thread_local)
    volatile uint64_t *prev_ = alloc->template allocateT<uint64_t>();

    // TODO: would the code be clearer if all of the ibv_* initialization
    // throughout this file used the new syntax?
    // [esl] I agree, i think its much cleaner
    ibv_sge sge{.addr = reinterpret_cast<uint64_t>(prev_),
                .length = sizeof(uint64_t),
                .lkey = rdma_memory_->mr()->lkey};

    ibv_send_wr send_wr_{};
    send_wr_.wr_id = index_as_id;
    send_wr_.num_sge = 1;
    send_wr_.sg_list = &sge;
    send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    send_wr_.wr.atomic.remote_addr = ptr.address();
    send_wr_.wr.atomic.rkey = info.rkey;
    send_wr_.wr.atomic.compare_add = expected;
    send_wr_.wr.atomic.swap = swap;

    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = 1;
    info.conn->send_onesided(&send_wr_);

    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = info.conn->poll_cq(1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
                  (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    T ret = T(*prev_);
    alloc->template deallocateT<uint64_t>((uint64_t *)prev_, 8);
    return ret;
  }

  template <typename T> inline remote_ptr<T> GetRemotePtr(const T *ptr) const {
    return remote_ptr<T>(self_.id, reinterpret_cast<uint64_t>(ptr));
  }

  template <typename T> inline remote_ptr<T> GetBaseAddress() const {
    return GetRemotePtr<T>(
        reinterpret_cast<const T *>(rdma_memory_->mr()->addr));
  }

private:
  /// Internal method implementing common code for RDMA read
  ///
  /// TODO: It appears that we *always* call this with bytes <= chunk_size.
  ///       Could we get rid of some of the complexity?
  template <typename T>
  void ReadInternal(remote_ptr<T> ptr, size_t offset, size_t bytes,
                    size_t chunk_size, remote_ptr<T> prealloc) {
    const int num_chunks =
        bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
    const size_t remainder = bytes % chunk_size;
    const bool is_multiple = remainder == 0;

    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    T *local = std::to_address(prealloc);
    ibv_sge sges[num_chunks];
    ibv_send_wr wrs[num_chunks];

    for (int i = 0; i < num_chunks; ++i) {
      auto chunk_offset = offset + i * chunk_size;
      sges[i].addr = reinterpret_cast<uint64_t>(local) + chunk_offset;
      if (is_multiple) {
        sges[i].length = chunk_size;
      } else {
        sges[i].length = (i == num_chunks - 1 ? remainder : chunk_size);
      }
      sges[i].lkey = rdma_memory_->mr()->lkey;

      wrs[i].wr_id = index_as_id;
      wrs[i].num_sge = 1;
      wrs[i].sg_list = &sges[i];
      wrs[i].opcode = IBV_WR_RDMA_READ;
      wrs[i].send_flags = IBV_SEND_FENCE;
      if (i == num_chunks - 1)
        wrs[i].send_flags |= IBV_SEND_SIGNALED;
      wrs[i].wr.rdma.remote_addr = ptr.address() + chunk_offset;
      wrs[i].wr.rdma.rkey = info.rkey;
      wrs[i].next = (i != num_chunks - 1 ? &wrs[i + 1] : nullptr);
    }

    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = num_chunks;
    info.conn->send_onesided(wrs);

    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = info.conn->poll_cq(1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(
          poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
          (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)), ptr);
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    // Update rdma per read
    //
    // TODO: If we're serious about being multithreaded, rdma_per_read needs to
    // be thread-local, or else all threads will contend on this.
    rdma_per_read_lock_.lock();
    rdma_per_read_ << num_chunks;
    rdma_per_read_lock_.unlock();
  }

  // [mfs]  According to [el], it is possible to post multiple requests on the
  //        same qp, and they'll finish in order, so we definitely will want a
  //        way to let that happen.
  // [esl] Just to cite my sources:
  // https://www.rdmamojo.com/2013/07/26/libibverbs-thread-safe-level/ (Thread
  // safe) https://www.rdmamojo.com/2013/01/26/ibv_post_send/ (Ordering
  // guarantee, for RC only)
  //    "In RC QP, there is a PSN (Packet Serial Number) that guarantees the
  //    order of the messages"
  // https://www.rdmamojo.com/2013/06/01/which-queue-pair-type-to-use/ (Ordering
  // guarantee also mentioned her)
};

} // namespace rome::rdma::internal
