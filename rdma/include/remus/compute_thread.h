#pragma once

#include <immintrin.h>

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
#include "ring.h"
#include "util.h"

namespace remus::internal {

/// @brief  A simple distributed bump allocator
/// @details
/// This is a size-preserving allocator,
/// because it does not coalesce adjacent free blocks into larger free
/// blocks. It is somewhat amenable to type-preservation, because there is
/// some room in object headers that a synchronization mechanism can use for
/// metadata.
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

  /// The header for allocated blocks of memory
  struct header_t {
    std::atomic<uint64_t> size_;    // The size of the block: write-once
    std::atomic<uint64_t> padding_; // Padding to 16B; can be used as a lock
  };

  /// Compute the slabclass for a given size.  Assumes size includes
  /// HEADER_SIZE.
  ///
  /// @param size The size of the allocation in bytes, including the header
  inline uint64_t calculate_slabclass(uint64_t size) {
    // TODO:  Check this math... do we need more parenthesis to get it right?
    //        Are we sure it's doing what we want?  Why not use roundup()
    //        instead?
    //
    // TODO: The magic numbers in this code should be constants
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
  /// TODO: The following warning is probably a MAJOR BUG
  ///
  /// @warning  Sizeof(T) is insufficient for variable-sized objects, such as
  ///           skip list nodes!
  ///
  /// @tparam T The type of the item to allocate, used for getting its size
  ///
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
  ///           available memory, if the available memory is in Segments that
  ///           the policy won't access.
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
      auto offset = faa(bump_counter, size);
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
      writer(rdma_ptr<uint64_t>(ptr + offsetof(header_t, size_)), size);
      writer(rdma_ptr<uint64_t>(ptr + offsetof(header_t, padding_)), 0);
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
    /// NB: The common case is not "big allocations", hence the branch hint
    if (size > ALLOC_MED_THRESH) [[unlikely]] {
      // For large allocations (>8192), use best-fit allocation from the
      // best_fit_freelist Find the first chunk in the freelist that can fit
      // this allocation
      //
      // TODO:  This is first-fit, not best-fit.  That's probably unwise?
      //        Also, reverse iteration is probably smarter.
      auto it = std::find_if(
          free_blocks_.begin(), free_blocks_.end(),
          [size](const auto &chunk) { return chunk.first >= size; });
      if (it != free_blocks_.end()) {
        uint64_t ptr = it->second;
        // TODO:  Erasing from a vector is costly.  Consider using a deque? Or
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
  /// TODO: I dislike the above warning.  If that's the case, why not have this
  ///       code read the size out of the header?
  ///
  /// @tparam T The type of object referenced.  Immaterial to this function
  ///
  /// @param ptr    An rdma_ptr
  /// @param size   The size that was read from ptr / the allocation size (not
  ///               sizeof(T))
  template <typename T> void reclaim(rdma_ptr<T> ptr, uint64_t size) {
    uint64_t slabclass = calculate_slabclass(size);
    // TODO: Again, small blocks are the common case, hence the branch hint
    if (slabclass > ALLOC_MED_THRESH) [[unlikely]] {
      free_blocks_.push_back({slabclass, ptr.raw() - HEADER_SIZE});
    } else {
      // TODO:  See constructor.  I think we pre-insert a vector, so .end() is
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
      : seg_size_(1ULL << args->uget(SEG_SIZE)), mn_alloc_pol_(args) {
    // initialize freelists
    //
    // TODO:  Reclaim() assumes they're not initialized, and initializes them
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
///
/// TODO: Need to implement the metrics
///
/// TODO: Check destructors and documentation
///
/// TODO: [ys] all rdma ops can be called with any possible interleaving,
///       But more experiemts need to be done to confirm this claim.
///
/// TODO: ComputeThread has a ton of sub-classes in it.  Why?  What purpose do
///       they serve?

/// @brief A ComputeThread is a thread that can perform RDMA operations
/// @details
/// ComputeThread is a per-node object that, once configured, provides all of
/// the underlying features needed by threads wishing to interact with RDMA
/// memory.
class ComputeThread {
protected:
  /// @brief An object to reserve and release slots in a staging buffer
  ///        for RDMA communication.
  /// @details
  /// op_counter_t provides the next available slot in the staging buffer based
  /// on the ComputeThread's op_counter_end and op_counter_assignments. The
  /// number of slots is determined by the CN_OPS_PER_THREAD argument.
  /// TODO: All field names should end with an underscore
  class op_counter_t {
    ComputeThread
        *const ct_; // The ComputeThread this struct is associated with
    size_t idx_;    // The index of this op_counter in the ring
    const size_t
        op_counter_num_; // The max number of concurrent operations per thread

  public:
    /// @brief Constructs a op_counter_t object
    /// @param ct The ComputeThread this op_counter_t is associated with
    op_counter_t(ComputeThread *ct)
        : ct_(ct), idx_(0),
          op_counter_num_(ct_->args_->uget(remus::CN_OPS_PER_THREAD)) {
      auto result = ring_counter_t::acquire(
          ct_->op_counter_end, ct_->op_counter_assignments, op_counter_num_);
      REMUS_ASSERT(result.has_value(), "op_counter is not available");
      idx_ = result.value();
      ///       REMUS_DEBUG("Debug: op_counter_t idx = {}", idx_);
    }

    /// @brief Returns a pointer to the operation counter associated with this
    /// slot.
    /// @return A pointer to the std::atomic<int> counter at index idx_
    std::atomic<int> *val() { return &ct_->op_counters_[idx_]; }

    /// @brief Destructs the op_counter_t object
    ~op_counter_t() {
      ///       REMUS_DEBUG("Debug: ~op_counter_t idx = {}", idx_);
      ring_counter_t::release(idx_, ct_->op_counter_start,
                              ct_->op_counter_assignments, op_counter_num_);
    }
  };

  /// @brief An object to reserve and release coroutine-local slots
  /// @details
  /// seq_idx_t provides the next available slot in the ring for a
  /// coroutine-local operation. The number of slots is determined by the
  /// CN_OPS_PER_THREAD argument. Each ComputeThread has a separate seq_idx_t
  /// for each coroutine index (coro_idx).
  struct seq_idx_t {
    ComputeThread
        *const ct_; // The ComputeThread this seq_idx_t is associated with
    size_t idx_;    // The index of this seq_idx in the ring
    const uint32_t
        coro_idx_; // The coroutine index this seq_idx_t is associated with
    const size_t seq_op_counter_num_; // The max number of concurrent operations

    /// @brief Constructs a seq_idx_t object
    /// @param ct The ComputeThread this seq_idx_t is associated with
    /// @param coro_idx The coroutine index this seq_idx_t is associated with
    seq_idx_t(ComputeThread *ct, uint32_t coro_idx)
        : ct_(ct), idx_(0), coro_idx_(coro_idx),
          seq_op_counter_num_(ct_->args_->uget(remus::CN_OPS_PER_THREAD)) {
      auto result = ring_counter_t::acquire(
          ct_->seq_op_counter_end[coro_idx_],
          ct_->seq_op_counter_assignments[coro_idx_], seq_op_counter_num_);
      REMUS_ASSERT(result.has_value(),
                   "seq_idx for coro_idx = {} is not available", coro_idx_);
      idx_ = result.value();
      REMUS_DEBUG("Debug: seq_idx_t idx = {}", idx_);
    }

    /// @brief Returns the index of this seq_idx_t in the ring
    /// @return The index of this seq_idx_t in the ring
    size_t val() { return idx_; }

    /// @brief Destructs the seq_idx_t object
    ~seq_idx_t() {
      // REMUS_DEBUG("Debug: seq_idx_t ~seq_idx_t idx = {}", idx_);
      ring_counter_t::release(idx_, ct_->seq_op_counter_start[coro_idx_],
                              ct_->seq_op_counter_assignments[coro_idx_],
                              ct_->args_->uget(remus::CN_OPS_PER_THREAD));
    }
  };

  /// @brief A staging_buf_t is a buffer that can be used for staging RDMA
  /// operations
  struct staging_buf_t {
    ComputeThread *const ct_; // Pointer to the parent ComputeThread
    const size_t size_;       // Size of the allocated buffer in bytes
    const size_t align_;      // Alignment of the buffer in bytes
    uint8_t *buf_;            // Pointer to the acquired buffer from ring_buf_t

    /// @brief Constructs a staging_buf_t object
    /// @param ct The ComputeThread this staging_buf_t is associated with
    /// @param size The size of the buffer to acquire in bytes
    /// @param align The alignment of the buffer in bytes
    staging_buf_t(ComputeThread *const ct, const size_t size,
                  const size_t align)
        : ct_(ct), size_(size), align_(align),
          buf_(ring_buf_t::acquire(
              ct_->staging_buf_, ct_->staging_buf_end, ct_->staging_buf_start,
              ct_->staging_buf_size_, ct_->staging_buf_allocations_, size_,
              align_)) {}

    /// @brief Returns a pointer to the staging buffer
    /// @return A pointer to the staging buffer
    uint8_t *val() {
      REMUS_ASSERT(buf_, "staging buf is not enough");
      REMUS_ASSERT(buf_ >= ct_->staging_buf_ &&
                       buf_ + size_ <=
                           ct_->staging_buf_ + ct_->staging_buf_size_,
                   "Staging buf out of range");
      return buf_;
    }

    /// @brief Destructs the staging_buf_t object
    ~staging_buf_t() {
      ring_buf_t::release(buf_, ct_->staging_buf_allocations_,
                          ct_->staging_buf_start, ct_->staging_buf_,
                          ct_->staging_buf_size_);
    }
  };

  /// @brief A seq_staging_buf_t is a buffer that can be used for staging RDMA
  /// operations in a coroutine-local context
  struct seq_staging_buf_t {
    ComputeThread *const ct_; // Pointer to the parent ComputeThread
    const size_t size_;       // Size of the allocated buffer in bytes
    const size_t align_;      // Alignment of the buffer in bytes
    uint8_t *buf_;            // Pointer to the acquired buffer from ring_buf_t

    /// @brief Constructs a seq_staging_buf_t object
    /// @param ct The Co mputeThread this staging_buf_t is associated with
    /// @param size The size of the buffer to acquire in bytes
    /// @param align The alignment of the buffer in bytes
    seq_staging_buf_t(ComputeThread *ct, const size_t size, const size_t align)
        : ct_(ct), size_(size), align_(align),
          buf_(ring_buf_t::acquire(
              ct_->staging_buf_, ct_->staging_buf_end, ct_->staging_buf_start,
              ct_->staging_buf_size_, ct_->staging_buf_allocations_, size_,
              align_)) {}

    /// @brief Returns a pointer to the staging buffer
    /// @return A pointer to the staging buffer
    uint8_t *val() {
      REMUS_ASSERT(buf_, "seq staging buf is not enough");
      REMUS_ASSERT(buf_ >= ct_->staging_buf_ &&
                       buf_ + size_ <=
                           ct_->staging_buf_ + ct_->staging_buf_size_,
                   "Staging buf out of range");
      return buf_;
    }

    /// @brief Destructs the staging_buf_t object
    ~seq_staging_buf_t() {
      ring_buf_t::release(buf_, ct_->staging_buf_allocations_,
                          ct_->staging_buf_start, ct_->staging_buf_,
                          ct_->staging_buf_size_);
    }
  };

  /// @brief A cached_buf_t is a buffer that can be reused across operations
  struct cached_buf_t {
    ComputeThread *const ct_; // Pointer to the parent ComputeThread
    const size_t size_;       // Size of the allocated buffer
    const size_t align_;      // Alignment of the buffer
    uint8_t *buf_;            // Pointer to the acquired buffer from ring_buf_t

    /// @brief Constructs a cached_buf_t object to acquire a buffer from the
    /// right buffer
    /// @param parent_ct The parent ComputeThread this cached_buf_t is
    /// associated with
    /// @param sz The size of the buffer to acquire in bytes
    /// @param al The alignment of the buffer in bytes
    cached_buf_t(ComputeThread *const parent_ct, const size_t sz,
                 const size_t al)
        : ct_(parent_ct), size_(sz), align_(al),
          buf_(ring_buf_t::acquire(
              parent_ct->cached_buf_, parent_ct->cached_buf_end,
              parent_ct->cached_buf_start, parent_ct->cached_buf_size_,
              parent_ct->cached_buf_allocations_, sz, al)) {}

    /// @brief Constructs a cached_buf_t object by moving from another
    /// cached_buf_t
    /// @param other The other cached_buf_t to move from
    cached_buf_t(cached_buf_t &&other) noexcept
        : ct_(other.ct_), size_(other.size_), align_(other.align_),
          buf_(other.buf_) {
      other.buf_ =
          nullptr; // Critical: Moved-from object no longer owns the buffer
    }

    /// @brief Destructs the cached_buf_t object
    ~cached_buf_t() {
      if (buf_) {
        ring_buf_t::release(buf_, ct_->cached_buf_allocations_,
                            ct_->cached_buf_start, ct_->cached_buf_,
                            ct_->cached_buf_size_);
      }
    }

    /// @brief Returns a pointer to the cached buffer
    /// @return The pointer to the cached buffer
    uint8_t *val() const {
      REMUS_ASSERT(buf_, "cached_buf_t::val() called on a null buffer "
                         "(moved-from or failed acquire?)");
      if (buf_) { // Only assert range if buf is not null
        REMUS_ASSERT(buf_ >= ct_->cached_buf_ &&
                         buf_ + size_ <=
                             ct_->cached_buf_ + ct_->cached_buf_size_,
                     "Cached buf out of range");
      }
      return buf_;
    }
    /// Default copy constructor and assignment operator are deleted
    cached_buf_t(const cached_buf_t &) = delete;
    cached_buf_t &operator=(const cached_buf_t &) = delete;

    /// @brief Move assignment operator for cached_buf_t
    /// @param other The cached_buf_t to move from
    /// @return A reference to this cached_buf_t after the move
    cached_buf_t &operator=(cached_buf_t &&other) noexcept {
      if (this != &other) {
        // Release current resource, if any and owned
        if (buf_) {
          ring_buf_t::release(buf_, ct_->cached_buf_allocations_,
                              ct_->cached_buf_start, ct_->cached_buf_,
                              ct_->cached_buf_size_);
        }
        buf_ = other.buf_;
        other.buf_ = nullptr;
      }
      return *this;
    }
  };

  /// @brief A ComputeThead's staging buffer allocations map
  std::unordered_map<uint8_t *, ring_buf_t::buf_allocation_t>
      staging_buf_allocations_;
  /// @brief A ComputeThread's cached buffer allocations map
  std::unordered_map<uint8_t *, ring_buf_t::buf_allocation_t>
      cached_buf_allocations_;
  /// @brief A ComputeThread's staging buffer manager
  std::unordered_map<uint8_t *, cached_buf_t> cached_buf_manager_;

  /// @brief  rdma_ptrs that are scheduled for reclamation
  ///
  /// NB: The actual type of these pointers does not matter, since the size
  ///     information is in the common object header.
  std::vector<uintptr_t> scheduled_reclamations;

  /// @brief Lane object to manage operations per rdma-channel or "Lane"
  ///
  /// TODO: Add underscore to end of field names
  struct Lane {
    const uint32_t lane_idx;                            // TODO
    std::vector<std::atomic<size_t>> &lane_op_counters; // TODO

    /// @brief Constructs a Lane object, which represents an single RDMA channel
    /// @param lane_idx The index of the lane in the vector of lanes
    /// @param lane_op_counters_ A reference to the vector of operation counters
    Lane(const uint32_t lane_idx,
         std::vector<std::atomic<size_t>> &lane_op_counters_)
        : lane_idx(lane_idx), lane_op_counters(lane_op_counters_) {
      if (lane_op_counters[lane_idx].fetch_add((uint64_t)1) + 1 >=
          remus::internal::kMaxWr) {
        REMUS_FATAL("lane_op_counters[{}] is greater than kMaxWr = {}, please "
                    "increase kMaxWr",
                    lane_idx, remus::internal::kMaxWr);
      }
    }

    /// @brief Returns the operation counter for this lane
    ~Lane() { lane_op_counters[lane_idx].fetch_sub((uint64_t)1); }
  };

  /// @brief A seq_send_wrs_t is a collection of send work requests for a
  /// specific sequence of operations
  struct seq_send_wrs_t {
    /// @brief A send work request for a specific operation,
    /// containing the work request and scatter-gather entry
    struct send_wr_t {
      std::shared_ptr<ibv_send_wr> wr;
      std::shared_ptr<ibv_sge> sge;
    };
    bool posted = false;
    std::unique_ptr<seq_idx_t> seq_idx;
    std::unique_ptr<Lane> lane;
    std::vector<std::unique_ptr<op_counter_t>> op_counters;
    std::vector<std::unique_ptr<seq_staging_buf_t>> staging_bufs;
    std::vector<send_wr_t> send_wrs;
  };

  uint64_t node_id;                           // The ComputeNode id
  uint64_t id_;                               // This thread's Id
  std::shared_ptr<ComputeNode> compute_node_; // The ComputeNode
  std::shared_ptr<ArgMap>
      args_; // The command-line args to the program
             /// op_counters for receiving ibv completion events
  ///
  /// TODO: A vector is overkill until we support async one-sided ops
  std::vector<std::atomic<int>> op_counters_;
  std::vector<ring_counter_t::State> op_counter_assignments;
  uint64_t op_counter_start = 0;
  uint64_t op_counter_end = 0;
  std::vector<std::vector<ring_counter_t::State>> seq_op_counter_assignments;
  std::vector<uint64_t> seq_op_counter_start;
  std::vector<uint64_t> seq_op_counter_end;
  std::vector<std::unordered_map<uint32_t, seq_send_wrs_t>> seq_send_wrs;

  /// The policy for deciding which QP to use when connecting to a MemoryNode
  internal::QpSchedPolicy qp_sched_pol_;
  internal::BumpAllocator allocator; // The allocator

  uint64_t staging_buf_size_;
  uint64_t cached_buf_size_;
  uint8_t *staging_buf_start;
  uint8_t *staging_buf_end;
  uint8_t *cached_buf_start;
  uint8_t *cached_buf_end;
  uint8_t *staging_buf_;
  uint8_t *cached_buf_;

public:
  /// Construct a ComputeThread
  ///
  /// @param id   The thread's zero-based numerical Id
  /// @param cn   The ComputeNode context for this machine
  /// @param args The command-line arguments to the program
  explicit ComputeThread(uint64_t id, std::shared_ptr<ComputeNode> cn,
                         std::shared_ptr<ArgMap> args)
      : node_id(id), compute_node_(cn), args_(args),
        op_counters_(args->uget(CN_OPS_PER_THREAD)),
        op_counter_assignments(args->uget(CN_OPS_PER_THREAD),
                               ring_counter_t::State::AVAILABLE),
        seq_op_counter_assignments(args->uget(CN_OPS_PER_THREAD),
                                   std::vector<ring_counter_t::State>(
                                       args->uget(CN_OPS_PER_THREAD),
                                       ring_counter_t::State::AVAILABLE)),
        seq_op_counter_start(args->uget(CN_OPS_PER_THREAD), 0),
        seq_op_counter_end(args->uget(CN_OPS_PER_THREAD), 0),
        seq_send_wrs(args->uget(CN_OPS_PER_THREAD)), qp_sched_pol_(args),
        allocator(args) {
    // TODO:  This would be much simpler if we could extract id_ from an
    //        initializer.  Consider switching to a factory?
    auto registration = compute_node_->register_thread();
    id_ = registration.first;
    auto seg_size_ = 1ULL << (args->uget(CN_THREAD_BUFSZ));
    auto seg_slice_ = registration.second;
    staging_buf_size_ = seg_size_ >> 1;
    staging_buf_ = seg_slice_;
    staging_buf_start = staging_buf_;
    staging_buf_end = staging_buf_start;
    cached_buf_size_ = seg_size_ >> 1;
    cached_buf_ = seg_slice_ + staging_buf_size_;
    cached_buf_start = cached_buf_;
    cached_buf_end = cached_buf_start;
    REMUS_INFO("Created thread #{}", id_);

    // Select the scheduling policies to use
    qp_sched_pol_.set_policy(
        internal::QpSchedPolicy::to_policy(args_->sget(QP_SCHED_POL)), id_);
    allocator.mn_alloc_pol_.set_policy(
        internal::MnAllocPolicy::to_policy(args->sget(ALLOC_POL)), args_, id_);
  }

  /// @brief Destructor for ComputeThread
  ~ComputeThread() {
    // send shutdown to all memory nodes's first segment's control block's
    // control_flag_
    for (uint64_t i = 0;
         i < args_->uget(LAST_MN_ID) - args_->uget(FIRST_MN_ID) + 1; i++) {
      auto control_flag =
          rdma_ptr<uint64_t>(compute_node_->get_seg_start(i, 0) +
                             offsetof(internal::ControlBlock, control_flag_));
      FetchAndAdd(control_flag, 1);
    }
    REMUS_ASSERT(no_leak_detected(), "Leak detected");
    REMUS_INFO("ComputeThread {} shutdown", id_);
  }

  /// @brief Returns the thread's unique identifier
  /// @return The thread id as uint64_t
  uint64_t get_tid() { return id_; }

  /// @brief Read a fixed-sized object from the RDMA heap
  /// @tparam T The type of the object to read
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param fence If true, a fence is issued after the read operation
  /// @return The object read from the RDMA heap
  template <typename T> T Read(rdma_ptr<T> ptr, bool fence = true) {
    /// Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto staging_buf = staging_buf_t(this, sizeof(T), alignof(T)).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                         op_counter, sizeof(T), true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    return *(T *)staging_buf;
  }

  /// @brief Alternative version of Read that allows reading directly into a
  /// segment
  /// @details This version of Read is the zero-copy version that reads
  /// directly into a segment, avoiding the need for an intermediate staging
  /// buffer.
  /// @tparam T The type of the object to read
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param seg A pointer to the segment where the data will be read into
  /// @param fence If true, a fence is issued after the read operation
  /// @param size The size of the object to read, defaults to sizeof(T)
  template <typename T>
  void Read(rdma_ptr<T> ptr, T *seg, bool fence = true,
            size_t size = sizeof(T)) {
    /// Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::ReadConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                         op_counter, size, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
  }

  /// @brief Write a fixed-sized object to the RDMA heap
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param val The value to write to the RDMA heap
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true and the address is local to the machine,
  ///                   the write is done locally without RDMA
  template <typename T>
  void Write(rdma_ptr<T> ptr, const T &val, bool fence = true,
             size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), &val, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      return;
    }
    // Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto staging_buf = staging_buf_t(this, size, alignof(T)).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                          op_counter, size, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
  }

  /// @brief An alternative version of Write that allows writing directly into a
  /// segment
  /// @details
  /// This version of Write is the zero-copy version that writes directly into a
  /// segment, avoiding the need for an intermediate staging buffer.
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param seg A pointer to the segment where the data will be written into
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true and the address is local to the machine,
  template <typename T>
  void Write(rdma_ptr<T> ptr, T *seg, bool fence = true,
             size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), seg, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      return;
    }
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                          op_counter, size, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
  }
  /// @brief Perform a CompareAndSwap on the RDMA heap
  /// @tparam T The type of the object to compare and swap
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param expected The expected value to compare against
  /// @param swap The value to swap in if the expected value matches
  /// @param fence If true, a fence is issued after the compare and swap
  /// @return The CAS result of type T
  template <typename T>
    requires(sizeof(T) <= 8)
  T CompareAndSwap(rdma_ptr<T> ptr, T expected, T swap, bool fence = true) {
    // Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto staging_buf = staging_buf_t(this, sizeof(T), alignof(T)).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::CompareAndSwapConfig(send_wr, sge, ptr, (uint64_t)expected,
                                   (uint64_t)swap, (uint64_t *)staging_buf,
                                   rkey, ci.lkey_, op_counter, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    return *(T *)staging_buf;
  }

  /// @brief Perform a FetchAndAdd operation on the RDMA heap
  /// @tparam T The type of the object to fetch and add
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param add The value to add to the object
  /// @param fence If true, a fence is issued after the fetch and add
  /// @return The value before the addition of type T
  template <typename T>
    requires(sizeof(T) <= 8)
  T FetchAndAdd(rdma_ptr<T> ptr, uint64_t add, bool fence = true) {
    // Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto staging_buf = staging_buf_t(this, sizeof(T), alignof(T)).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::FetchAndAddConfig(send_wr, sge, ptr, add, (uint64_t *)staging_buf,
                                rkey, ci.lkey_, op_counter, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    return *(T *)staging_buf;
  }

  /// NB: ensure all ptrs in seq belong to the same memory segment
  template <typename T>
  std::optional<std::vector<T>> ReadSeq(rdma_ptr<T> ptr, bool signal = false,
                                        bool fence = false) {
    auto coro_idx =
        0; // because we don't support more than one top level coroutine
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto staging_buf_ptr =
        std::make_unique<seq_staging_buf_t>(this, sizeof(T), alignof(T));
    auto staging_buf = staging_buf_ptr->val();
    REMUS_ASSERT(staging_buf != nullptr,
                 "Staging buffer is not enough, increase the staging buffers "
                 "or reduce the number of requests");
    seq_send_wrs[coro_idx][seq_idx].staging_bufs.push_back(
        std::move(staging_buf_ptr));
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                           nullptr, sizeof(T), signal, fence);
      return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                         op_counter, sizeof(T), signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    return result;
  }

  // this is a special version that allows read to seg.
  /// NB: Although this can be used interleaved with other read/write_seq
  /// operations, results of read by calling this function won't be in the
  /// result vector.
  template <typename T>
  std::optional<std::vector<T>> ReadSeq(rdma_ptr<T> ptr, T *seg,
                                        bool signal = false, bool fence = false,
                                        size_t size = sizeof(T)) {
    /// Use the scheduling policy to select the next connection
    auto coro_idx = 0;
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::ReadConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                           nullptr, size, signal, fence);
      return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::ReadConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                         op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    return result;
  }

  /// NB: ensure all ptrs in seq belong to the same memory segment
  template <typename T>
  std::optional<std::vector<T>>
  WriteSeq(rdma_ptr<T> ptr, const T &val, bool signal = false,
           bool fence = false, size_t size = sizeof(T),
           bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), &val, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      return std::nullopt;
    }
    auto coro_idx =
        0; // because we don't support more than one top level coroutine
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto staging_buf_ptr =
        std::make_unique<seq_staging_buf_t>(this, sizeof(T), alignof(T));
    auto staging_buf = staging_buf_ptr->val();
    REMUS_ASSERT(staging_buf != nullptr,
                 "Staging buffer is not enough, increase the staging buffers "
                 "or reduce the number of requests");
    seq_send_wrs[coro_idx][seq_idx].staging_bufs.push_back(
        std::move(staging_buf_ptr));
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                            nullptr, sizeof(T), signal, fence);
      return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                          op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    return result;
  }

  // this is a special version that allows write from seg
  template <typename T>
  std::optional<std::vector<T>>
  WriteSeq(rdma_ptr<T> ptr, T *seg, bool signal = false, bool fence = false,
           size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), seg, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      return std::nullopt;
    }
    auto coro_idx =
        0; // because we don't support more than one top level coroutine
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                            nullptr, size, signal, fence);
      return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                          op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    internal::Poll(ci.conn_.get(), op_counter, ptr);
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    return result;
  }

  /// Determine if a rdma_ptr is local to the machine
  template <class T> bool is_local(rdma_ptr<T> ptr) {
    return ptr.id() == node_id;
  }

  /// @brief Extract the segment id from a rdma_ptr
  template <typename T> uint64_t seg_id(rdma_ptr<T> ptr) {
    return ptr.raw() >> args_->uget(SEG_SIZE);
  }

  /// @brief  Arrive at the global barrier in Segment 0 of MemoryNode 0
  /// @param total_threads The total number of threads that will arrive at the
  /// barrier
  /// @return True if this thread was the last to arrive, false otherwise
  bool arrive_control_barrier(int total_threads) {
    auto barrier =
        rdma_ptr<uint64_t>(compute_node_->get_seg_start(0, 0) +
                           offsetof(internal::ControlBlock, barrier_));
    // Arrive via a simple increment, use low bit to get the next "sense"
    auto was = FetchAndAdd(barrier, 2);
    uint64_t new_sense = 1 - (was & 1);
    // If the preceding FAA was the last one, reset the barrier, otherwise spin
    if ((was >> 1) == (total_threads - 1)) {
      Write(barrier, new_sense);
      return true;
    }
    while ((Read(barrier) & 1) != new_sense) {
    }
    return true;
  }

  /// @brief Allocate a region of n * sizeof(T) bytes.  This will use the memory
  /// allocation policy to choose a memory node from which to allocate.
  /// @tparam T The type of the object to allocate
  /// @param n The number of elements to allocate, defaults to 1
  /// @return An rdma_ptr<T> pointing to the allocated memory, or an empty
  template <typename T> rdma_ptr<T> allocate(std::size_t n = 1) {
    auto size = allocator.compute_size<T>(n);
    auto local = allocator.try_allocate_local(size);
    if (local.has_value())
      return rdma_ptr<T>(local.value());
    // TODO:  The use of four lambdas here is really icky, this should be
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

  /// @brief Deallocate a region of memory, so it can be used again.
  /// @tparam T The type of the object to deallocate
  /// @param ptr The rdma_ptr<T> pointing to the memory to deallocate
  /// @return True if the deallocation was successful, false otherwise
  template <typename T> bool deallocate(rdma_ptr<T> ptr) {
    auto size = Read<uint64_t>(
        rdma_ptr<uint64_t>(ptr.raw() - internal::BumpAllocator::HEADER_SIZE));
    allocator.reclaim(ptr, size);
    return true;
  }

  /// @brief Only allocates memory in local memory seg_slice_, n means n bytes
  /// FOR T
  /// @tparam T The type of the object to allocate
  /// @param num_elements The number of elements to allocate, defaults to 1
  /// @return A pointer to the allocated memory, or nullptr if allocation failed
  template <typename T> T *local_allocate(std::size_t num_elements = 1) {
    size_t n_bytes = sizeof(T) * num_elements;
    auto temp_obj_owner =
        std::make_unique<cached_buf_t>(this, n_bytes, alignof(T));
    uint8_t *key_buf_ptr = temp_obj_owner->val();

    if (!key_buf_ptr) {
      return nullptr;
    }

    auto [iterator, success] = cached_buf_manager_.emplace(
        std::piecewise_construct, std::forward_as_tuple(key_buf_ptr),
        std::forward_as_tuple(std::move(*temp_obj_owner)));

    REMUS_ASSERT(success, "Failed to allocate memory");

    return reinterpret_cast<T *>(iterator->second.val());
  }

  /// @brief Deallocate local memory allocated with local_allocate
  /// @tparam T The type of the object to deallocate
  /// @param buf The pointer to the memory to deallocate
  template <typename T> void local_deallocate(T *buf) {
    cached_buf_manager_.erase((uint8_t *)buf);
  }

  /// @brief Reset the local cache slice, clearing all cached buffers
  void reset_cache_slice() { cached_buf_manager_.clear(); }

  /// @brief Set the root pointer in MemoryNode 0, Segment 0, to `root`
  /// @tparam T The type of the root pointer
  /// @param root The rdma_ptr<T> pointing to the root object in the RDMA heap
  /// TODO: Should we pass in the memory node and segment id?
  template <typename T> void set_root(rdma_ptr<T> root) {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    Write(root_ptr, root.raw());
  }

  /// @brief Read the root pointer in MemoryNode 0, Segment 0
  /// @tparam T The type of the root pointer
  /// @return An rdma_ptr<T> pointing to the root object in the RDMA heap
  /// TODO: Should we pass in the memory node and segment id?
  template <typename T> rdma_ptr<T> get_root() {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    auto root = Read<uint64_t>(root_ptr);
    return rdma_ptr<T>(root);
  }

  /// @brief Compare and swap the root pointer in MemoryNode 0, Segment 0
  /// @tparam T The type of the root pointer
  /// @param old_root The expected old root pointer to compare against
  /// @param new_root The new root pointer to set if the old one matches
  /// @return The old root pointer if the CAS was successful, or the current
  /// root
  template <typename T> T cas_root(rdma_ptr<T> old_root, rdma_ptr<T> new_root) {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    return CompareAndSwap(root_ptr, old_root.raw(), new_root.raw());
  }

  /// @brief Fetch-and-add the root pointer in MemoryNode 0, Segment 0
  /// @tparam T The type of the root pointer
  /// @param Add the value to add to the root pointer
  /// @return The old root pointer before the addition
  template <typename T> T faa_root(size_t add) {
    rdma_ptr<uint64_t> root_ptr(compute_node_->get_seg_start(0, 0) +
                                offsetof(internal::ControlBlock, root_));
    return FetchAndAdd(root_ptr, add);
  }

  /// @brief Create a new object of type T in the RDMA heap
  /// @tparam T The type of the object to allocate
  /// @param n The number of elements to allocate, defaults to 1
  /// @return A pointer to the allocated object of type T
  template <typename T> T *New(std::size_t n = 1) {
    auto ptr = allocate<T>(n);
    REMUS_ASSERT(ptr != nullptr, "Failed to allocate memory");
    return (T *)((uintptr_t)ptr);
  }

  /// @brief Delete an object of type T from the RDMA heap
  /// @tparam T The type of the object to delete
  /// @param ptr The pointer to the object to delete
  template <typename T> void Delete(T *ptr) {
    REMUS_ASSERT(ptr != nullptr, "Pointer is nullptr");
    deallocate<T>(rdma_ptr<T>((uintptr_t)ptr));
  }

  /// @brief Schedule a pointer for reclamation ome time in the future
  /// @tparam T The type of the object to reclaim
  /// @param ptr The pointer to the object to reclaim
  template <typename T> void SchedReclaim(T *ptr) {
    REMUS_ASSERT(ptr != nullptr, "Pointer is nullptr");
    scheduled_reclamations.push_back((uintptr_t)ptr);
  }

  /// Reclaim everything that has been scheduled for reclamation
  void ReclaimDeferred() {
    for (uintptr_t x : scheduled_reclamations) {
      Delete<uintptr_t>((uintptr_t *)x);
    }
    scheduled_reclamations.clear();
  }

  /// @brief A count of metrics.  For now, we keep it really simple, but we
  /// probably want histograms and other fine-grained details at some point.
  struct {
    /// @brief A metric for the number of operations and bytes written
    struct metric_t {
      size_t ops;
      size_t bytes;
      metric_t() : ops(0), bytes(0) {}
    };
    metric_t write{};
    metric_t read{};
    uint64_t faa{0};
    uint64_t cas{0};
  } metrics_;

  /// @brief Check for memory leaks in the RDMA heap
  /// @return True if no leaks are detected, false otherwise
  bool inline no_leak_detected() {
    const auto coro_idx = 0;
    REMUS_ASSERT(op_counter_start == op_counter_end,
                 "Leak detected, op_counter_start = {}, op_counter_end = {}",
                 op_counter_start, op_counter_end);
    REMUS_ASSERT(
        seq_op_counter_start[coro_idx] == seq_op_counter_end[coro_idx],
        "Leak detected, seq_op_counter_start = {}, seq_op_counter_end = {}",
        seq_op_counter_start[coro_idx], seq_op_counter_end[coro_idx]);
    REMUS_ASSERT(seq_send_wrs[coro_idx].empty(),
                 "Leak detected, seq_send_wrs[{}] is not empty", coro_idx);
    // For the global staging buffer
    REMUS_ASSERT(staging_buf_start == staging_buf_end,
                 "Leak detected in global staging buffer, start = {}, end = {}",
                 (void *)staging_buf_start, (void *)staging_buf_end);
    // check the allocations map if it's the primary tracking mechanism
    // REMUS_ASSERT(staging_buf_allocations_.empty(),
    //             "Leak detected, staging_buf_allocations_ is not empty, size =
    //             {}", staging_buf_allocations_.size());
    if (!staging_buf_allocations_.empty()) {
      for (auto &[key, value] : staging_buf_allocations_) {
        REMUS_INFO("staging_buf_allocations is not empty, key = {}, in_use = "
                   "{}, next_available_addr = {}",
                   (void *)key, (uint64_t)value.in_use,
                   (void *)value.next_available_addr);
      }
    }

    // For the global cached buffer
    REMUS_ASSERT(cached_buf_start == cached_buf_end,
                 "Leak detected in global cached buffer, start = {}, end = {}",
                 (void *)cached_buf_start, (void *)cached_buf_end);
    // check the allocations map
    REMUS_ASSERT(
        cached_buf_allocations_.empty(),
        "Leak detected, cached_buf_allocations is not empty, size = {}",
        cached_buf_allocations_.size());
    // check the lane_op_counters_
    for (auto &v : compute_node_->lane_op_counters_) {
      REMUS_ASSERT(v.load() == 0,
                   "Leak detected, lane_op_counters_ is not 0, value = "
                   "{}",
                   v.load());
    }
    return true;
  }

protected:
  /// @brief Link the sequence send work requests together
  /// @param seq_idx
  /// @param coro_idx
  inline void link_seq_send_wrs(uint32_t seq_idx, uint32_t coro_idx) {
    for (uint64_t i = 0;
         i < seq_send_wrs[coro_idx][seq_idx].send_wrs.size() - 1; i++) {
      seq_send_wrs[coro_idx][seq_idx].send_wrs[i].wr->next =
          seq_send_wrs[coro_idx][seq_idx].send_wrs[i + 1].wr.get();
    }
    seq_send_wrs[coro_idx][seq_idx].send_wrs.back().wr->next = nullptr;
  }
  /// @brief Get the result of a sequence operation
  /// @tparam T
  /// @param seq_idx
  /// @param coro_idx
  /// @param result
  template <typename T>
  inline void get_seq_op_result(uint32_t seq_idx, uint32_t coro_idx,
                                std::vector<T> &result) {
    for (size_t i = 0; i < seq_send_wrs[coro_idx][seq_idx].staging_bufs.size();
         i++) {
      if (seq_send_wrs[coro_idx][seq_idx].send_wrs[i].wr->opcode !=
          IBV_WR_RDMA_WRITE) {
        result.push_back(
            *(T *)seq_send_wrs[coro_idx][seq_idx].staging_bufs[i]->val());
      }
      REMUS_DEBUG("release {} from seq_send_wrs[{}][{}]",
                  (uint64_t)(uint8_t *)seq_send_wrs[coro_idx][seq_idx]
                      .staging_bufs[i]
                      ->val(),
                  coro_idx, seq_idx);
    }
  }
  /// @brief
  /// @tparam T
  /// @param ptr
  /// @param coro_idx
  /// @return
  template <typename T>
  inline uint32_t find_seq_idx(rdma_ptr<T> ptr, uint32_t coro_idx) {
    /// Use the scheduling policy to select the next connection
    uint32_t seq_idx = 0;
    if (!seq_send_wrs[coro_idx].empty()) {
      auto last_seq_idx =
          (seq_op_counter_end[coro_idx] + args_->uget(CN_OPS_PER_THREAD) - 1) %
          args_->uget(CN_OPS_PER_THREAD);
      REMUS_DEBUG("Debug: seq_op_counter_end[{}] = {}, CN_OPS_PER_THREAD = {}, "
                  "calculated last_seq_idx = {}",
                  coro_idx, seq_op_counter_end[coro_idx],
                  args_->uget(CN_OPS_PER_THREAD), last_seq_idx);
      REMUS_DEBUG("Debug: seq_send_wrs[{}] size = {}", coro_idx,
                  seq_send_wrs[coro_idx].size());
      auto it = seq_send_wrs[coro_idx].find(last_seq_idx);
      if (it != seq_send_wrs[coro_idx].end() && !it->second.posted) {
        REMUS_ASSERT(it->second.send_wrs.size() < args_->uget(CN_WRS_PER_SEQ),
                     "seq_send_wrs[{}] is full, increase the number of "
                     "seq_send_wrs",
                     it->first);
        seq_idx = last_seq_idx;
        REMUS_DEBUG("seq_idx = last_seq_idx = {} is found and not posted",
                    last_seq_idx);
        return seq_idx;
      }
    }
    // There is no seq_idx found or the last seq_idx is posted, create new
    // seq_idx
    REMUS_ASSERT(seq_send_wrs[coro_idx].size() < args_->uget(CN_OPS_PER_THREAD),
                 "seq_send_wrs is full, increase the number of seq_send_wrs");
    auto seq_idx_ptr = std::make_unique<seq_idx_t>(this, coro_idx);
    seq_idx = seq_idx_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].seq_idx = std::move(seq_idx_ptr);
    auto lane_ptr = std::make_unique<Lane>(qp_sched_pol_.get_lane_idx(ptr.id()),
                                           compute_node_->lane_op_counters_);
    seq_send_wrs[coro_idx][seq_idx].lane = std::move(lane_ptr);
    REMUS_DEBUG("seq_send_wrs is empty, add a new seq_idx = {}, lane_idx = {}",
                seq_idx, seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    return seq_idx;
  }
};
} // namespace remus
