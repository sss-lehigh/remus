#pragma once

#include <atomic>
#include <vector>
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

#include <protos/metrics.pb.h>
#include <protos/rdma.pb.h>

#include <spdlog/fmt/fmt.h> // [mfs] Used in remote_ptr... factor away?
#include <vector>

#include "../logging/logging.h"
#include "../metrics/summary.h"
#include "../vendor/sss/status.h"

#include "peer.h"
#include "remote_ptr.h"

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

class Connection;

/// The "remote memory partition map"
///
/// Each node in the system allocates a big region of memory, initializes an
/// allocator in it, and registers it with the RDMA device.  They then exchange
/// the names of these regions with each other.  This partition map keeps track
/// of all of those regions.
///
/// TODO: This isn't just the map... it is also responsible for making the
///       current process's big region of memory.  Can that be refactored?
///
/// TODO: Do we still want the following old documentation?
///
/// Remotely accessible memory backed by either raw memory or hugepages (if
/// enabled). This is just a flat buffer that is preallocated. The user of this
/// memory region can then use it directly, or wrap it around some more complex
/// allocation mechanism.
///
/// TODO: This is used by connection_manager.h... can we refactor?
class RdmaMemory {
  // Handles deleting memory allocated using mmap (when using hugepages)
  struct mmap_deleter {
    void operator()(uint8_t raw[]) { munmap(raw, sizeof(*raw)); }
  };

  static constexpr char kDefaultId[] = "default";

  // Preallocated size.
  const uint64_t capacity_;

  // Either points to an array of bytes allocated with the system allocator or
  // with `mmap`. At some point, this could be replaced with a custom allocator.
  std::variant<std::unique_ptr<uint8_t[]>,
               std::unique_ptr<uint8_t[], mmap_deleter>>
      raw_;

  struct ibv_mr_deleter {
    void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
  };
  using ibv_mr_unique_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

  // A map of memory regions registered with this particular memory.
  std::unordered_map<std::string, ibv_mr_unique_ptr> memory_regions_;

  // Tries to read the number of available hugepages from the system. This is
  // only implemented for Linux-based operating systems.
  inline sss::StatusVal<int> GetNumHugepages(std::string path) {
    // Try to open file.
    std::ifstream file(path);
    if (!file.is_open()) {
      sss::Status err = {sss::Unknown, "Failed to open file: "};
      err << path;
      return {err, {}};
    }

    // Read file.
    int nr_hugepages;
    file >> nr_hugepages;
    if (!file.fail()) {
      return {sss::Status::Ok(), nr_hugepages};
    } else {
      return {{sss::Unknown, "Failed to read nr_hugepages"}, {}};
    }
  }

public:
  static constexpr int kDefaultAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  ~RdmaMemory() { memory_regions_.clear(); }
  RdmaMemory(uint64_t capacity, std::optional<std::string> path,
             ibv_pd *const pd)
      : capacity_(capacity) {
    bool use_hugepages = false;
    if (path.has_value()) {
      auto nr_hugepages = GetNumHugepages(path.value());
      if (nr_hugepages.status.t == sss::Ok && nr_hugepages.val.value() > 0) {
        use_hugepages = true;
      }
    }

    if (!use_hugepages) {
      ROME_TRACE("Not using hugepages; performance might suffer.");
      auto bytes = ((capacity >> 6) + 1) << 6; // Round up to nearest 64B
      raw_ = std::unique_ptr<uint8_t[]>(
          reinterpret_cast<uint8_t *>(std::aligned_alloc(64, bytes)));
      ROME_ASSERT(std::get<0>(raw_) != nullptr, "Allocation failed.");
    } else {
      ROME_INFO("Using hugepages");
      raw_ =
          std::unique_ptr<uint8_t[], mmap_deleter>(reinterpret_cast<uint8_t *>(
              mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)));
      ROME_ASSERT(reinterpret_cast<void *>(std::get<1>(raw_).get()) !=
                      MAP_FAILED,
                  "mmap failed.");
    }
    OK_OR_FAIL(RegisterMemoryRegion(kDefaultId, pd, 0, capacity_));
  }

  RdmaMemory(const RdmaMemory &) = delete;
  RdmaMemory(RdmaMemory &&rm)
      : capacity_(rm.capacity_), raw_(std::move(rm.raw_)),
        memory_regions_(std::move(rm.memory_regions_)) {}

  // Getters.
  uint64_t capacity() const { return capacity_; }

  uint8_t *raw() const {
    return std::visit([](const auto &r) { return r.get(); }, raw_);
  }

  // Creates a new memory region associated with the given protection domain
  // `pd` at the provided offset and with the given length. If a region with the
  // same `id` already exists then it returns `AlreadyExists`.
  sss::Status RegisterMemoryRegion(std::string id, int offset, int length) {
    return RegisterMemoryRegion(id, GetDefaultMemoryRegion()->pd, offset,
                                length);
  }

  sss::Status RegisterMemoryRegion(std::string id, ibv_pd *const pd, int offset,
                                   int length) {
    if (!ValidateRegion(offset, length)) {
      sss::Status err = {sss::FailedPrecondition,
                         "Requested memory region invalid: "};
      err << id;
      return err;
    }

    auto iter = memory_regions_.find(id);
    if (iter != memory_regions_.end()) {
      sss::Status err = {sss::AlreadyExists, "Memory region exists: {}"};
      err << id;
      return err;
    }

    auto *base = reinterpret_cast<uint8_t *>(std::visit(
                     [](const auto &raw) { return raw.get(); }, raw_)) +
                 offset;
    auto mr = ibv_mr_unique_ptr(ibv_reg_mr(pd, base, length, kDefaultAccess));
    if (mr == nullptr) {
      return {sss::InternalError, "Failed to register memory region"};
    }
    memory_regions_.emplace(id, std::move(mr));
    ROME_TRACE("Memory region registered: {} @ {} to {} (length={})", id,
               fmt::ptr(base), fmt::ptr(base + length), length);
    return sss::Status::Ok();
  }

  ibv_mr *GetDefaultMemoryRegion() const {
    return memory_regions_.find(kDefaultId)->second.get();
  }

  sss::StatusVal<ibv_mr *> GetMemoryRegion(std::string id) const {
    auto iter = memory_regions_.find(id);
    if (iter == memory_regions_.end()) {
      sss::Status err = {sss::NotFound, "Memory region not found: "};
      err << id;
      return {err, {}};
    }
    return {sss::Status::Ok(), iter->second.get()};
  }

private:
  // Validates that the given offset and length are not ill formed w.r.t. to the
  // capacity of this memory.
  bool ValidateRegion(int offset, int length) {
    if (offset < 0 || length < 0)
      return false;
    if (offset + length > capacity_)
      return false;
    return true;
  }
};

/// Specialization of a `memory_resource` that wraps RDMA accessible memory.
///
/// TODO: This is only used by this file, so it doesn't need to be publicly
///       visible
class rdma_memory_resource : public std::experimental::pmr::memory_resource {
public:
  virtual ~rdma_memory_resource() {}
  explicit rdma_memory_resource(size_t bytes, ibv_pd *pd)
      : rdma_memory_(std::make_unique<RdmaMemory>(
            bytes, "/proc/sys/vm/nr_hugepages", pd)),
        head_(rdma_memory_->raw() + bytes) {
    std::memset(alignments_.data(), 0, sizeof(alignments_));
    ROME_TRACE("rdma_memory_resource: {} to {} (length={})",
               fmt::ptr(rdma_memory_->raw()), fmt::ptr(head_.load()), bytes);
  }

  rdma_memory_resource(const rdma_memory_resource &) = delete;
  rdma_memory_resource &operator=(const rdma_memory_resource &) = delete;
  ibv_mr *mr() const { return rdma_memory_->GetDefaultMemoryRegion(); }

private:
  static constexpr uint8_t kMinSlabClass = 3;
  static constexpr uint8_t kMaxSlabClass = 20;
  static constexpr uint8_t kNumSlabClasses = kMaxSlabClass - kMinSlabClass + 1;
  static constexpr size_t kMaxAlignment = 1 << 8;
  static constexpr char kLogTable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
      -1,    0,     1,     1,     2,     2,     2,     2,
      3,     3,     3,     3,     3,     3,     3,     3,
      LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6), LT(7),
      LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7),
  };

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

  // Returns a region of RDMA-accessible memory that satisfies the given memory
  // allocation request of `bytes` with `alignment`. First, it checks whether
  // there exists a region in `freelists_` that satisfies the request, then it
  // attempts to allocate a new region. If the request cannot be satisfied, then
  // `nullptr` is returned.
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

  std::unique_ptr<RdmaMemory> rdma_memory_;
  std::atomic<uint8_t *> head_;

  // Stores addresses of freed memory for a given slab class.
  inline static thread_local std::array<uint8_t, kNumSlabClasses> alignments_;
  inline static thread_local std::array<std::list<std::pair<size_t, void *>>,
                                        kNumSlabClasses>
      freelists_;
};

/// An allocator wrapping `rdma_memory_resource` to be used to allocate new
/// RDMA-accessible memory.
///
/// TODO: This is only used by this file, so it doesn't need to be publicly
///       visible
template <typename T> class rdma_allocator {
public:
  typedef T value_type;

  rdma_allocator() : memory_resource_(nullptr) {}
  rdma_allocator(rdma_memory_resource *memory_resource)
      : memory_resource_(memory_resource) {}
  rdma_allocator(const rdma_allocator &other) = default;

  template <typename U>
  rdma_allocator(const rdma_allocator<U> &other) noexcept {
    memory_resource_ = other.memory_resource();
  }

  rdma_allocator &operator=(const rdma_allocator &) = delete;

  // Getters
  rdma_memory_resource *memory_resource() const { return memory_resource_; }

  [[nodiscard]] constexpr T *allocate(std::size_t n = 1) {
    return reinterpret_cast<T *>(memory_resource_->allocate(sizeof(T) * n, 64));
  }

  constexpr void deallocate(T *p, std::size_t n = 1) {
    memory_resource_->deallocate(reinterpret_cast<T *>(p), sizeof(T) * n, 64);
  }

private:
  rdma_memory_resource *memory_resource_;
};

/// TODO: This is templated on the CM to break a circular dependence without
///       resorting to virtual methods.  There's probably a better approach?
template <class CM> class MemoryPool {
private:
  static inline void cpu_relax() { asm volatile("pause\n" ::: "memory"); }

  struct conn_info_t {
    Connection *conn;
    uint32_t rkey;
    uint32_t lkey;
  };

  Peer self_;
  /// Used to protect the id generator and the thread_ids
  std::mutex control_lock_;
  /// Used to protect the rdma_per_read_ summary statistics thing
  std::mutex rdma_per_read_lock_;
  /// A counter to increment
  uint64_t id_gen = 0;
  /// A mapping of thread id to an index into the reordering_semaphores array. Passed as the wr_id in work requests.
  std::unordered_map<std::thread::id, uint64_t> thread_ids;
  /// a vector of semaphores, one for each thread that can send an operation. Threads will use this to recover from polling another thread's wr_id
  std::array<std::atomic<int>, 20> reordering_counters;

  std::unique_ptr<CM> connection_manager_;
  std::unique_ptr<rdma_memory_resource> rdma_memory_;
  ibv_mr *mr_;

  std::unordered_map<uint16_t, conn_info_t> conn_info_;

  rome::metrics::Summary<size_t> rdma_per_read_;

public:
  MemoryPool(const Peer &self, std::unique_ptr<CM> connection_manager)
      : self_(self), connection_manager_(std::move(connection_manager)),
        rdma_per_read_("rdma_per_read", "ops", 10000) {}

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;

  CM *connection_manager() const { return connection_manager_.get(); }
  rome::metrics::MetricProto rdma_per_read_proto() {
    return rdma_per_read_.ToProto();
  }
  conn_info_t conn_info(uint16_t id) const { return conn_info_.at(id); }

  /// This method does two things.
  /// - It creates a memory region with `capacity` as its size.
  /// - It does an all-all communication with every peer, to create a connection
  ///   with each, and then it exchanges regions with all peers.
  ///
  /// TODO: Should there be some kind of "shutdown()" method?
  ///
  /// [mfs] This method is the *only* reason we need memory_pool.h to know about
  ///       ConnectionManager.  If we made this something that CM did, and then
  ///       gave memory chunks over to the MemoryPool, we could break the
  ///       circular dependence.  That might also let us turn MemoryPool into a
  ///       single-responsibility object.
  inline sss::Status Init(uint32_t capacity, const std::vector<Peer> &peers) {
    auto status = connection_manager_->Start(self_.address, self_.port);
    RETURN_STATUS_ON_ERROR(status);

    // Create a memory region (mr) in the current protection domain (pd)
    rdma_memory_ = std::make_unique<rdma_memory_resource>(
        capacity + sizeof(uint64_t), connection_manager_->pd());
    mr_ = rdma_memory_->mr();

    // Go through the list of peers and connect to each of them
    for (const auto &p : peers) {
      auto connected = connection_manager_->Connect(p.id, p.address, p.port);
      while (connected.status.t == sss::Unavailable) {
        connected = connection_manager_->Connect(p.id, p.address, p.port);
      }
      RETURN_STATUSVAL_ON_ERROR(connected);
    }

    // Send the memory region to all peers
    RemoteObjectProto rm_proto;
    rm_proto.set_rkey(mr_->rkey);
    rm_proto.set_raddr(reinterpret_cast<uint64_t>(mr_->addr));
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      STATUSVAL_OR_DIE(conn);
      status = conn.val.value()->channel()->Send(rm_proto);
      RETURN_STATUS_ON_ERROR(status);
    }

    // Get all peers' memory regions
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      STATUSVAL_OR_DIE(conn);
      auto got =
          conn.val.value()->channel()->template Deliver<RemoteObjectProto>();
      RETURN_STATUSVAL_ON_ERROR(got);
      // [mfs] I don't understand why we use mr_->lkey?
      conn_info_.emplace(p.id, conn_info_t{conn.val.value(),
                                           got.val.value().rkey(), mr_->lkey});
    }

    return {sss::Ok, {}};
  }

  void RegisterThread() {
    control_lock_.lock();
    std::thread::id mid = std::this_thread::get_id();
    if (this->thread_ids.find(mid) != this->thread_ids.end()) {
      ROME_FATAL("Cannot register the same thread twice");
      return;
    }
    if (this->id_gen >= THREAD_MAX){
      ROME_FATAL("Hit upper limit on THREAD_MAX. todo: fix this condition");
      return;
    }
    this->thread_ids.insert(std::make_pair(mid, this->id_gen));
    this->reordering_counters[this->id_gen] = 0;
    this->id_gen++;
    // this->reordering_counters.resize(this->id_gen);
    control_lock_.unlock();
  }

  /// Allocate some memory from the local RDMA heap
  template <typename T> remote_ptr<T> Allocate(size_t size = 1) {
    auto ret = remote_ptr<T>(
        self_.id, rdma_allocator<T>(rdma_memory_.get()).allocate(size));
    return ret;
  }

  /// Deallocate some memory to the local RDMA heap (must be from this node)
  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1) {
    ROME_ASSERT(p.id() == self_.id,
                "Alloc/dealloc on remote node not implemented...");
    rdma_allocator<T>(rdma_memory_.get()).deallocate(std::to_address(p), size);
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
      auto alloc = rdma_allocator<T>(rdma_memory_.get());
      local = alloc.allocate();
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
    sge.lkey = mr_->lkey;

    ibv_send_wr send_wr_{};
    send_wr_.wr_id = index_as_id;
    send_wr_.num_sge = 1;
    send_wr_.sg_list = &sge;
    send_wr_.opcode = IBV_WR_RDMA_WRITE;
    send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    send_wr_.wr.rdma.remote_addr = ptr.address();
    send_wr_.wr.rdma.rkey = info.rkey;

    ibv_send_wr *bad = nullptr;
    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = 1;
    // make the send
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
    // TODO: [esl] poll for more than 1
    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})", (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)), (std::stringstream() << ptr).str());
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    if (prealloc == remote_nullptr) {
      auto alloc = rdma_allocator<T>(rdma_memory_.get());
      alloc.deallocate(local);
    }
  }

  /// Do a 64-bit swap over RDMA
  ///
  /// [mfs] If the swap value is always a uint64_t, why is this templated on T?
  template <typename T>
  T AtomicSwap(remote_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    auto alloc = rdma_allocator<uint64_t>(rdma_memory_.get());
    // [esl] There is probably a better way to avoid allocating every time we do this call (maybe be preallocating the space thread_local)
    volatile uint64_t *prev_ = alloc.allocate();

    ibv_sge sge{.addr = reinterpret_cast<uint64_t>(prev_),
                .length = sizeof(uint64_t),
                .lkey = mr_->lkey};

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

    ibv_send_wr *bad = nullptr;
    while (true) {
      // set the counter to the number of work completions we expect
      reordering_counters[index_as_id] = 1;
      RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
      
      // Poll until we match on the condition
      ibv_wc wc;
      while (reordering_counters[index_as_id] != 0) {
        int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
        if (poll == 0 || (poll < 0 && errno == EAGAIN))
          continue;
        // Assert a good result
        ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}", (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
        int old = reordering_counters[wc.wr_id].fetch_sub(1);
        ROME_ASSERT(old >= 1, "Broken synchronization");
      }

      if (*prev_ == send_wr_.wr.atomic.compare_add)
        break;
      send_wr_.wr.atomic.compare_add = *prev_;
    };
    T ret = T(*prev_);
    alloc.deallocate((uint64_t *) prev_, 8);
    return ret;
  }

  /// Do a 64-bit CAS over RDMA
  ///
  /// [mfs] If the swap value is always a uint64_t, why is this templated on T?
  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    // [esl] Getting the thread's index to determine it's owned flag
    uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

    auto alloc = rdma_allocator<uint64_t>(rdma_memory_.get());
    // [esl] There is probably a better way to avoid allocating every time we do this call (maybe be preallocating the space thread_local)
    volatile uint64_t *prev_ = alloc.allocate();
    
    // TODO: would the code be clearer if all of the ibv_* initialization
    // throughout this file used the new syntax?
    // [esl] I agree, i think its much cleaner
    ibv_sge sge{.addr = reinterpret_cast<uint64_t>(prev_),
                .length = sizeof(uint64_t),
                .lkey = mr_->lkey};

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

    ibv_send_wr *bad = nullptr;
    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = 1;
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);

    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}", (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    // ROME_TRACE("CompareAndSwap: expected={:x}, swap={:x}, actual={:x}  (id={})", expected, swap, *prev_, static_cast<uint64_t>(self_.id));
    T ret = T(*prev_);
    alloc.deallocate((uint64_t *)prev_, 8);
    return ret;
  }

  template <typename T> inline remote_ptr<T> GetRemotePtr(const T *ptr) const {
    return remote_ptr<T>(self_.id, reinterpret_cast<uint64_t>(ptr));
  }

  template <typename T> inline remote_ptr<T> GetBaseAddress() const {
    return GetRemotePtr<T>(reinterpret_cast<const T *>(mr_->addr));
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
      sges[i].lkey = mr_->lkey;

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

    ibv_send_wr *bad;
    // set the counter to the number of work completions we expect
    reordering_counters[index_as_id] = num_chunks;
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, wrs, &bad);

    // Poll until we match on the condition
    ibv_wc wc;
    while (reordering_counters[index_as_id] != 0) {
      int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // Assert a good result
      ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}", (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)), ptr);
      int old = reordering_counters[wc.wr_id].fetch_sub(1);
      ROME_ASSERT(old >= 1, "Broken synchronization");
    }

    // Update rdma per read
    rdma_per_read_lock_.lock();
    rdma_per_read_ << num_chunks;
    rdma_per_read_lock_.unlock();
  }

  // [mfs]  According to [el], it is possible to post multiple requests on the
  //        same qp, and they'll finish in order, so we definitely will want a
  //        way to let that happen.
  // [esl] Just to cite my sources:
  // https://www.rdmamojo.com/2013/07/26/libibverbs-thread-safe-level/ (Thread safe)
  // https://www.rdmamojo.com/2013/01/26/ibv_post_send/ (Ordering guarantee, for RC only)
  //    "In RC QP, there is a PSN (Packet Serial Number) that guarantees the order of the messages"
  // https://www.rdmamojo.com/2013/06/01/which-queue-pair-type-to-use/ (Ordering guarantee also mentioned her)
};

} // namespace rome::rdma::internal