#pragma once

// [mfs]  Inasmuch as remote_ptr and remote_nullptr are just utility wrappers,
//        `MemoryPool` is the entire surface of the API between an application
//        and rome::rdma.  Thus it is essential that we fully understand this
//        API, and document it well.

#include <cstdint>
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>

#include <protos/metrics.pb.h>
#include <protos/rdma.pb.h>

#include <spdlog/fmt/fmt.h> // [mfs] Used in remote_ptr... factor away?

#include "../logging/logging.h"
#include "../metrics/summary.h"
#include "../vendor/sss/status.h"
#include "channel.h"
#include "connection_manager.h"
#include "messenger.h"
#include "rmalloc.h"

namespace rome::rdma {

template <typename T> class remote_ptr;
struct nullptr_type {};
typedef remote_ptr<nullptr_type> remote_nullptr_t;

template <typename T> class remote_ptr {
public:
  using element_type = T;
  using pointer = T *;
  using reference = T &;
  using id_type = uint16_t;
  using address_type = uint64_t;

  template <typename U> using rebind = remote_ptr<U>;

  // Constructors
  constexpr remote_ptr() : raw_(0) {}
  explicit remote_ptr(uint64_t raw) : raw_(raw) {}
  remote_ptr(uint64_t id, T *address)
      : remote_ptr(id, reinterpret_cast<uint64_t>(address)) {}
  remote_ptr(id_type id, uint64_t address)
      : raw_((((uint64_t)id) << kAddressBits) | (address & kAddressBitmask)) {}

  // Copy and Move
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  remote_ptr(const remote_ptr &p) : raw_(p.raw_) {}
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  remote_ptr(remote_ptr &&p) : raw_(p.raw_) {}
  constexpr remote_ptr(const remote_nullptr_t &n) : raw_(0) {}

  // Getters
  uint64_t id() const { return (raw_ & kIdBitmask) >> kAddressBits; }
  uint64_t address() const { return raw_ & kAddressBitmask; }
  uint64_t raw() const { return raw_; }

  // Assignment
  void operator=(const remote_ptr &p) volatile { raw_ = p.raw_; }
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  void operator=(const remote_nullptr_t &n) volatile {
    raw_ = 0;
  }

  // Increment operator
  remote_ptr &operator+=(size_t s) {
    const auto address = (raw_ + (sizeof(element_type) * s)) & kAddressBitmask;
    raw_ = (raw_ & kIdBitmask) | address;
    return *this;
  }
  remote_ptr &operator++() {
    *this += 1;
    return *this;
  }
  remote_ptr operator++(int) {
    remote_ptr prev = *this;
    *this += 1;
    return prev;
  }

  // Conversion operators
  explicit operator uint64_t() const { return raw_; }
  template <typename U> explicit operator remote_ptr<U>() const {
    return remote_ptr<U>(raw_);
  }

  // Pointer-like functions
  static constexpr element_type *to_address(const remote_ptr &p) {
    return (element_type *)p.address();
  }
  static constexpr remote_ptr pointer_to(element_type &p) {
    return remote_ptr(-1, &p);
  }
  pointer get() const { return (element_type *)address(); }
  pointer operator->() const noexcept { return (element_type *)address(); }
  reference operator*() const noexcept { return *((element_type *)address()); }

  // Stream operator
  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const remote_ptr<U> &p);

  // Equivalence
  bool operator==(const volatile remote_nullptr_t &n) const volatile {
    return raw_ == 0;
  }
  bool operator==(remote_ptr &n) { return raw_ == n.raw_; }
  template <typename U>
  friend bool operator==(remote_ptr<U> &p1, remote_ptr<U> &p2);
  template <typename U>
  friend bool operator==(const volatile remote_ptr<U> &p1,
                         const volatile remote_ptr<U> &p2);

  bool operator<(const volatile remote_ptr<T> &p) { return raw_ < p.raw_; }
  friend bool operator<(const volatile remote_ptr<T> &p1,
                        const volatile remote_ptr<T> &p2) {
    return p1.raw() < p2.raw();
  }

private:
  static inline constexpr uint64_t bitsof(const uint32_t &bytes) {
    return bytes * 8;
  }

  static constexpr auto kAddressBits =
      (bitsof(sizeof(uint64_t))) - bitsof(sizeof(id_type));
  static constexpr auto kAddressBitmask = ((1ul << kAddressBits) - 1);
  static constexpr auto kIdBitmask = (uint64_t)(-1) ^ kAddressBitmask;

  uint64_t raw_;
};

constexpr remote_nullptr_t remote_nullptr{};

template <typename U>
std::ostream &operator<<(std::ostream &os, const remote_ptr<U> &p) {
  return os << "<id=" << p.id() << ", address=0x" << std::hex << p.address()
            << std::dec << ">";
}

template <typename U>
bool operator==(const volatile remote_ptr<U> &p1,
                const volatile remote_ptr<U> &p2) {
  return p1.raw_ == p2.raw_;
}
} // namespace rome::rdma

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

namespace rome::rdma {

// [mfs] This is a concerning "secret" configuration parameter
#define THREAD_MAX 50

class MemoryPool {
  // [mfs] These should probably be template parameters
  static constexpr size_t kMemoryPoolMessengerCapacity = 1 << 12;
  static constexpr size_t kMemoryPoolMessageSize = 1 << 8;

public:
  struct Peer {
    uint16_t id;
    std::string address;
    uint16_t port;

    Peer() : Peer(0, "", 0) {}
    Peer(uint16_t id, std::string address, uint16_t port)
        : id(id), address(address), port(port) {}
  };

  using channel_type =
      RdmaChannel<TwoSidedRdmaMessenger<kMemoryPoolMessengerCapacity,
                                        kMemoryPoolMessageSize>>;

  using cm_type = ConnectionManager<channel_type>;

  using conn_type = cm_type::conn_type;

private:
  static inline void cpu_relax() { asm volatile("pause\n" ::: "memory"); }

  struct conn_info_t {
    conn_type *conn;
    uint32_t rkey;
    uint32_t lkey;
  };

  Peer self_;

  volatile uint64_t *prev_ = nullptr;

  std::unique_ptr<ConnectionManager<channel_type>> connection_manager_;
  std::unique_ptr<rdma_memory_resource> rdma_memory_;
  ibv_mr *mr_;

  std::unordered_map<uint16_t, conn_info_t> conn_info_;
  ibv_send_wr send_wr_{};

  rome::metrics::Summary<size_t> rdma_per_read_;

public:
  MemoryPool(
      const Peer &self,
      std::unique_ptr<ConnectionManager<channel_type>> connection_manager)
      : self_(self), connection_manager_(std::move(connection_manager)),
        rdma_per_read_("rdma_per_read", "ops", 10000) {}

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;

  cm_type *connection_manager() const { return connection_manager_.get(); }
  rome::metrics::MetricProto rdma_per_read_proto() {
    return rdma_per_read_.ToProto();
  }
  conn_info_t conn_info(uint16_t id) const { return conn_info_.at(id); }

  inline sss::Status Init(uint32_t capacity, const std::vector<Peer> &peers) {
    auto status = connection_manager_->Start(self_.address, self_.port);
    RETURN_STATUS_ON_ERROR(status);
    rdma_memory_ = std::make_unique<rdma_memory_resource>(
        capacity + sizeof(uint64_t), connection_manager_->pd());
    mr_ = rdma_memory_->mr();

    for (const auto &p : peers) {
      auto connected = connection_manager_->Connect(p.id, p.address, p.port);
      while (connected.status.t == sss::Unavailable) {
        connected = connection_manager_->Connect(p.id, p.address, p.port);
      }
      RETURN_STATUSVAL_ON_ERROR(connected);
    }

    RemoteObjectProto rm_proto;
    rm_proto.set_rkey(mr_->rkey);
    rm_proto.set_raddr(reinterpret_cast<uint64_t>(mr_->addr));
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      STATUSVAL_OR_DIE(conn);
      status = conn.val.value()->channel()->Send(rm_proto);
      RETURN_STATUS_ON_ERROR(status);
    }

    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      STATUSVAL_OR_DIE(conn);
      auto got = conn.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      while (!(got.status.t == sss::Ok) && got.status.t == sss::Unavailable) {
        got = conn.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      }
      RETURN_STATUSVAL_ON_ERROR(got);
      conn_info_.emplace(p.id, conn_info_t{conn.val.value(),
                                           got.val.value().rkey(), mr_->lkey});
    }

    return {sss::Ok, {}};
  }

  template <typename T> remote_ptr<T> Allocate(size_t size = 1) {
    // ROME_INFO("Allocating {} bytes ({} {} times)", sizeof(T)*size, sizeof(T),
    // size);
    auto ret = remote_ptr<T>(
        self_.id, rdma_allocator<T>(rdma_memory_.get()).allocate(size));
    return ret;
  }

  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1) {
    // ROME_INFO("Deallocating {} bytes ({} {} times)", sizeof(T)*size,
    // sizeof(T), size); else ROME_INFO("Deallocating {} bytes", sizeof(T));
    ROME_ASSERT(p.id() == self_.id,
                "Alloc/dealloc on remote node not implemented...");
    rdma_allocator<T>(rdma_memory_.get()).deallocate(std::to_address(p), size);
  }

  template <typename T>
  remote_ptr<T> Read(remote_ptr<T> ptr,
                     remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>();
    ReadInternal(ptr, 0, sizeof(T), sizeof(T), prealloc);
    return prealloc;
  }

  template <typename T>
  remote_ptr<T> ExtendedRead(remote_ptr<T> ptr, int size,
                             remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>(size);
    // TODO: What happens if I decrease chunk size (* size to sizeT)
    ReadInternal(ptr, 0, sizeof(T) * size, sizeof(T) * size, prealloc);
    return prealloc;
  }

  template <typename T>
  remote_ptr<T> PartialRead(remote_ptr<T> ptr, size_t offset, size_t bytes,
                            remote_ptr<T> prealloc = remote_nullptr) {
    if (prealloc == remote_nullptr)
      prealloc = Allocate<T>();
    ReadInternal(ptr, offset, bytes, sizeof(T), prealloc);
    return prealloc;
  }

  template <typename T>
  void Write(remote_ptr<T> ptr, const T &val,
             remote_ptr<T> prealloc = remote_nullptr) {
    ROME_DEBUG("Write: {:x} @ {}", (uint64_t)val, ptr);
    auto info = conn_info_.at(ptr.id());

    T *local;
    if (prealloc == remote_nullptr) {
      auto alloc = rdma_allocator<T>(rdma_memory_.get());
      local = alloc.allocate();
      ROME_DEBUG("Allocated memory for Write: {} bytes @ 0x{:x}", sizeof(T),
                 (uint64_t)local);
    } else {
      local = std::to_address(prealloc);
      ROME_DEBUG("Preallocated memory for Write: {} bytes @ 0x{:x}", sizeof(T),
                 (uint64_t)local);
    }

    ROME_ASSERT((uint64_t)local != ptr.address(), "WTF");
    std::memset(local, 0, sizeof(T));
    *local = val;
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local);
    sge.length = sizeof(T);
    sge.lkey = mr_->lkey;

    send_wr_.num_sge = 1;
    send_wr_.sg_list = &sge;
    send_wr_.opcode = IBV_WR_RDMA_WRITE;
    send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    send_wr_.wr.rdma.remote_addr = ptr.address();
    send_wr_.wr.rdma.rkey = info.rkey;

    ibv_send_wr *bad = nullptr;
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
    ibv_wc wc;
    auto poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
    while (poll == 0 || (poll < 0 && errno == EAGAIN)) {
      poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
    }

    if (prealloc == remote_nullptr) {
      auto alloc = rdma_allocator<T>(rdma_memory_.get());
      alloc.deallocate(local);
    }
    ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                "ibv_poll_cq(): {} ({})",
                (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                (std::stringstream() << ptr).str());
  }

  template <typename T>
  T AtomicSwap(remote_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(prev_);
    sge.length = sizeof(uint64_t);
    sge.lkey = mr_->lkey;

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
      RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
      ibv_wc wc;
      auto poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      while (poll == 0 || (poll < 0 && errno == EAGAIN)) {
        poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      }
      ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
                  (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));

      ROME_DEBUG("Swap: expected={:x}, swap={:x}, prev={:x} (id={})",
                 send_wr_.wr.atomic.compare_add, (uint64_t)swap, *prev_,
                 self_.id);
      if (*prev_ == send_wr_.wr.atomic.compare_add)
        break;
      send_wr_.wr.atomic.compare_add = *prev_;
    };
    return T(*prev_);
  }

  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    static_assert(sizeof(T) == 8);
    auto info = conn_info_.at(ptr.id());

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(prev_);
    sge.length = sizeof(uint64_t);
    sge.lkey = mr_->lkey;

    send_wr_.num_sge = 1;
    send_wr_.sg_list = &sge;
    send_wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    send_wr_.wr.atomic.remote_addr = ptr.address();
    send_wr_.wr.atomic.rkey = info.rkey;
    send_wr_.wr.atomic.compare_add = expected;
    send_wr_.wr.atomic.swap = swap;

    ibv_send_wr *bad = nullptr;
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
    ibv_wc wc;
    auto poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
    while (poll == 0 || (poll < 0 && errno == EAGAIN)) {
      poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
    }
    ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
                (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
    ROME_DEBUG("CompareAndSwap: expected={:x}, swap={:x}, actual={:x}  (id={})",
               expected, swap, *prev_, static_cast<uint64_t>(self_.id));
    return T(*prev_);
  }

  template <typename T> inline remote_ptr<T> GetRemotePtr(const T *ptr) const {
    return remote_ptr<T>(self_.id, reinterpret_cast<uint64_t>(ptr));
  }

  template <typename T> inline remote_ptr<T> GetBaseAddress() const {
    return GetRemotePtr<T>(reinterpret_cast<const T *>(mr_->addr));
  }

private:
  template <typename T>
  void ReadInternal(remote_ptr<T> ptr, size_t offset, size_t bytes,
                    size_t chunk_size, remote_ptr<T> prealloc) {
    const int num_chunks =
        bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
    const size_t remainder = bytes % chunk_size;
    const bool is_multiple = remainder == 0;

    auto info = conn_info_.at(ptr.id());

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
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, wrs, &bad);
    ibv_wc wc;
    int poll = 0;

    // [mfs] Removed "kill"
    for (; poll == 0; poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc))
      ;
    ROME_ASSERT(
        poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
        (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)), ptr);

    rdma_per_read_ << num_chunks;
  }

  // [mfs]  According to [el], it is possible to post multiple requests on the
  //        same qp, and they'll finish in order, so we definitely will want a
  //        way to let that happen.
};

} // namespace rome::rdma