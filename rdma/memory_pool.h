#pragma once

#include <asm-generic/errno-base.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <protos/rdma.pb.h>

#include "../../metrics/summary.h"
#include "../../util/thread_util.h"
#include "../../vendor/sss/status.h"
#include "connection_manager.h"
#include "memory_pool.h"
#include "messenger.h"
#include "remote_ptr.h"
#include "rmalloc.h"

#define THREAD_MAX 50

namespace rome::rdma {

using ::rome::rdma::RdmaChannel;
using ::rome::rdma::RemoteObjectProto;
using ::rome::rdma::TwoSidedRdmaMessenger;

class MemoryPool {
#ifndef ROME_MEMORY_POOL_MESSENGER_CAPACITY
  static constexpr size_t kMemoryPoolMessengerCapacity = 1 << 12;
#else
  static constexpr size_t kMemoryPoolMessengerCapacity =
      ROME_MEMORY_POOL_MESSENGER_CAPACITY;
#endif
#ifndef ROME_MEMORY_POOL_MESSAGE_SIZE
  static constexpr size_t kMemoryPoolMessageSize = 1 << 8;
#else
  static constexpr size_t kMemoryPoolMessageSize =
      ROME_MEMORY_POOL_MESSAGE_SIZE;
#endif
public:
  typedef RdmaChannel<TwoSidedRdmaMessenger<kMemoryPoolMessengerCapacity,
                                            kMemoryPoolMessageSize>>
      channel_type;
  typedef ConnectionManager<channel_type> cm_type;
  typedef cm_type::conn_type conn_type;

  struct Peer {
    uint16_t id;
    std::string address;
    uint16_t port;

    Peer() : Peer(0, "", 0) {}
    Peer(uint16_t id, std::string address, uint16_t port)
        : id(id), address(address), port(port) {}
  };

  struct conn_info_t {
    conn_type *conn;
    uint32_t rkey;
    uint32_t lkey;
  };

  inline MemoryPool(
      const Peer &self,
      std::unique_ptr<ConnectionManager<channel_type>> connection_manager);

  class DoorbellBatch {
  public:
    ~DoorbellBatch() {
      delete wrs_;
      delete[] sges_;
    }

    explicit DoorbellBatch(const conn_info_t &conn_info, int capacity)
        : conn_info_(conn_info), capacity_(capacity) {
      wrs_ = new ibv_send_wr[capacity];
      sges_ = new ibv_sge *[capacity];
      std::memset(wrs_, 0, sizeof(ibv_send_wr) * capacity);
      wrs_[capacity - 1].send_flags = IBV_SEND_SIGNALED;
      for (auto i = 1; i < capacity; ++i) {
        wrs_[i - 1].next = &wrs_[i];
      }
    }

    std::pair<ibv_send_wr *, ibv_sge *> Add(int num_sge = 1) {
      if (size_ == capacity_)
        return {nullptr, nullptr};
      auto *sge = new ibv_sge[num_sge];
      sges_[size_] = sge;
      auto wr = &wrs_[size_];
      wr->num_sge = num_sge;
      wr->sg_list = sge;
      std::memset(sge, 0, sizeof(*sge) * num_sge);
      ++size_;
      return {wr, sge};
    }

    void SetKillSwitch(std::atomic<bool> *kill_switch) {
      kill_switch_ = kill_switch;
    }

    ibv_send_wr *GetWr(int i) { return &wrs_[i]; }

    inline int size() const { return size_; }
    inline int capacity() const { return capacity_; }
    inline conn_info_t conn_info() const { return conn_info_; }
    inline bool is_mortal() const { return kill_switch_ != nullptr; }

    friend class MemoryPool;

  private:
    conn_info_t conn_info_;

    int size_ = 0;
    const int capacity_;

    ibv_send_wr *wrs_;
    ibv_sge **sges_;
    std::atomic<bool> *kill_switch_ = nullptr;
  };

  class DoorbellBatchBuilder {
  public:
    DoorbellBatchBuilder(MemoryPool *pool, uint16_t id, int num_ops = 1)
        : pool_(pool) {
      batch_ = std::make_unique<DoorbellBatch>(pool->conn_info(id), num_ops);
    }

    template <typename T>
    remote_ptr<T> AddRead(remote_ptr<T> rptr, bool fence = false,
                          remote_ptr<T> prealloc = remote_nullptr);

    template <typename T>
    remote_ptr<T> AddPartialRead(remote_ptr<T> ptr, size_t offset, size_t bytes,
                                 bool fence,
                                 remote_ptr<T> prealloc = remote_nullptr);

    template <typename T>
    void AddWrite(remote_ptr<T> rptr, const T &t, bool fence = false);

    template <typename T>
    void AddWrite(remote_ptr<T> rptr, remote_ptr<T> prealloc = remote_nullptr,
                  bool fence = false);

    template <typename T>
    void AddWriteBytes(remote_ptr<T> rptr, remote_ptr<T> prealloc, int bytes,
                       bool fence = false);

    void AddKillSwitch(std::atomic<bool> *kill_switch) {
      batch_->SetKillSwitch(kill_switch);
    }

    std::unique_ptr<DoorbellBatch> Build();

  private:
    template <typename T>
    void AddReadInternal(remote_ptr<T> rptr, size_t offset, size_t bytes,
                         size_t chunk, bool fence, remote_ptr<T> prealloc);
    std::unique_ptr<DoorbellBatch> batch_;
    MemoryPool *pool_;
  };

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;

  // Getters.
  cm_type *connection_manager() const { return connection_manager_.get(); }
  rome::metrics::MetricProto rdma_per_read_proto() {
    return rdma_per_read_.ToProto();
  }
  conn_info_t conn_info(uint16_t id) const { return conn_info_.at(id); }

  inline sss::Status Init(uint32_t capacity, const std::vector<Peer> &peers);

  template <typename T> remote_ptr<T> Allocate(size_t size = 1);

  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1);

  void Execute(DoorbellBatch *batch);

  template <typename T>
  remote_ptr<T> Read(remote_ptr<T> ptr, remote_ptr<T> prealloc = remote_nullptr,
                     std::atomic<bool> *kill = nullptr);

  template <typename T>
  remote_ptr<T> ExtendedRead(remote_ptr<T> ptr, int size,
                             remote_ptr<T> prealloc = remote_nullptr,
                             std::atomic<bool> *kill = nullptr);

  template <typename T>
  remote_ptr<T> PartialRead(remote_ptr<T> ptr, size_t offset, size_t bytes,
                            remote_ptr<T> prealloc = remote_nullptr);

  template <typename T>
  void Write(remote_ptr<T> ptr, const T &val,
             remote_ptr<T> prealloc = remote_nullptr);

  template <typename T>
  T AtomicSwap(remote_ptr<T> ptr, uint64_t swap, uint64_t hint = 0);

  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap);

  template <typename T> inline remote_ptr<T> GetRemotePtr(const T *ptr) const {
    return remote_ptr<T>(self_.id, reinterpret_cast<uint64_t>(ptr));
  }

  template <typename T> inline remote_ptr<T> GetBaseAddress() const {
    return GetRemotePtr<T>(reinterpret_cast<const T *>(mr_->addr));
  }

  /// @brief Identify an op thread to the service "worker" thread. (Must be done
  /// before operations can be run)
  void RegisterThread();

  // Do I need this?
  void KillWorkerThread();

private:
  template <typename T>
  inline void ReadInternal(remote_ptr<T> ptr, size_t offset, size_t bytes,
                           size_t chunk_size, remote_ptr<T> prealloc,
                           std::atomic<bool> *kill = nullptr);

  void WorkerThread();

  Peer self_;

  std::mutex control_lock_;
  std::mutex rdma_per_read_lock_;

  uint64_t id_gen = 0;
  std::unordered_set<uint64_t> wr_ids;
  std::unordered_map<std::thread::id, uint64_t> thread_ids;
  std::condition_variable cond_vars[THREAD_MAX]; // max of "THREAD_MAX" threads,
                                                 // can trivially increase
  std::atomic<bool> mailboxes[THREAD_MAX];
  bool run_worker = true;
  std::mutex mutex_vars[THREAD_MAX];

  std::unique_ptr<ConnectionManager<channel_type>> connection_manager_;
  std::unique_ptr<rdma_memory_resource> rdma_memory_;
  ibv_mr *mr_;

  std::unordered_map<uint16_t, conn_info_t> conn_info_;

  rome::metrics::Summary<size_t> rdma_per_read_;
};

} // namespace rome::rdma

namespace rome::rdma {

template <typename T>
remote_ptr<T>
MemoryPool::DoorbellBatchBuilder::AddRead(remote_ptr<T> rptr, bool fence,
                                          remote_ptr<T> prealloc) {
  auto local = (prealloc == remote_nullptr) ? pool_->Allocate<T>() : prealloc;
  AddReadInternal(rptr, 0, sizeof(T), sizeof(T), fence, local);
  return local;
}

template <typename T>
remote_ptr<T> MemoryPool::DoorbellBatchBuilder::AddPartialRead(
    remote_ptr<T> rptr, size_t offset, size_t bytes, bool fence,
    remote_ptr<T> prealloc) {
  auto local = (prealloc == remote_nullptr) ? pool_->Allocate<T>() : prealloc;
  AddReadInternal(rptr, offset, bytes, bytes, fence, local);
  return local;
}

template <typename T>
void MemoryPool::DoorbellBatchBuilder::AddReadInternal(remote_ptr<T> rptr,
                                                       size_t offset,
                                                       size_t bytes,
                                                       size_t chunk, bool fence,
                                                       remote_ptr<T> prealloc) {
  const int num_chunks = bytes % chunk ? (bytes / chunk) + 1 : bytes / chunk;
  const size_t remainder = bytes % chunk;
  const bool is_multiple = remainder == 0;

  T *local = std::to_address(prealloc);
  for (int i = 0; i < num_chunks; ++i) {
    auto wr_sge = batch_->Add();
    auto wr = wr_sge.first;
    auto sge = wr_sge.second;

    auto chunk_offset = offset + i * chunk;
    sge->addr = reinterpret_cast<uint64_t>(local) + chunk_offset;
    if (is_multiple) {
      sge->length = chunk;
    } else {
      sge->length = (i == num_chunks - 1 ? remainder : chunk);
    }
    sge->lkey = batch_->conn_info().lkey;

    wr->opcode = IBV_WR_RDMA_READ;
    if (fence)
      wr->send_flags |= IBV_SEND_FENCE;
    wr->wr.rdma.remote_addr = rptr.address() + chunk_offset;
    wr->wr.rdma.rkey = batch_->conn_info().rkey;
  }
}

template <typename T>
void MemoryPool::DoorbellBatchBuilder::AddWrite(remote_ptr<T> rptr,
                                                const T &value, bool fence) {
  auto local = pool_->Allocate<T>();
  std::memcpy(std::to_address(local), &value, sizeof(value));
  AddWrite(rptr, local, fence);
}

template <typename T>
void MemoryPool::DoorbellBatchBuilder::AddWrite(remote_ptr<T> rptr,
                                                remote_ptr<T> prealloc,
                                                bool fence) {
  auto wr_sge = batch_->Add();
  ibv_send_wr *wr = wr_sge.first;
  ibv_sge *sge = wr_sge.second;

  sge->addr = (uint64_t)std::to_address(prealloc);
  sge->length = sizeof(T);
  sge->lkey = batch_->conn_info().lkey;

  wr->opcode = IBV_WR_RDMA_WRITE;
  if (fence)
    wr->send_flags |= IBV_SEND_FENCE;
  wr->wr.rdma.remote_addr = (uint64_t)std::to_address(rptr);
  wr->wr.rdma.rkey = batch_->conn_info().rkey;
}

template <typename T>
void MemoryPool::DoorbellBatchBuilder::AddWriteBytes(remote_ptr<T> rptr,
                                                     remote_ptr<T> prealloc,
                                                     int bytes, bool fence) {
  auto wr_sge = batch_->Add();
  ibv_send_wr *wr = wr_sge.first;
  ibv_sge *sge = wr_sge.second;

  sge->addr = (uint64_t)std::to_address(prealloc);
  sge->length = bytes;
  sge->lkey = batch_->conn_info().lkey;

  wr->opcode = IBV_WR_RDMA_WRITE;
  if (fence)
    wr->send_flags |= IBV_SEND_FENCE;
  wr->wr.rdma.remote_addr = (uint64_t)std::to_address(rptr);
  wr->wr.rdma.rkey = batch_->conn_info().rkey;
}

inline std::unique_ptr<MemoryPool::DoorbellBatch>
MemoryPool::DoorbellBatchBuilder::Build() {
  const int size = batch_->size();
  const int capcity = batch_->capacity();
  ROME_ASSERT(size > 0, "Cannot build an empty batch.");
  ROME_ASSERT(size == capcity, "Batch must be full");
  for (int i = 0; i < size; ++i) {
    batch_->wrs_[i].wr_id = batch_->wrs_[i].wr.rdma.remote_addr;
  }
  return std::move(batch_);
}

MemoryPool::MemoryPool(
    const Peer &self,
    std::unique_ptr<ConnectionManager<channel_type>> connection_manager)
    : self_(self), connection_manager_(std::move(connection_manager)),
      rdma_per_read_("rdma_per_read", "ops", 10000) {}

sss::Status MemoryPool::Init(uint32_t capacity,
                             const std::vector<Peer> &peers) {
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
    conn_info_.emplace(
        p.id, conn_info_t{conn.val.value(), got.val.value().rkey(), mr_->lkey});
  }

  std::thread t = std::thread([this] {
    WorkerThread();
  }); // TODO: Can I lower/raise the priority of this thread?
  t.detach();
  return {sss::Ok, {}};
}

void MemoryPool::KillWorkerThread() {
  this->run_worker = false;
  // Notify all mailboxes
  for (int i = 0; i < THREAD_MAX; i++) {
    std::unique_lock lck(this->mutex_vars[i]);
    this->mailboxes[i] = true;
    this->cond_vars[i].notify_one();
  }
}

void MemoryPool::WorkerThread() {
  while (this->run_worker) {
    for (auto it : this->conn_info_) {
      // TODO: Load balance the connections we check. Threads should have a way
      // to let us know what is worth checking Also might no be an issue?
      // Polling isn't expensive

      // Poll from conn
      conn_info_t info = it.second;
      ibv_wc wc;
      int poll = ibv_poll_cq(info.conn->id()->send_cq, 1, &wc);
      if (poll == 0)
        continue;

      // We polled something :)
      ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {}",
                  (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)));
      // notify wc.wr_id;
      std::unique_lock lck(this->mutex_vars[wc.wr_id]);
      this->mailboxes[wc.wr_id] = true;
      this->cond_vars[wc.wr_id].notify_one();
    }
  }
}

void MemoryPool::RegisterThread() {
  control_lock_.lock();
  std::thread::id mid = std::this_thread::get_id();
  if (this->thread_ids.find(mid) != this->thread_ids.end()) {
    ROME_FATAL("Cannot register the same thread twice");
    return;
  }
  if (this->id_gen == THREAD_MAX) {
    ROME_FATAL("Increase THREAD_MAX of memory pool");
    return;
  }
  this->thread_ids.insert(std::make_pair(mid, this->id_gen));
  this->id_gen++;
  control_lock_.unlock();
}

template <typename T> remote_ptr<T> MemoryPool::Allocate(size_t size) {
  // ROME_INFO("Allocating {} bytes ({} {} times)", sizeof(T)*size, sizeof(T),
  // size);
  auto ret = remote_ptr<T>(
      self_.id, rdma_allocator<T>(rdma_memory_.get()).allocate(size));
  return ret;
}

template <typename T>
void MemoryPool::Deallocate(remote_ptr<T> p, size_t size) {
  // ROME_INFO("Deallocating {} bytes ({} {} times)", sizeof(T)*size, sizeof(T),
  // size); else ROME_INFO("Deallocating {} bytes", sizeof(T));
  ROME_ASSERT(p.id() == self_.id,
              "Alloc/dealloc on remote node not implemented...");
  rdma_allocator<T>(rdma_memory_.get()).deallocate(std::to_address(p), size);
}

inline void MemoryPool::Execute(DoorbellBatch *batch) {
  ibv_send_wr *bad;
  auto *conn = batch->conn_info().conn;
  RDMA_CM_ASSERT(ibv_post_send, conn->id()->qp, batch->wrs_, &bad);

  int poll;
  ibv_wc wc;
  while ((poll = ibv_poll_cq(conn->id()->send_cq, 1, &wc)) == 0) {
    if (batch->is_mortal() && *kill)
      return;
    cpu_relax();
  }
  ROME_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
              "ibv_poll_cq(): {} (dest={})",
              (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
              remote_ptr<uint8_t>(wc.wr_id));
}

template <typename T>
remote_ptr<T> MemoryPool::ExtendedRead(remote_ptr<T> ptr, int size,
                                       remote_ptr<T> prealloc,
                                       std::atomic<bool> *kill) {
  if (prealloc == remote_nullptr)
    prealloc = Allocate<T>(size);
  ReadInternal(
      ptr, 0, sizeof(T) * size, sizeof(T) * size, prealloc,
      kill); // TODO: What happens if I decrease chunk size (* size to sizeT)
  return prealloc;
}

template <typename T>
remote_ptr<T> MemoryPool::Read(remote_ptr<T> ptr, remote_ptr<T> prealloc,
                               std::atomic<bool> *kill) {
  if (prealloc == remote_nullptr)
    prealloc = Allocate<T>();
  ReadInternal(ptr, 0, sizeof(T), sizeof(T), prealloc, kill);
  return prealloc;
}

template <typename T>
remote_ptr<T> MemoryPool::PartialRead(remote_ptr<T> ptr, size_t offset,
                                      size_t bytes, remote_ptr<T> prealloc) {
  if (prealloc == remote_nullptr)
    prealloc = Allocate<T>();
  ReadInternal(ptr, offset, bytes, sizeof(T), prealloc);
  return prealloc;
}

template <typename T>
void MemoryPool::ReadInternal(remote_ptr<T> ptr, size_t offset, size_t bytes,
                              size_t chunk_size, remote_ptr<T> prealloc,
                              std::atomic<bool> *kill) {
  // TODO: Has a kill that doesn't do anything
  const int num_chunks =
      bytes % chunk_size ? (bytes / chunk_size) + 1 : bytes / chunk_size;
  const size_t remainder = bytes % chunk_size;
  const bool is_multiple = remainder == 0;

  auto info = conn_info_.at(ptr.id());

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
  RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, wrs, &bad);

  std::unique_lock<std::mutex> lck(this->mutex_vars[index_as_id]);
  while (!this->mailboxes[index_as_id]) {
    this->cond_vars[index_as_id].wait(lck);
  }
  this->mailboxes[index_as_id] = false;
  rdma_per_read_lock_.lock();
  rdma_per_read_ << num_chunks;
  rdma_per_read_lock_.unlock();
}

template <typename T>
void MemoryPool::Write(remote_ptr<T> ptr, const T &val,
                       remote_ptr<T> prealloc) {
  ROME_DEBUG("Write: {:x} @ {}", (uint64_t)val, ptr);
  auto info = conn_info_.at(ptr.id());

  uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

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

  ROME_ASSERT((uint64_t)local != ptr.address() || ptr.id() != self_.id, "WTF");
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
  RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);

  std::unique_lock<std::mutex> lck(this->mutex_vars[index_as_id]);
  while (!this->mailboxes[index_as_id]) {
    this->cond_vars[index_as_id].wait(lck);
  }
  this->mailboxes[index_as_id] = false;

  if (prealloc == remote_nullptr) {
    auto alloc = rdma_allocator<T>(rdma_memory_.get());
    alloc.deallocate(local);
  }
}

template <typename T>
T MemoryPool::AtomicSwap(remote_ptr<T> ptr, uint64_t swap, uint64_t hint) {
  static_assert(sizeof(T) == 8);
  auto info = conn_info_.at(ptr.id());

  uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

  ibv_sge sge{};
  auto alloc = rdma_allocator<uint64_t>(rdma_memory_.get());
  volatile uint64_t *prev_ = alloc.allocate();
  sge.addr = reinterpret_cast<uint64_t>(prev_);
  sge.length = sizeof(uint64_t);
  sge.lkey = mr_->lkey;

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
    RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);
    std::unique_lock<std::mutex> lck(this->mutex_vars[index_as_id]);
    while (!this->mailboxes[index_as_id]) {
      this->cond_vars[index_as_id].wait(lck);
    }
    this->mailboxes[index_as_id] = false;
    ROME_DEBUG("Swap: expected={:x}, swap={:x}, prev={:x} (id={})",
               send_wr_.wr.atomic.compare_add, (uint64_t)swap, *prev_,
               self_.id);
    if (*prev_ == send_wr_.wr.atomic.compare_add)
      break;
    send_wr_.wr.atomic.compare_add = *prev_;
  };
  T ret = T(*prev_);
  alloc.deallocate((uint64_t *)prev_, 8);
  return ret;
}

template <typename T>
T MemoryPool::CompareAndSwap(remote_ptr<T> ptr, uint64_t expected,
                             uint64_t swap) {
  static_assert(sizeof(T) == 8);
  auto info = conn_info_.at(ptr.id());

  uint64_t index_as_id = this->thread_ids.at(std::this_thread::get_id());

  ibv_sge sge{};
  auto alloc = rdma_allocator<uint64_t>(rdma_memory_.get());
  volatile uint64_t *prev_ = alloc.allocate();
  sge.addr = reinterpret_cast<uint64_t>(prev_);
  sge.length = sizeof(uint64_t);
  sge.lkey = mr_->lkey;

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
  RDMA_CM_ASSERT(ibv_post_send, info.conn->id()->qp, &send_wr_, &bad);

  std::unique_lock<std::mutex> lck(this->mutex_vars[index_as_id]);
  while (!this->mailboxes[index_as_id]) {
    this->cond_vars[index_as_id].wait(lck);
  }
  this->mailboxes[index_as_id] = false;

  ROME_DEBUG("CompareAndSwap: expected={:x}, swap={:x}, actual={:x}  (id={})",
             expected, swap, *prev_, static_cast<uint64_t>(self_.id));
  T ret = T(*prev_);
  alloc.deallocate((uint64_t *)prev_, 8);
  return ret;
}

} // namespace rome::rdma