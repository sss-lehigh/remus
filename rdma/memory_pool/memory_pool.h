#pragma once

#include <array>
#include <infiniband/verbs.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <semaphore>

#include "../../metrics/summary.h"
#include "../../util/thread_util.h"
#include "../../vendor/sss/status.h"
#include "../channel/twosided_messenger.h"
#include "../connection_manager/connection.h"
#include "../connection_manager/connection_manager.h"
#include "../rmalloc/rmalloc.h"
#include "protos/rdma.pb.h"
#include "remote_ptr.h"

#define THREAD_MAX 10
// todo: I can't define a vector of semaphores? This makes it hard to reserve the space necessary for the semaphores...
// https://stackoverflow.com/questions/73449114/how-to-declare-and-initialize-a-vector-of-semaphores-in-c

namespace rome::rdma {

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
                                            kMemoryPoolMessageSize>,
                      EmptyRdmaAccessor>
      channel_type;
  typedef ConnectionManager<channel_type> cm_type;
  typedef cm_type::conn_type conn_type;

  /**
   * @brief Construct a peer object to represent a node/machine. 
   * Used in creating connections via the memorypool or connection managers
   */
  struct Peer {
    uint16_t id;
    std::string address;
    uint16_t port;

    /**
     * @brief Construct a new Peer object with id and port of 0, empty address
     */
    Peer() : Peer(0, "", 0) {}
    /**
     * @brief Construct a new Peer object
     * 
     * @param id the peer's unique ID
     * @param address the address (hostname or IP) of the peer's machine
     * @param port the port to connect to
     */
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

  /// @brief Allocate resources (semaphore) for the thread before it can run operations
  void RegisterThread();

private:
  template <typename T>
  inline void ReadInternal(remote_ptr<T> ptr, size_t offset, size_t bytes,
                           size_t chunk_size, remote_ptr<T> prealloc,
                           std::atomic<bool> *kill = nullptr);

  Peer self_;

  /// Used to protect the id generator, the thread_ids, and the reordering_semaphores
  std::mutex control_lock_;
  /// Used to protect the rdma_per_read_ summary statistics thing
  std::mutex rdma_per_read_lock_;

  /// A counter to increment
  uint64_t id_gen = 0;
  /// A mapping of thread id to an index into the reordering_semaphores array. Passed as the wr_id in work requests.
  std::unordered_map<std::thread::id, uint64_t> thread_ids;
  /// a vector of semaphores, one for each thread that can send an operation. Threads will use this to recover from polling another thread's wr_id
  std::array<std::binary_semaphore, THREAD_MAX> reordering_semaphores = {
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0),
      std::binary_semaphore(0)
  };
  
  std::unique_ptr<ConnectionManager<channel_type>> connection_manager_;
  std::unique_ptr<rdma_memory_resource> rdma_memory_;
  ibv_mr *mr_;

  std::unordered_map<uint16_t, conn_info_t> conn_info_;

  rome::metrics::Summary<size_t> rdma_per_read_;
};

} // namespace rome::rdma

#include "memory_pool_impl.h"