#pragma once

#include <memory>

#include "../logging/logging.h"

// TODO: move to "internal" subfolder
#include "connection_manager.h"
#include "memory_pool.h"

#include "peer.h"
#include "remote_ptr.h"

namespace rome::rdma {

class rdma_capability {
  internal::MemoryPool<internal::ConnectionManager> pool;

public:
  explicit rdma_capability(const Peer &self)
      : //    cm(my_id),
        pool(self, std::unique_ptr<internal::ConnectionManager>(
                       new internal::ConnectionManager((self.id)))) {}

  // TODO: Why can't we merge this into the constructor?
  //
  // [mfs]  Let's be more ambitious... now that the surface is smaller, can we
  //        completely decouple the broker, the pool, and the connection
  //        manager?  We could move logic from pool.Init into this method...
  void init_pool(uint32_t block_size, std::vector<Peer> &peers) {
    auto status_pool = pool.Init(block_size, peers);
    OK_OR_FAIL(status_pool);
    ROME_INFO("Created memory pool");
  }

  /// Allocate some memory from the local RDMA heap
  template <typename T> remote_ptr<T> Allocate(size_t size = 1) {
    return pool.Allocate<T>(size);
  }

  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1) {
    pool.Deallocate(p, size);
  }

  template <typename T>
  void Write(remote_ptr<T> ptr, const T &val,
             remote_ptr<T> prealloc = remote_nullptr) {
    pool.Write(ptr, val, prealloc);
  }

  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    return pool.CompareAndSwap<T>(ptr, expected, swap);
  }

  template <typename T>
  remote_ptr<T> ExtendedRead(remote_ptr<T> ptr, int size,
                             remote_ptr<T> prealloc = remote_nullptr) {
    return pool.ExtendedRead(ptr, size, prealloc);
  }

  template <typename T>
  remote_ptr<T> Read(remote_ptr<T> ptr,
                     remote_ptr<T> prealloc = remote_nullptr) {
    return pool.Read(ptr, prealloc);
  }

  template <class T> sss::Status Send(const Peer &to, T &proto) {
    // Form a connection with the machine
    auto conn_or = pool.connection_manager()->GetConnection(to.id);
    RETURN_STATUSVAL_ON_ERROR(conn_or);

    // Send the proto over
    return conn_or.val.value()->channel()->Send(proto);
  }

  template <class T> sss::StatusVal<T> Recv(const Peer &from) {
    // Listen for a connection
    auto conn_or = pool.connection_manager()->GetConnection(from.id);
    if (conn_or.status.t != sss::Ok)
      return {conn_or.status, {}};

    // Try to get the data from the machine, repeatedly trying until
    // successful
    //
    // [mfs]  Since the connection is shared, I need to get a better
    //        understanding on how this data gets into a buffer that is
    //        allocated and owned by the current thread.
    auto got = conn_or.val.value()->channel()->Deliver<T>();
    return got;
  }

  /// [el] Register a thread means allocating resources to that specific thread
  /// that allows them to synchronize with each other
  void RegisterThread() { pool.RegisterThread(); }
};
} // namespace rome::rdma