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

  /// [mfs] Why do we need to return connections?  Seems broken!
  ///
  /// [mfs] It looks like GetConnection is being used as a pathway into doing
  ///       RPCs for synchronous communication.  We should replace it with
  ///       explicit "blocking send to" and "blocking receive from" methods.
  ///
  /// [mfs] Note that right now, rome only allows synchronous RPC of protobufs.
  ///       Long-term, that's kind of silly, but for now, we can live with it.
  sss::StatusVal<internal::Connection *> GetConnection(const Peer &host) {
    return pool.connection_manager()->GetConnection(host.id);
  }
};
} // namespace rome::rdma