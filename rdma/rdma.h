#pragma once

#include <memory>

#include "../logging/logging.h"

// TODO: Should these three headers be in an "internal" subfolder?
#include "connection_manager.h"
#include "memory_pool.h"
#include "peer.h"

#include "remote_ptr.h"

namespace rome::rdma {

/// rdma_node is a per-node object that, once configured, provides all of the
/// underlying features needed by threads wishing to interact with RDMA memory.
/// Broadly, this means that it has a set of connections to other machines and a
/// set of mappings from local addresses to remote addresses.
///
/// TODO: This object is not quite right yet (and it isn't renamed yet):
///       - RegisterThread should return a thread capability object
///       - All of the memory operations should be members of that thread
///         capability object.
///
/// TODO: A major issue right now is that a process makes *multiple* rdma_nodes,
///       because each only has one QP between nodes.  This means that there is
///       lots of redundancy in the rdma_nodes' maps, and it also means there's
///       another layer sitting on top of this, when the goal was for this to be
///       the top-level object.
///
/// TODO: It would also be nice if the remote_ptr infrastructure was fully
///       encapsulated by this, rather than being an independent feature.
///
/// TODO: The memory access functions currently have a lot of allocation
///       overhead, because they need a pinned region for staging data to/from
///       RDMA.  When we have a proper thread object, we can probably fix this
///       issue via per-thread buffers.
class rdma_capability {
  using cm_ptr = std::unique_ptr<internal::ConnectionManager>;
  Peer self_;                 // Identifying information for this node
  internal::MemoryPool pool;  // A map of all the RDMA heaps
  cm_ptr connection_manager_; // The connections associated with the heaps

public:
  explicit rdma_capability(const Peer &self)
      : self_(self), pool(self_),
        connection_manager_(
            std::make_unique<internal::ConnectionManager>(self_.id)) {}

  /// Create a block of distributed memory and connect to all the other nodes in
  /// the system, so that they can have access to that memory region.
  ///
  /// This method does three things.
  /// - It creates a memory region with `capacity` as its size.
  /// - It creates an all-all communication network with every peer
  /// - It exchanges the region with all peers
  ///
  /// TODO: Should there be some kind of "shutdown()" method?
  ///
  /// TODO: Why can't we merge this into the constructor?
  ///
  /// TODO: The IHT code uses this in an awkward way, due to the blocking
  ///       behavior of the methods this calls.  We should think about how to do
  ///       a better job.
  void init_pool(uint32_t capacity, std::vector<Peer> &peers) {
    // Launch a connection manager, to enable other machines to connect to the
    // memory pool we're going to create.
    //
    // TODO: Make Start() fail-stop
    auto status = connection_manager_->Start(self_.address, self_.port);
    if (status.t != sss::Ok) {
      ROME_FATAL(status.message);
      std::terminate();
    }

    // Go through the list of peers and connect to each of them
    for (const auto &p : peers) {
      // TODO: Why not have a blocking connect call?
      auto connected = connection_manager_->Connect(p.id, p.address, p.port);
      while (connected.status.t == sss::Unavailable) {
        connected = connection_manager_->Connect(p.id, p.address, p.port);
      }
      if (connected.status.t != sss::Ok) {
        ROME_FATAL(connected.status.message);
        std::terminate();
      }
    }

    // Create a memory region of the requested size
    auto mem = pool.init_memory(capacity, connection_manager_->pd());

    // Send the memory region to all peers
    RemoteObjectProto rm_proto;
    rm_proto.set_rkey(mem->mr()->rkey);
    rm_proto.set_raddr(reinterpret_cast<uint64_t>(mem->mr()->addr));
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      if (conn.status.t != sss::Ok) {
        ROME_FATAL(conn.status.message);
        std::terminate();
      }
      auto status = conn.val.value()->Send(rm_proto);
      if (status.t != sss::Ok) {
        ROME_FATAL(status.message);
        std::terminate();
      }
    }

    // Receive all peers' memory regions and pass them to the memory_pool, so it
    // can manage them
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      if (conn.status.t != sss::Ok) {
        ROME_FATAL(conn.status.message);
        std::terminate();
      }
      auto got = conn.val.value()->template Deliver<RemoteObjectProto>();
      if (got.status.t != sss::Ok) {
        ROME_FATAL(got.status.message);
        std::terminate();
      }
      // [mfs] I don't understand why we use mr_->lkey?
      pool.receive_conn(p.id, conn.val.value(), got.val.value().rkey(),
                        mem->mr()->lkey);
    }

    // TODO: This message isn't informative enough
    ROME_INFO("Created memory pool");
  }

  /// Allocate some memory from the local RDMA heap
  template <typename T> remote_ptr<T> Allocate(size_t size = 1) {
    return pool.Allocate<T>(size);
  }

  /// Return some memory to the local RDMA heap
  template <typename T> void Deallocate(remote_ptr<T> p, size_t size = 1) {
    pool.Deallocate(p, size);
  }

  /// Write to an RDMA heap
  template <typename T>
  void Write(remote_ptr<T> ptr, const T &val,
             remote_ptr<T> prealloc = remote_nullptr) {
    pool.Write(ptr, val, prealloc);
  }

  /// Perform a CAS on the RDMA heap
  template <typename T>
  T CompareAndSwap(remote_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    return pool.CompareAndSwap<T>(ptr, expected, swap);
  }

  /// Read a variable-sized object from the RDMA heap
  template <typename T>
  remote_ptr<T> ExtendedRead(remote_ptr<T> ptr, int size,
                             remote_ptr<T> prealloc = remote_nullptr) {
    return pool.ExtendedRead(ptr, size, prealloc);
  }

  /// Read a fixed-sized object from the RDMA heap
  template <typename T>
  remote_ptr<T> Read(remote_ptr<T> ptr,
                     remote_ptr<T> prealloc = remote_nullptr) {
    return pool.Read(ptr, prealloc);
  }

  /// Do a blocking Send() to another machine
  template <class T> sss::Status Send(const Peer &to, T &proto) {
    // Form a connection with the machine
    auto conn_or = connection_manager_->GetConnection(to.id);
    RETURN_STATUSVAL_ON_ERROR(conn_or);

    // Send the proto over
    return conn_or.val.value()->Send(proto);
  }

  /// Do a blocking Recv() from another machine
  template <class T> sss::StatusVal<T> Recv(const Peer &from) {
    // Listen for a connection
    auto conn_or = connection_manager_->GetConnection(from.id);
    if (conn_or.status.t != sss::Ok)
      return {conn_or.status, {}};
    auto got = conn_or.val.value()->Deliver<T>();
    return got;
  }

  /// Connect a thread to this pool.  Registered threads can ack each other's
  /// one-sided operations.  Threads *must* be registered!
  ///
  /// TODO: It may be possible to do away with registration once
  ///       pool.RegisterThread is rewritten.
  void RegisterThread() { pool.RegisterThread(); }
};
} // namespace rome::rdma