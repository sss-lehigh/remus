#pragma once

#include "listener.h"

#include <memory>

#include "remus/logging/logging.h"

// TODO:  Should these headers be in an "internal" subfolder?
#include "connection.h"
#include "connection_map.h"
#include "connector.h"
#include "memory_pool.h"
#include "peer.h"

#include "rdma_ptr.h"

namespace remus::rdma {

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
/// TODO: It would also be nice if the rdma_ptr infrastructure was fully
///       encapsulated by this, rather than being an independent feature.
///
/// TODO: The memory access functions currently have a lot of allocation
///       overhead, because they need a pinned region for staging data to/from
///       RDMA.  When we have a proper thread object, we can probably fix this
///       issue via per-thread buffers.
class rdma_capability {
  using cm_ptr = std::unique_ptr<internal::ConnectionMap>;
  using listen_ptr = std::unique_ptr<internal::Listener>;
  using connector_ptr = std::unique_ptr<internal::Connector>;
  Peer self_;                 // Identifying information for this node
  internal::MemoryPool pool;  // A map of all the RDMA heaps
  cm_ptr connection_manager_; // The connections associated with the heaps
  listen_ptr listener_;       // The listening socket/thread
  connector_ptr connector_;   // A utility for connecting to other nodes

public:
  // TODO: This needs to be redesigned.  It's odd to construct the
  // connection_manager and listener_, but not the connector, and it's odd to
  // need to explicitly call init_pool.  This should be able to be merged.
  explicit rdma_capability(const Peer &self)
      : self_(self), pool(self_),
        connection_manager_(
            std::make_unique<internal::ConnectionMap>(self_.id)),
        listener_(std::make_unique<internal::Listener>(
            self_.id, [&](uint32_t id, internal::Connection *conn) {
              connection_manager_->put_connection(id, conn);
            })) {}

  ~rdma_capability() {
    // TODO: Make sure we aren't using the word "broker" anymore?
    REMUS_TRACE("Stopping listening thread...");
    listener_->StopListeningThread();
  }

  /// Create a block of distributed memory and connect to all the other nodes in
  /// the system, so that they can have access to that memory region.
  ///
  /// This method does three things.
  /// - It creates an all-all communication network with every peer
  /// - It creates a memory region with `capacity` as its size.
  /// - It exchanges the region with all peers
  ///
  /// TODO: Should there be some kind of "shutdown()" method?
  ///
  /// TODO: Why can't we merge this into the constructor?
  ///
  /// TODO: The IHT code uses this in an awkward way, due to the blocking
  ///       behavior of the methods this calls.  We should think about how to do
  ///       a better job.
  ///
  /// TODO: I feel like this should have a "connections-per-machine-pair"
  ///       argument.
  ///
  /// TODO: I also feel like we shouldn't be leaving it up to the caller to make
  ///       `peers`.  Why can't we pass in a list of node ids, and a port, and
  ///       be done with it?  And why can't every listener use the same port,
  ///       now that there's only one per node.
  void init_pool(uint32_t capacity, std::vector<Peer> &peers) {
    using namespace std::string_literals;

    // Start a listening thread, so we can get nodes connected to this node
    listener_->StartListeningThread(self_.address, self_.port);

    connector_ = std::make_unique<internal::Connector>(
        self_.id, listener_->pd(), listener_->address(),
        [&](uint32_t id, internal::Connection *conn) {
          connection_manager_->put_connection(id, conn);
        });

    // Connect on the loopback first, as a sort of canary for testing IB
    //
    // [mfs] I have confirmed that multiple calls to ConnectLoopback do work
    connector_->ConnectLoopback(self_.id, self_.address, self_.port);

    // Go through the list of peers and connect to each of them.  Note that we
    // only connect with "bigger" peers, in terms of Id, so that we don't have
    // collisions between nodes trying to connect with each other.
    for (const auto &p : peers) {
      if (p.id != self_.id && p.id > self_.id) {
        REMUS_DEBUG("Connecting to remote peer"s + p.address + ":" +
                   std::to_string(p.port) + " (id = " + std::to_string(p.id) +
                   ") from " + std::to_string(self_.id));
        // Connect to the remote nodes
        //
        // [mfs] I have confirmed that multiple calls to ConnectLoopback do work
        connector_->ConnectRemote(p.id, p.address, p.port);
      }
    }
    REMUS_DEBUG("Finished connecting to remotes");

    // Spin until we have the right number of peers
    while(true){
      if (connection_manager_->size() == peers.size()) break;
      if (connection_manager_->size() > peers.size()) REMUS_FATAL("Unexpected number of connections in the map");
    }

    // Create a memory region of the requested size
    auto mem = pool.init_memory(capacity, listener_->pd());

    // Send the memory region to all peers
    //
    // TODO:  This is probably not safe... we're blasting out a whole bunch of
    //        Sends, then doing a whole bunch of Recvs.  The first problem is
    //        that we want to have more than one qp between each pair of nodes,
    //        so what happens if the sender and receiver disagree on which qp to
    //        use?  The second problem is that the buffers for send/recv are
    //        bounded, so what happens if we have so many nodes that we overflow
    //        our buffer?  Will sends get rejected, or will they block?  Both
    //        aren't handled by this code.
    RemoteObjectProto rm_proto;
    rm_proto.set_rkey(mem->mr()->rkey);
    rm_proto.set_raddr(reinterpret_cast<uint64_t>(mem->mr()->addr));
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      auto status = conn->Send(rm_proto);
      if (status.t != remus::util::Ok) {
        REMUS_FATAL(status.message.value());
      }
    }

    // Receive all peers' memory regions and pass them to the memory_pool, so it
    // can manage them
    for (const auto &p : peers) {
      auto conn = connection_manager_->GetConnection(p.id);
      auto got = conn->template Deliver<RemoteObjectProto>();
      if (got.status.t != remus::util::Ok) {
        REMUS_FATAL(got.status.message.value());
      }
      // [mfs] I don't understand why we use mr_->lkey?
      pool.receive_conn(p.id, conn, got.val.value().rkey(), mem->mr()->lkey);
    }

    // TODO: This message isn't informative enough
    REMUS_INFO("Created memory pool");
  }

  /// Allocate some memory from the local RDMA heap
  template <typename T> rdma_ptr<T> Allocate(size_t size = 1) {
    return pool.Allocate<T>(size);
  }

  /// Return some memory to the local RDMA heap
  template <typename T> void Deallocate(rdma_ptr<T> p, size_t size = 1) {
    pool.Deallocate(p, size);
  }

  /// Write to an RDMA heap
  template <typename T>
  void Write(rdma_ptr<T> ptr, const T &val,
             rdma_ptr<T> prealloc = nullptr) {
    pool.Write(ptr, val, prealloc);
  }

  /// Perform a CAS on the RDMA heap
  template <typename T>
  T CompareAndSwap(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    return pool.CompareAndSwap<T>(ptr, expected, swap);
  }

  /// Perform a CAS on the RDMA heap that loops until it succeeds
  template <typename T>
  T AtomicSwap(rdma_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    return pool.AtomicSwap<T>(ptr, swap, hint);
  }

  /// Read a variable-sized object from the RDMA heap
  template <typename T>
  rdma_ptr<T> ExtendedRead(rdma_ptr<T> ptr, int size,
                             rdma_ptr<T> prealloc = nullptr) {
    return pool.ExtendedRead(ptr, size, prealloc);
  }

  /// Read a fixed-sized object from the RDMA heap
  template <typename T>
  rdma_ptr<T> Read(rdma_ptr<T> ptr,
                     rdma_ptr<T> prealloc = nullptr) {
    return pool.Read(ptr, prealloc);
  }

  /// Do a blocking Send() to another machine
  template <class T> remus::util::Status Send(const Peer &to, T &proto) {
    auto conn = connection_manager_->GetConnection(to.id);
    return conn->Send(proto);
  }

  /// Do a blocking Recv() from another machine
  template <class T> remus::util::StatusVal<T> Recv(const Peer &from) {
    // Listen for a connection
    auto conn = connection_manager_->GetConnection(from.id);
    auto got = conn->Deliver<T>();
    return got;
  }

  /// Connect a thread to this pool.  Registered threads can ack each other's
  /// one-sided operations.  Threads *must* be registered!
  ///
  /// TODO: Registration should define the affinity between a thread and a
  ///       Connection to each node, since there can be >1 Connection from this
  ///       node to each other node.  Some thread safety issues are likely to
  ///       arise, too.
  void RegisterThread() { pool.RegisterThread(); }

  /// Determine if a rdma_ptr is local to the machine
  /// Utilizies the peer inputted to rdma_capability to check the id of the ptr
  template <class T>
  bool is_local(rdma_ptr<T> ptr){
    return ptr.id() == self_.id;
  }
};
} // namespace remus::rdma
