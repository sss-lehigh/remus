#pragma once

#include "listener.h"

#include <memory>
#include <unordered_map>

#include "remus/logging/logging.h"

// TODO:  Should these headers be in an "internal" subfolder?
#include "connection.h"
#include "connection_map.h"
#include "connector.h"
#include "memory_pool.h"
#include "peer.h"

#include "rdma_ptr.h"

namespace remus::rdma {

/// Draft for per-thread rdma capability
/// Currently only supporting the one-sided API
/// We can create multiple of these to distribute to the threads
class rdma_capability_thread {
  friend class rdma_capability;
  const Peer& self;
  internal::MemoryPool pool;  // A map of all the RDMA heaps

  rdma_capability_thread(const Peer &self, internal::rdma_memory_resource* rdma_memory_) : self(self), pool(self, rdma_memory_) {
    
  }
public:
  /// Allocate some memory from the local RDMA heap
  template <typename T> rdma_ptr<T> Allocate(size_t size = 1) { return pool.Allocate<T>(size); }

  /// Return some memory to the local RDMA heap
  template <typename T> void Deallocate(rdma_ptr<T> p, size_t size = 1) { pool.Deallocate(p, size); }

  /// Write to an RDMA heap
  template <typename T> void Write(rdma_ptr<T> ptr, const T &val, rdma_ptr<T> prealloc = nullptr) {
    pool.Write(ptr, val, prealloc);
  }

  /// Perform a CAS on the RDMA heap
  template <typename T> T CompareAndSwap(rdma_ptr<T> ptr, uint64_t expected, uint64_t swap) {
    return pool.CompareAndSwap<T>(ptr, expected, swap);
  }

  /// Perform a CAS on the RDMA heap that loops until it succeeds
  template <typename T> T AtomicSwap(rdma_ptr<T> ptr, uint64_t swap, uint64_t hint = 0) {
    return pool.AtomicSwap<T>(ptr, swap, hint);
  }

  /// Read a variable-sized object from the RDMA heap
  template <typename T> rdma_ptr<T> ExtendedRead(rdma_ptr<T> ptr, int size, rdma_ptr<T> prealloc = nullptr) {
    return pool.ExtendedRead(ptr, size, prealloc);
  }

  /// Read a fixed-sized object from the RDMA heap
  template <typename T> rdma_ptr<T> Read(rdma_ptr<T> ptr, rdma_ptr<T> prealloc = nullptr) {
    return pool.Read(ptr, prealloc);
  }

  /// Determine if a rdma_ptr is local to the machine
  /// Utilizies the peer inputted to rdma_capability to check the id of the ptr
  template <class T> bool is_local(rdma_ptr<T> ptr) { return ptr.id() == self.id; }
};

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
  using capability_ptr = std::unique_ptr<rdma_capability_thread>;
  using memory_ptr = std::unique_ptr<internal::rdma_memory_resource>;
  Peer self_;                 // Identifying information for this node
  std::vector<capability_ptr> pools;  // A map of all the RDMA heaps
  cm_ptr connection_manager_; // The connections associated with the heaps
  listen_ptr listener_;       // The listening socket/thread
  connector_ptr connector_;   // A utility for connecting to other nodes
  memory_ptr rdma_memory_; // memory region
  
  // Registering threads with capability
  std::mutex control_lock_;
  int counter = 0;
  std::unordered_map<thread::id, int> capability_mapping;

  internal::rdma_memory_resource *init_memory(uint32_t capacity, ibv_pd *pd) {
    // Create a memory region (mr) in the current protection domain (pd)
    rdma_memory_ = std::make_unique<internal::rdma_memory_resource>(capacity + sizeof(uint64_t), pd);
    return rdma_memory_.get();
  }

public:
  // TODO: This needs to be redesigned.  It's odd to construct the
  // connection_manager and listener_, but not the connector, and it's odd to
  // need to explicitly call init_pool.  This should be able to be merged.
  explicit rdma_capability(const Peer &self, int conns_per_node_conn = 1)
    : self_(self), connection_manager_(std::make_unique<internal::ConnectionMap>(self_.id)),
      listener_(std::make_unique<internal::Listener>(
        self_.id, [&](uint32_t id, internal::Connection *conn) { connection_manager_->put_connection(id, conn); })) {
          pools = std::vector<capability_ptr>(conns_per_node_conn);
          counter = 0;
        }

  ~rdma_capability() {}

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
    // Create a memory region of the requested size
    auto mem = init_memory(capacity, listener_->pd());
    for(int pool_index = 0; pool_index < pools.size(); pool_index++){
      pools[pool_index] = std::unique_ptr<rdma_capability_thread>(new rdma_capability_thread(self_, mem));
    }
    REMUS_DEBUG("Inited memory");

    connector_ = std::make_unique<internal::Connector>(
      self_.id, listener_->pd(), listener_->address(),
      [&](uint32_t id, internal::Connection *conn) { connection_manager_->put_connection(id, conn); });

    for (int pool_index = 0; pool_index < pools.size(); pool_index++){
      // Connect on the loopback first, as a sort of canary for testing IB
      // [mfs] I have confirmed that multiple calls to ConnectLoopback do work
      connector_->ConnectLoopback(self_.id, self_.address, self_.port);

      // Iterate and connect to peers only with larger IDs. Avoids collisions between nodes
      for (const auto &p : peers) {
        if (p.id != self_.id && p.id > self_.id) {
          REMUS_DEBUG("Connecting to remote peer {}:{} (id = {}) from {}", p.address, p.port, p.id, self_.id);
          // Connect to the remote nodes
          // [mfs] I have confirmed that multiple calls to ConnectLoopback do work
          connector_->ConnectRemote(p.id, p.address, p.port);
        } else {
           REMUS_DEBUG("Other peer will connect to me {}:{} (id = {}) to {}", p.address, p.port, p.id, self_.id);
        }
      }
    }

    // Spin until we have the right number of peers
    auto last_size = connection_manager_->size();
    int goal_size = peers.size() * pools.size();
    REMUS_DEBUG("Connection manager has {} connections looking for {}", connection_manager_->size(), goal_size);
    while (true) {
      if (connection_manager_->size() == goal_size)
        break;
      if (connection_manager_->size() > goal_size)
        REMUS_FATAL("Unexpected number of connections in the map");
      if (connection_manager_->size() != last_size) {
        REMUS_DEBUG("Connection manager has {} connections looking for {}", connection_manager_->size(), goal_size);
        last_size = connection_manager_->size();
      }
    }

    // Send the memory region to all peers
    //
    //  [mfs] This is probably not safe... we're blasting out a whole bunch of
    //        Sends, then doing a whole bunch of Recvs.  The first problem is
    //        that we want to have more than one qp between each pair of nodes,
    //        so what happens if the sender and receiver disagree on which qp to
    //        use?  The second problem is that the buffers for send/recv are
    //        bounded, so what happens if we have so many nodes that we overflow
    //        our buffer?  Will sends get rejected, or will they block?  Both
    //        aren't handled by this code.

    // Updated to provide an order to sending vs recieving info about the memory region from all peers
    
    RemoteObjectProto rm_proto;
    rm_proto.set_rkey(mem->mr()->rkey);
    rm_proto.set_raddr(reinterpret_cast<uint64_t>(mem->mr()->addr));
    for (int pool_index = 0; pool_index < pools.size(); pool_index++){
      // Get a connection group by index in the connection map
      auto connection_group = connection_manager_->SliceConnections(pool_index);
      for (const auto &p : peers) {
        REMUS_ASSERT(connection_group.find(p.id) != connection_group.end(), "Enough of one peer, {}, doesn't exist in connection map", p.id);
        auto conn = connection_group[p.id];
        if (p.id >= self_.id) {
          auto status = conn->Send(rm_proto);
          REMUS_ASSERT(status.t == remus::util::Ok, status.message.value());
          auto got = conn->template Deliver<RemoteObjectProto>();
          REMUS_ASSERT(got.status.t == remus::util::Ok, got.status.message.value());
          // [mfs] I don't understand why we use mr_->lkey?
          pools[pool_index]->pool.receive_conn(p.id, conn, got.val.value().rkey(), mem->mr()->lkey);
        } else {
          auto got = conn->template Deliver<RemoteObjectProto>();
          REMUS_ASSERT(got.status.t == remus::util::Ok, got.status.message.value());
          auto status = conn->Send(rm_proto);
          REMUS_ASSERT(status.t == remus::util::Ok, status.message.value());
          // [mfs] I don't understand why we use mr_->lkey?
          pools[pool_index]->pool.receive_conn(p.id, conn, got.val.value().rkey(), mem->mr()->lkey);
        }
        REMUS_DEBUG("Adding a connection with id {} to pool {}", p.id, pool_index);
      }
    }
    REMUS_INFO("RDMA init_pool Completed for Peer {}", self_.id);

    REMUS_TRACE("Stopping listening thread...");
    listener_->StopListeningThread();
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

  /// Do a blocking Recv of a non-proto from another machine
  remus::util::StatusVal<std::vector<uint8_t>> RecvBytes(const Peer &from) {
    // Listen for a connection
    auto conn = connection_manager_->GetConnection(from.id);
    auto got = conn->DeliverBytes();
    return got;
  }

  /// Connect a thread to this pool.  Registered threads can ack each other's
  /// one-sided operations.  Threads *must* be registered!
  ///
  /// TODO: Registration should define the affinity between a thread and a
  ///       Connection to each node, since there can be >1 Connection from this
  ///       node to each other node.  Some thread safety issues are likely to
  ///       arise, too.
  /// TODO: Threads are assigned a capability in an unfair manner. 
  ///       Each capabilty has its own set of QPs to every node and if the number 
  ///       of pools doesn't divide evenly by the number of threads, then a set
  ///       of QPs will be underutilized.
  rdma_capability_thread* RegisterThread() {
    control_lock_.lock();
    std::thread::id mid = std::this_thread::get_id();
    int index;
    if (this->capability_mapping.find(mid) != this->capability_mapping.end()) {
      index = capability_mapping[mid];
      control_lock_.unlock();
    } else {
      index = counter++;
      capability_mapping[mid] = index;
      pools[index % pools.size()]->pool.RegisterThread();
      control_lock_.unlock();
    }
    return pools[index % pools.size()].get();
  }

  /// Determine if a rdma_ptr is local to the machine
  /// Utilizies the peer inputted to rdma_capability to check the id of the ptr
  template <class T> bool is_local(rdma_ptr<T> ptr) { return ptr.id() == self_.id; }
};
} // namespace remus::rdma
