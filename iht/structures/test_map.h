#pragma once

#include <atomic>
#include <cstdint>
#include <infiniband/verbs.h>

#include "../../logging/logging.h"
#include "../../rdma/channel/sync_accessor.h"
#include "../../rdma/connection_manager/connection.h"
#include "../../rdma/connection_manager/connection_manager.h"
#include "../../rdma/memory_pool/memory_pool.h"
#include "../../rdma/rdma_memory.h"
#include "../../vendor/sss/status.h"
#include "../common.h"

#define LEN 8

using ::rome::rdma::ConnectionManager;
using ::rome::rdma::MemoryPool;
using ::rome::rdma::remote_nullptr;
using ::rome::rdma::remote_ptr;
using ::rome::rdma::RemoteObjectProto;

// [mfs] Need documentation
template <class K, class V> class TestMap {
private:
  // [mfs] I'm concerned about keeping the Peer in the map like this...
  MemoryPool::Peer self_;

  // [mfs] Is this in use?
  struct alignas(64) List {
    uint64_t vals[LEN];
  };

  inline void InitList(remote_ptr<List> p) {
    for (size_t i = 0; i < LEN; i++) {
      p->vals[i] = i;
    }
  }

  remote_ptr<List> root; // Start of list

  template <typename T> inline bool is_local(remote_ptr<T> ptr) {
    return ptr.id() == self_.id;
  }

  template <typename T> inline bool is_null(remote_ptr<T> ptr) {
    return ptr == remote_nullptr;
  }

public:
  // [mfs] Why is this part of the data structure?
  MemoryPool *pool_;

  using conn_type = MemoryPool::conn_type;

  TestMap(MemoryPool::Peer self, MemoryPool *pool) : self_(self), pool_(pool){};

  /// @brief Initialize the IHT by connecting to the peers and exchanging the
  /// PList pointer
  /// @param host the leader of the initialization
  /// @param peers all the nodes in the neighborhood
  /// @return status code for the function
  //
  // [mfs] I feel like this should be happening in main(), and distributing the
  //       IHT, not like this.  Among other things, I think doing things this
  //       way makes the IHT less usable in real code (e.g., does it have to be
  //       a singleton?)
  sss::Status Init(MemoryPool::Peer host,
                   const std::vector<MemoryPool::Peer> &peers) {
    bool is_host_ = self_.id == host.id;

    if (is_host_) {
      // Host machine, it is my responsibility to initiate configuration
      //
      // [mfs]  It seems like overkill to use a ProtoBuf just to send a 64-bit
      //        address.
      RemoteObjectProto proto;
      // [mfs] Is this doing "list" instead of IHT?  Should the data type be a
      // template parameter?
      remote_ptr<List> iht_root = pool_->Allocate<List>();
      // Init plist and set remote proto to communicate its value
      InitList(iht_root);
      this->root = iht_root;
      proto.set_raddr(iht_root.address());

      // Iterate through peers
      for (auto p = peers.begin(); p != peers.end(); p++) {
        // Ignore sending pointer to myself
        if (p->id == self_.id)
          continue;

        // Form a connection with the machine
        auto conn_or = pool_->connection_manager()->GetConnection(p->id);
        RETURN_STATUSVAL_ON_ERROR(conn_or);

        // Send the proto over
        auto status = conn_or.val.value()->channel()->Send(proto);
        RETURN_STATUS_ON_ERROR(status);
      }
    } else {
      // Listen for a connection
      auto conn_or = pool_->connection_manager()->GetConnection(host.id);
      RETURN_STATUSVAL_ON_ERROR(conn_or);

      // Try to get the data from the machine, repeatedly trying until
      // successful
      //
      // [mfs]  Since the connection is shared, I need to get a better
      //        understanding on how this data gets into a buffer that is
      //        allocated and owned by the current thread.
      auto got =
          conn_or.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      while (got.status.t == sss::Unavailable) {
        got = conn_or.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      }
      RETURN_STATUSVAL_ON_ERROR(got);

      // From there, decode the data into a value
      remote_ptr<List> iht_root =
          decltype(iht_root)(host.id, got.val.value().raddr());
      this->root = iht_root;
    }

    return {sss::Ok, {}};
  }

  /// @brief Gets a value at the key.
  /// @param key the key to search on
  /// @return if the key was found or not. The value at the key is stored in
  /// RdmaIHT::result
  HT_Res<V> contains(K key) {
    // [mfs] I don't understand the point of prealloc here?
    //
    // [mfs] It looks like this code is incomplete?
    remote_ptr<List> prealloc = pool_->Allocate<List>();
    List temp = *std::to_address(prealloc);
    temp.vals[0] = 100;
    *std::to_address(prealloc) = temp;
    remote_ptr<List> list = pool_->Read<List>(this->root, prealloc);
    if (list.address() != prealloc.address()) {
      ROME_INFO("Prealloc not working as expected");
    }
    List l = *std::to_address(list);
    for (int i = 0; i < LEN; i++) {
      if (l.vals[i] != i) {
        ROME_INFO("Illegal inequality {} {}", l.vals[i], i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    pool_->Deallocate<List>(list);

    /*
    List old = *std::to_address(this->root);
    uint64_t test = old.vals[0];
    old.vals[0]++;
    pool_->Write<List>(this->root, old);
    List l = *std::to_address(this->root);
    if(l.vals[0] == test){
        ROME_INFO("Illegal equality");
    }
    */

    return HT_Res<V>(TRUE_STATE, key);
  }

  /// @brief Insert a key and value into the iht. Result will become the value
  /// at the key if already present.
  /// @param key the key to insert
  /// @param value the value to associate with the key
  /// @return if the insert was successful
  HT_Res<V> insert(K key, V value) {
    // [mfs] Is this not actually implemented yet?
    return HT_Res<V>(TRUE_STATE, 0);
  }

  /// @brief Will remove a value at the key. Will stored the previous value in
  /// result.
  /// @param key the key to remove at
  /// @return if the remove was successful
  HT_Res<V> remove(K key) {
    // [mfs] Is this not actually implemented yet?
    return HT_Res<V>(FALSE_STATE, 0);
  }

  /// Function signature added to match map interface. No intermediary cleanup
  /// necessary so unused
  //
  // [mfs] I don't understand why any map interface would need this?
  void try_rehash() {
    // Unused function b/c no cleanup necessary
  }

  /// @brief Populate only works when we have numerical keys. Will add data
  /// @param count the number of values to insert. Recommended in total to do
  /// key_range / 2
  /// @param key_lb the lower bound for the key range
  /// @param key_ub the upper bound for the key range
  /// @param value the value to associate with each key. Currently, we have
  /// asserts for result to be equal to the key. Best to set value equal to key!
  void populate(int op_count, K key_lb, K key_ub, std::function<K(V)> value) {
    // Populate only works when we have numerical keys
    K key_range = key_ub - key_lb;

    // Create a random operation generator that is
    // - evenly distributed among the key range
    std::uniform_real_distribution<double> dist =
        std::uniform_real_distribution<double>(0.0, 1.0);
    std::default_random_engine gen((unsigned)std::time(NULL));
    for (int c = 0; c < op_count; c++) {
      int k = dist(gen) * key_range + key_lb;
      insert(k, value(k));
      // Wait some time before doing next insert...
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
  }
};
