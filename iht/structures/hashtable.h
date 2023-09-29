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
#include "../common.h"
#include "linked_set.h"

using ::rome::rdma::ConnectionManager;
using ::rome::rdma::MemoryPool;
using ::rome::rdma::remote_nullptr;
using ::rome::rdma::remote_ptr;
using ::rome::rdma::RemoteObjectProto;

template <class K, class V, int INITIAL_SIZE> class Hashtable {
private:
  MemoryPool::Peer self_;

  // Remote bucket type definition
  typedef LinkedSet<K, V> LinkedKV;
  typedef remote_ptr<LinkedKV> remote_bucket;

  // An "array" object to be used with RDMA verbs
  struct alignas(64) HashArray {
    long count;
    remote_bucket bucket_start;
  };

  typedef remote_ptr<HashArray> remote_array;
  remote_array root;     // Start of hashtable
  std::hash<K> pre_hash; // Hash function from k -> size_t

  template <typename T> inline bool is_local(remote_ptr<T> ptr) {
    return ptr.id() == self_.id;
  }

  template <typename T> inline bool is_null(remote_ptr<T> ptr) {
    return ptr == remote_nullptr;
  }

  // Hashing function to decide bucket size
  inline uint64_t keyhash(const K &key, size_t count) {
    return pre_hash(key) % count;
  }

  /// @brief Get remote_bucket an index amount of LinkedSets away (essentially
  /// array accesss)
  /// @param start the start of the array
  /// @param index the index in the array
  /// @return the remote_ptr to the bucket in the array
  remote_bucket indexAt(remote_bucket start, int index) {
    uint64_t new_address = start.address();
    new_address += sizeof(LinkedKV) * index;
    return remote_bucket(start.id(), new_address);
  }

  /// @brief Initialize the root of the hashtable
  /// @param arr the pointer to initialize
  void InitArray(remote_array root) {
    // Allocate and init the hashtable
    HashArray hashArrayTemp;
    hashArrayTemp.count = INITIAL_SIZE;
    hashArrayTemp.bucket_start = pool_->Allocate<LinkedKV>(INITIAL_SIZE);
    for (int i = 0; i < INITIAL_SIZE; i++) {
      remote_bucket bucket = indexAt(hashArrayTemp.bucket_start, i);
      *std::to_address(bucket) = LinkedKV(pool_);
    }
    *std::to_address(root) = hashArrayTemp;
  }

public:
  MemoryPool *pool_;

  using conn_type = MemoryPool::conn_type;

  Hashtable(MemoryPool::Peer self, MemoryPool *pool)
      : self_(self), pool_(pool){};

  /// @brief Initialize the IHT by connecting to the peers and exchanging the
  /// PList pointer
  /// @param host the leader of the initialization
  /// @param peers all the nodes in the neighborhood
  /// @return status code for the function
  absl::Status Init(MemoryPool::Peer host,
                    const std::vector<MemoryPool::Peer> &peers) {
    bool is_host_ = self_.id == host.id;

    if (is_host_) {
      // Host machine, it is my responsibility to initiate configuration
      RemoteObjectProto proto;
      remote_array hashtable_root = pool_->Allocate<HashArray>();
      // Init hashtable and set remote proto to communicate its value
      InitArray(hashtable_root);
      this->root = hashtable_root;
      proto.set_raddr(hashtable_root.address());

      // Iterate through peers
      for (auto p = peers.begin(); p != peers.end(); p++) {
        // Ignore sending pointer to myself
        if (p->id == self_.id)
          continue;

        // Form a connection with the machine
        auto conn_or = pool_->connection_manager()->GetConnection(p->id);
        ROME_CHECK_OK(ROME_RETURN(conn_or.status()), conn_or);

        // Send the proto over
        absl::Status status = conn_or.value()->channel()->Send(proto);
        ROME_CHECK_OK(ROME_RETURN(status), status);
      }
    } else {
      // Listen for a connection
      auto conn_or = pool_->connection_manager()->GetConnection(host.id);
      ROME_CHECK_OK(ROME_RETURN(conn_or.status()), conn_or);

      // Try to get the data from the machine, repeatedly trying until
      // successful
      auto got = conn_or.value()->channel()->TryDeliver<RemoteObjectProto>();
      while (got.status().code() == absl::StatusCode::kUnavailable) {
        got = conn_or.value()->channel()->TryDeliver<RemoteObjectProto>();
      }
      ROME_CHECK_OK(ROME_RETURN(got.status()), got);

      // From there, decode the data into a value
      remote_array hashtable_root =
          decltype(hashtable_root)(host.id, got->raddr());
      this->root = hashtable_root;
    }
    return absl::OkStatus();
  }

  /// @brief Gets a value at the key.
  /// @param key the key to search on
  /// @return if the key was found or not. The value at the key is stored in
  /// Hashtable::result
  HT_Res<V> contains(K key) {
  start:
    remote_array ds = pool_->Read<HashArray>(root);
    HashArray hasharray = *std::to_address(ds);
    int bucket_index = keyhash(key, hasharray.count);
    remote_bucket bucket = indexAt(hasharray.bucket_start, bucket_index);
    remote_bucket bucket_red = pool_->Read<LinkedKV>(bucket);
    LinkedSet<K, V> set = *std::to_address(bucket_red);
    HT_Res<V> state = set.contains(pool_, key);
    if (state.status == REHASH_DELETED) {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      goto start;
    } else {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      return HT_Res<V>(state.status == TRUE_STATE ? TRUE_STATE : FALSE_STATE,
                       state.result);
    }
  }

  /// @brief Insert a key and value into the iht. Result will become the value
  /// at the key if already present.
  /// @param key the key to insert
  /// @param value the value to associate with the key
  /// @return if the insert was successful
  HT_Res<V> insert(K key, V value) {
  start:
    remote_array ds = pool_->Read<HashArray>(root);
    HashArray hasharray = *std::to_address(ds);
    int bucket_index = keyhash(key, hasharray.count);
    remote_bucket bucket = indexAt(hasharray.bucket_start, bucket_index);
    remote_bucket bucket_red = pool_->Read<LinkedKV>(bucket);
    LinkedSet<K, V> set = *std::to_address(bucket_red);
    HT_Res<V> state = set.insert(pool_, key, value);
    if (state.status == REHASH_DELETED) {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      goto start;
    } else {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      return HT_Res<V>(state.status == TRUE_STATE ? TRUE_STATE : FALSE_STATE,
                       state.result);
    }
  }

  /// @brief Will remove a value at the key. Will stored the previous value in
  /// result.
  /// @param key the key to remove at
  /// @return if the remove was successful
  HT_Res<V> remove(K key) {
  start:
    remote_array ds = pool_->Read<HashArray>(root);
    HashArray hasharray = *std::to_address(ds);
    int bucket_index = keyhash(key, hasharray.count);
    remote_bucket bucket = indexAt(hasharray.bucket_start, bucket_index);
    remote_bucket bucket_red = pool_->Read<LinkedKV>(bucket);
    LinkedSet<K, V> set = *std::to_address(bucket_red);
    HT_Res<V> state = set.remove(pool_, key);
    if (state.status == REHASH_DELETED) {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      goto start;
    } else {
      pool_->Deallocate(ds);
      pool_->Deallocate(bucket_red);
      return HT_Res<V>(state.status == TRUE_STATE ? TRUE_STATE : FALSE_STATE,
                       state.result);
    }
  }

  /// Print data
  void print() {
    // Read root
    remote_array ds = pool_->Read<HashArray>(root);
    HashArray hashtable = *std::to_address(ds);
    remote_bucket buckets =
        pool_->ExtendedRead<LinkedKV>(hashtable.bucket_start, hashtable.count);

    // Sum up count of elements
    for (int i = 0; i < hashtable.count; i++) {
      remote_bucket bucket = indexAt(buckets, i);
      LinkedKV set = *std::to_address(bucket);
      ROME_INFO("{} bucket", i);
      set.foreach (pool_, [&](K k, V v) { ROME_INFO("\t{}", k); });
    }

    // Deallocate the data
    pool_->Deallocate<HashArray>(ds);
    pool_->Deallocate<LinkedKV>(buckets, hashtable.count);
  }

  /// Rehash function
  void try_rehash() {
    if (is_local<HashArray>(root)) {
      // Read root
      remote_array ds = pool_->Read<HashArray>(root);
      HashArray hashtable = *std::to_address(ds);
      remote_bucket buckets = pool_->ExtendedRead<LinkedKV>(
          hashtable.bucket_start, hashtable.count);

      // Sum up count of elements
      long total_elements = 0;
      for (int i = 0; i < hashtable.count; i++) {
        remote_bucket bucket = indexAt(buckets, i);
        LinkedKV set = *std::to_address(bucket);
        int ll_length = set.list_length(pool_);
        total_elements += ll_length;
      }

      // Check if total elements exceeds load factor
      if (total_elements < hashtable.count * 10) {
        pool_->Deallocate<HashArray>(ds);
        pool_->Deallocate<LinkedKV>(buckets, hashtable.count);
        return;
      }

      // Acquire lock on each LinkedSet
      for (int i = 0; i < hashtable.count; i++) {
        remote_bucket bucket = indexAt(buckets, i);
        LinkedKV set = *std::to_address(bucket);
        set.freeze(pool_);
      }

      // Allocate the hashtable
      HashArray hashArrayTemp;
      hashArrayTemp.count = hashtable.count * 2;
      hashArrayTemp.bucket_start =
          pool_->Allocate<LinkedKV>(hashtable.count * 2);
      for (int i = 0; i < hashtable.count * 2; i++) {
        remote_bucket bucket = indexAt(hashArrayTemp.bucket_start, i);
        *std::to_address(bucket) = LinkedKV(pool_);
      }
      *std::to_address(root) = hashArrayTemp;

      // Re-add the data from the existing hashtable
      for (int i = 0; i < hashtable.count; i++) {
        remote_bucket bucket = indexAt(buckets, i);
        LinkedKV set = *std::to_address(bucket);
        set.foreach (pool_, [&](K k, V v) {
          int bucket_index = keyhash(k, hashtable.count * 2);
          remote_bucket bucket =
              indexAt(hashArrayTemp.bucket_start, bucket_index);
          LinkedKV ll = *std::to_address(bucket);
          ll.insert(pool_, k, v);
        });
      }

      // Mark all linked lists as deleted
      for (int i = 0; i < hashtable.count; i++) {
        remote_bucket bucket = indexAt(buckets, i);
        LinkedKV set = *std::to_address(bucket);
        set.melt(pool_);
      }

      // Deallocate the data
      pool_->Deallocate<HashArray>(ds);
      pool_->Deallocate<LinkedKV>(buckets, hashtable.count);
    }
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
