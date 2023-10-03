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

using ::rome::rdma::ConnectionManager;
using ::rome::rdma::MemoryPool;
using ::rome::rdma::remote_nullptr;
using ::rome::rdma::remote_ptr;
using ::rome::rdma::RemoteObjectProto;

template <class K, class V, int ELIST_SIZE, int PLIST_SIZE> class RdmaIHT {
private:
  MemoryPool::Peer self_;

  // "Poor-mans" enum to represent the state of a node. P-lists cannot be locked
  // E_LOCKED = 1, E_UNLOCKED = 2, P_UNLOCKED = 3
  const uint64_t E_LOCKED = 1, E_UNLOCKED = 2, P_UNLOCKED = 3;

  // "Super class" for the elist and plist structs
  struct Base {};
  typedef uint64_t lock_type;
  typedef remote_ptr<Base> remote_baseptr;
  typedef remote_ptr<lock_type> remote_lock;

  // ElementList stores a bunch of K/V pairs. IHT employs a "seperate
  // chaining"-like approach. Rather than storing via a linked list (with easy
  // append), it uses a fixed size array
  struct alignas(64) EList : Base {
    struct pair_t {
      K key;
      V val;
    };

    size_t count = 0;         // The number of live elements in the Elist
    pair_t pairs[ELIST_SIZE]; // A list of pairs to store (stored as remote
                              // pointer to start of the contigous memory block)

    // Insert into elist a deconstructed pair
    void elist_insert(const K key, const V val) {
      pairs[count] = {key, val};
      count++;
    }

    // Insert into elist a pair
    void elist_insert(const pair_t pair) {
      pairs[count] = pair;
      count++;
    }

    EList() { ROME_DEBUG("Running EList Constructor!"); }
  };

  // A pointer lock pair
  struct plist_pair_t {
    remote_baseptr base; // Pointer to base, the super class of Elist or Plist
    remote_lock lock;    // A lock to represent if the base is open or not
    // TODO: Maybe I can manipulate the lock without needing a pointer?
  };

  // PointerList stores EList pointers and assorted locks
  struct alignas(64) PList : Base {
    plist_pair_t buckets[PLIST_SIZE]; // Pointer lock pairs
  };

  typedef remote_ptr<PList> remote_plist;
  typedef remote_ptr<EList> remote_elist;

  /// @brief Initialize the plist with values.
  /// @param p the plist pointer to init
  /// @param depth the depth of p, needed for PLIST_SIZE == base_size * (2 **
  /// (depth - 1)) pow(2, depth)
  inline void InitPList(remote_plist p, int mult_modder) {
    for (size_t i = 0; i < PLIST_SIZE * mult_modder; i++) {
      p->buckets[i].lock = pool_->Allocate<lock_type>();
      *p->buckets[i].lock = E_UNLOCKED;
      p->buckets[i].base = remote_nullptr;
    }
  }

  remote_plist root;     // Start of plist
  std::hash<K> pre_hash; // Hash function from k -> size_t [this currently does
                         // nothing as the value of the int can just be returned
                         // :: though included for templating this class]

  /// Acquire a lock on the bucket. Will prevent others from modifying it
  bool acquire(remote_lock lock) {
    // Spin while trying to acquire the lock
    while (true) {
      lock_type v =
          pool_->CompareAndSwap<lock_type>(lock, E_UNLOCKED, E_LOCKED);

      // Permanent unlock
      if (v == P_UNLOCKED)
        return false;
      // If we can switch from unlock to lock status
      if (v == E_UNLOCKED)
        return true;
    }
  }

  /// @brief Unlock a lock ==> the reverse of acquire
  /// @param lock the lock to unlock
  /// @param unlock_status what should the end lock status be.
  inline void unlock(remote_lock lock, uint64_t unlock_status) {
    remote_lock temp = pool_->Allocate<lock_type>();
    pool_->Write<lock_type>(lock, unlock_status, temp);
    // Have to deallocate "8" of them to account for alignment
    pool_->Deallocate<lock_type>(temp, 8);
  }

  template <typename T> inline bool is_local(remote_ptr<T> ptr) {
    return ptr.id() == self_.id;
  }

  template <typename T> inline bool is_null(remote_ptr<T> ptr) {
    return ptr == remote_nullptr;
  }

  /// @brief Change the baseptr from a given bucket (could be remote as well)
  /// @param before_localized_curr the start of the bucket list (plist)
  /// @param bucket the bucket to write to
  /// @param baseptr the new pointer that bucket should point to
  inline void change_bucket_pointer(remote_plist before_localized_curr,
                                    uint64_t bucket, remote_baseptr baseptr) {
    uint64_t address_of_baseptr = before_localized_curr.address();
    address_of_baseptr += sizeof(plist_pair_t) * bucket;
    remote_ptr<remote_baseptr> magic_baseptr = remote_ptr<remote_baseptr>(
        before_localized_curr.id(), address_of_baseptr);
    if (!is_local(magic_baseptr)) {
      // Have to use a temp variable to account for alignment. Remote pointer is
      // 8 bytes!
      auto temp = pool_->Allocate<remote_baseptr>();
      pool_->Write<remote_baseptr>(magic_baseptr, baseptr, temp);
      pool_->Deallocate<remote_baseptr>(temp, 8);
    } else
      *magic_baseptr = baseptr;
  }

  // Hashing function to decide bucket size
  inline uint64_t level_hash(const K &key, size_t level, size_t count) {
    return (level ^ pre_hash(key)) %
           (count - 1); // we use count-1 because this prevents the collision
                        // errors associated with "mod 2A" given "mod A"
  }

  /// Rehash function
  /// @param parent The P-List whose bucket needs rehashing
  /// @param pcount The number of elements in `parent`
  /// @param pdepth The depth of `parent`
  /// @param pidx   The index in `parent` of the bucket to rehash
  remote_plist rehash(remote_plist parent, size_t pcount, size_t pdepth,
                      size_t pidx) {
    pcount = pcount * 2;
    int plist_size_factor =
        (pcount / PLIST_SIZE); // pow(2, pdepth); // how much bigger than
                               // original size we are

    // 2 ^ (depth) ==> in other words (depth:factor). 0:1, 1:2, 2:4, 3:8, 4:16,
    // 5:32.
    remote_plist new_p = pool_->Allocate<PList>(plist_size_factor);
    InitPList(new_p, plist_size_factor);

    // hash everything from the full elist into it
    remote_elist parent_bucket =
        static_cast<remote_elist>(parent->buckets[pidx].base);
    remote_elist source = is_local(parent_bucket)
                              ? parent_bucket
                              : pool_->Read<EList>(parent_bucket);
    for (size_t i = 0; i < source->count; i++) {
      uint64_t b = level_hash(source->pairs[i].key, pdepth + 1, pcount);
      if (is_null(new_p->buckets[b].base)) {
        remote_elist e = pool_->Allocate<EList>();
        new_p->buckets[b].base = static_cast<remote_baseptr>(e);
      }
      remote_elist dest = static_cast<remote_elist>(new_p->buckets[b].base);
      dest->elist_insert(source->pairs[i]);
    }
    // Deallocate the old elist
    pool_->Deallocate<EList>(source);
    return new_p;
  }

public:
  MemoryPool *pool_;

  using conn_type = MemoryPool::conn_type;

  RdmaIHT(MemoryPool::Peer self, MemoryPool *pool) : self_(self), pool_(pool) {
    if ((PLIST_SIZE * 8) % 64 != 0)
      ROME_INFO("Warning: Suboptimal PLIST_SIZE b/c PList needs to be aligned "
                "to 64 bytes");
    if (((ELIST_SIZE * 8) + 4) % 64 < 60)
      ROME_INFO("Warning: Suboptimal ELIST_SIZE b/c EList needs to be aligned "
                "to 64 bytes");
  };

  /// @brief Initialize the IHT by connecting to the peers and exchanging the
  /// PList pointer
  /// @param host the leader of the initialization
  /// @param peers all the nodes in the neighborhood
  /// @return status code for the function
  sss::Status Init(MemoryPool::Peer host,
                   const std::vector<MemoryPool::Peer> &peers) {
    bool is_host_ = self_.id == host.id;

    if (is_host_) {
      // Host machine, it is my responsibility to initiate configuration
      RemoteObjectProto proto;
      remote_plist iht_root = pool_->Allocate<PList>();
      // Init plist and set remote proto to communicate its value
      InitPList(iht_root, 1);
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
      auto got =
          conn_or.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      while (got.status.t == sss::Unavailable) {
        got = conn_or.val.value()->channel()->TryDeliver<RemoteObjectProto>();
      }
      RETURN_STATUSVAL_ON_ERROR(got);

      // From there, decode the data into a value
      remote_plist iht_root =
          decltype(iht_root)(host.id, got.val.value().raddr());
      this->root = iht_root;
    }

    return sss::Status::Ok();
  }

  /// @brief Gets a value at the key.
  /// @param key the key to search on
  /// @return if the key was found or not. The value at the key is stored in
  /// RdmaIHT::result
  HT_Res<V> contains(K key) {
    // start at root
    remote_plist curr = pool_->Read<PList>(root);
    remote_plist before_localized_curr = root;
    size_t depth = 1, count = PLIST_SIZE;
    bool oldBucketBase = true;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      if (!acquire(curr->buckets[bucket].lock)) {
        // Can't lock then we are at a sub-plist
        // Therefore we must re-fetch the PList to ensure freshness of our
        // pointers (1 << depth-1 to adjust size of read with customized
        // ExtendedRead)
        remote_plist curr_temp =
            pool_->ExtendedRead<PList>(before_localized_curr, 1 << (depth - 1));
        remote_plist bucket_base =
            static_cast<remote_plist>(curr_temp->buckets[bucket].base);
        remote_plist base_ptr =
            is_local(bucket_base) || is_null(bucket_base)
                ? bucket_base
                : pool_->ExtendedRead<PList>(bucket_base, 1 << depth);
        pool_->Deallocate<PList>(curr_temp, 1 << (depth - 1));

        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        oldBucketBase = !is_local(bucket_base); // setting the old bucket base

        before_localized_curr = bucket_base;
        curr = base_ptr;
        depth++;
        count *= 2;
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base =
          static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e = is_local(bucket_base) || is_null(bucket_base)
                           ? bucket_base
                           : pool_->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // empty elist
        unlock(curr->buckets[bucket].lock, E_UNLOCKED);
        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        return HT_Res<V>(FALSE_STATE, 0);
      }

      // Get elist and linear search
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the key
        if (e->pairs[i].key == key) {
          K result = e->pairs[i].val;
          unlock(curr->buckets[bucket].lock, E_UNLOCKED);
          if (!is_local(bucket_base))
            pool_->Deallocate<EList>(e);
          if (oldBucketBase)
            pool_->Deallocate<PList>(
                curr, 1 << (depth - 1)); // deallocate if curr was not ours
          return HT_Res<V>(TRUE_STATE, result);
        }
      }

      // Can't find, unlock and return false
      unlock(curr->buckets[bucket].lock, E_UNLOCKED);
      if (!is_local(bucket_base))
        pool_->Deallocate<EList>(e);
      if (oldBucketBase)
        pool_->Deallocate<PList>(
            curr, 1 << (depth - 1)); // deallocate if curr was not ours
      return HT_Res<V>(FALSE_STATE, 0);
    }
  }

  /// @brief Insert a key and value into the iht. Result will become the value
  /// at the key if already present.
  /// @param key the key to insert
  /// @param value the value to associate with the key
  /// @return if the insert was successful
  HT_Res<V> insert(K key, V value) {
    // start at root
    remote_plist curr = pool_->Read<PList>(root);
    remote_plist before_localized_curr = root;
    size_t depth = 1, count = PLIST_SIZE;
    bool oldBucketBase = true;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      if (!acquire(curr->buckets[bucket].lock)) {
        // Can't lock then we are at a sub-plist
        // Therefore we must re-fetch the PList to ensure freshness of our
        // pointers
        remote_plist curr_temp =
            pool_->ExtendedRead<PList>(before_localized_curr, 1 << (depth - 1));
        remote_plist bucket_base =
            static_cast<remote_plist>(curr_temp->buckets[bucket].base);
        remote_plist base_ptr =
            is_local(bucket_base) || is_null(bucket_base)
                ? bucket_base
                : pool_->ExtendedRead<PList>(bucket_base, 1 << depth);
        pool_->Deallocate<PList>(curr_temp, 1 << (depth - 1));

        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        oldBucketBase = !is_local(bucket_base); // setting the old bucket base

        before_localized_curr = bucket_base;
        curr = base_ptr;
        depth++;
        count *= 2;
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base =
          static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e = is_local(bucket_base) || is_null(bucket_base)
                           ? bucket_base
                           : pool_->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // empty elist
        remote_elist e_new = pool_->Allocate<EList>();
        e_new->elist_insert(key, value);
        remote_baseptr e_base = static_cast<remote_baseptr>(e_new);
        // modify the bucket's pointer
        change_bucket_pointer(before_localized_curr, bucket, e_base);
        unlock(curr->buckets[bucket].lock, E_UNLOCKED);
        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        // successful insert
        return HT_Res<V>(TRUE_STATE, 0);
      }

      // We have recursed to an non-empty elist
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the key
        if (e->pairs[i].key == key) {
          K result = e->pairs[i].val;
          // Contains the key => unlock and return false
          unlock(curr->buckets[bucket].lock, E_UNLOCKED);
          if (bucket_base.id() != self_.id)
            pool_->Deallocate<EList>(e);
          if (oldBucketBase)
            pool_->Deallocate<PList>(
                curr, 1 << (depth - 1)); // deallocate if curr was not ours
          return HT_Res<V>(FALSE_STATE, result);
        }
      }

      // Check for enough insertion room
      if (e->count < ELIST_SIZE) {
        // insert, unlock, return
        e->elist_insert(key, value);
        // If we are modifying a local copy, we need to write to the remote at
        // the end
        if (bucket_base.id() != self_.id)
          pool_->Write<EList>(static_cast<remote_elist>(bucket_base), *e);
        // unlock and return true
        unlock(curr->buckets[bucket].lock, E_UNLOCKED);
        if (bucket_base.id() != self_.id)
          pool_->Deallocate<EList>(e);
        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        return HT_Res<V>(TRUE_STATE, 0);
      }

      // Need more room so rehash into plist and perma-unlock
      remote_plist p = rehash(curr, count, depth, bucket);
      // modify the bucket's pointer
      change_bucket_pointer(before_localized_curr, bucket,
                            static_cast<remote_baseptr>(p));
      // keep local curr updated with remote curr
      curr->buckets[bucket].base = static_cast<remote_baseptr>(p);
      // unlock bucket
      unlock(curr->buckets[bucket].lock, P_UNLOCKED);
      if (!is_local(bucket_base))
        pool_->Deallocate<EList>(e);
      // repeat from top in a way to progress past the plist we just inserted,
      // without deallocating it.
      oldBucketBase = false;
    }
  }

  /// @brief Will remove a value at the key. Will stored the previous value in
  /// result.
  /// @param key the key to remove at
  /// @return if the remove was successful
  HT_Res<V> remove(K key) {
    // start at root
    remote_plist curr = pool_->Read<PList>(root);
    remote_plist before_localized_curr = root;
    size_t depth = 1, count = PLIST_SIZE;
    bool oldBucketBase = true;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      if (!acquire(curr->buckets[bucket].lock)) {
        // Can't lock then we are at a sub-plist
        // Therefore we must re-fetch the PList to ensure freshness of our
        // pointers (1 << depth-1 to adjust size of read with customized
        // ExtendedRead)
        remote_plist curr_temp =
            pool_->ExtendedRead<PList>(before_localized_curr, 1 << (depth - 1));
        remote_plist bucket_base =
            static_cast<remote_plist>(curr_temp->buckets[bucket].base);
        remote_plist base_ptr =
            is_local(bucket_base) || is_null(bucket_base)
                ? bucket_base
                : pool_->ExtendedRead<PList>(bucket_base, 1 << depth);
        pool_->Deallocate<PList>(curr_temp, 1 << (depth - 1));

        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        oldBucketBase = !is_local(bucket_base); // setting the old bucket base

        before_localized_curr = bucket_base;
        curr = base_ptr;
        depth++;
        count *= 2;
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base =
          static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e = is_local(bucket_base) || is_null(bucket_base)
                           ? bucket_base
                           : pool_->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // empty elist, can just unlock and return false
        unlock(curr->buckets[bucket].lock, E_UNLOCKED);
        if (oldBucketBase)
          pool_->Deallocate<PList>(
              curr, 1 << (depth - 1)); // deallocate if curr was not ours
        return HT_Res<V>(FALSE_STATE, 0);
      }

      // Get elist and linear search
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the value
        if (e->pairs[i].key == key) {
          K result = e->pairs[i].val; // saving the previous value at key
          if (e->count > 1) {
            // Edge swap if not count=0|1
            e->pairs[i] = e->pairs[e->count - 1];
          }
          e->count -= 1;
          // If we are modifying the local copy, we need to write to the remote
          // at the end...
          if (!is_local(bucket_base))
            pool_->Write<EList>(static_cast<remote_elist>(bucket_base), *e);
          // Unlock and return
          unlock(curr->buckets[bucket].lock, E_UNLOCKED);
          if (!is_local(bucket_base))
            pool_->Deallocate<EList>(e);
          if (oldBucketBase)
            pool_->Deallocate<PList>(
                curr, 1 << (depth - 1)); // deallocate if curr was not ours
          return HT_Res<V>(TRUE_STATE, result);
        }
      }

      // Can't find, unlock and return false
      unlock(curr->buckets[bucket].lock, E_UNLOCKED);
      if (!is_local(bucket_base))
        pool_->Deallocate<EList>(e);
      if (oldBucketBase)
        pool_->Deallocate<PList>(
            curr, 1 << (depth - 1)); // deallocate if curr was not ours
      return HT_Res<V>(FALSE_STATE, 0);
    }
  }

  /// Function signature added to match map interface. No intermediary cleanup
  /// necessary so unusued
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
