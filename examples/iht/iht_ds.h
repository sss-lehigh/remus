#pragma once

#include <random>

#include "common.h"
#include <remus/rdma/rdma.h>

using namespace remus::rdma;

template <class K, class V, int ELIST_SIZE, int PLIST_SIZE> class RdmaIHT {
private:
  Peer self_;

  /// State of a bucket
  /// E_LOCKED - The bucket is in use by a thread. The bucket points to an EList
  /// E_UNLOCKED - The bucket is free for manipulation. The bucket baseptr
  /// points to an EList P_UNLOCKED - The bucket will always be free for
  /// manipulation because it points to a PList. It is "calcified"
  const uint64_t E_LOCKED = 1, E_UNLOCKED = 2, P_UNLOCKED = 3;

  // "Super class" for the elist and plist structs
  struct Base {};
  typedef uint64_t lock_type;
  typedef rdma_ptr<Base> remote_baseptr;
  typedef rdma_ptr<lock_type> remote_lock;

  struct pair_t {
    K key;
    V val;
  };

  // ElementList stores a bunch of K/V pairs. IHT employs a "separate
  // chaining"-like approach. Rather than storing via a linked list (with easy
  // append), it uses a fixed size array
  struct alignas(64) EList : Base {
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

    EList() {
      this->count = 0; // ensure count is 0
    }
  };

  // [mfs]  This probably needs to be aligned
  // [esl]  I'm confused, this is only used as part of the plist, which is
  // aligned...
  /// A PList bucket. It is a pointer-lock pair
  /// @param base is a pointer to the next elist/plist.
  /// @param lock is the data that represents the state of the bucket. See
  /// lock_type.
  struct plist_pair_t {
    remote_baseptr base; // Pointer to base, the super class of Elist or Plist
    lock_type lock;      // A lock to represent if the base is open or not
  };

  // PointerList stores EList pointers and assorted locks
  struct alignas(64) PList : Base {
    plist_pair_t buckets[PLIST_SIZE]; // Pointer lock pairs
  };

  typedef rdma_ptr<PList> remote_plist;
  typedef rdma_ptr<EList> remote_elist;

  // Get the address of the lock at bucket (index)
  remote_lock get_lock(remote_plist arr_start, int index) {
    uint64_t new_addy = arr_start.address();
    new_addy += (sizeof(plist_pair_t) * index) + 8;
    return remote_lock(arr_start.id(), new_addy);
  }

  // Get the address of the baseptr at bucket (index)
  rdma_ptr<remote_baseptr> get_baseptr(remote_plist arr_start, int index) {
    uint64_t new_addy = arr_start.address();
    new_addy += sizeof(plist_pair_t) * index;
    return rdma_ptr<remote_baseptr>(arr_start.id(), new_addy);
  }

  /// @brief Initialize the plist with values.
  /// @param p the plist pointer to init
  /// @param depth the depth of p, needed for PLIST_SIZE == base_size * (2 **
  /// (depth - 1)) pow(2, depth)
  inline void InitPList(remote_plist p, int mult_modder) {
    assert(sizeof(plist_pair_t) == 16); // Assert I did my math right...
    for (size_t i = 0; i < PLIST_SIZE * mult_modder; i++) {
      p->buckets[i].lock = E_UNLOCKED;
      p->buckets[i].base = nullptr;
    }
  }

  remote_plist root; // Start of plist

  /// @brief Prehash a value and then apply a finalizer (mix13)
  /// @param key the value to start with
  /// @param randomizer a value to xor with to permutate key (use depth)
  /// @return the hashed value
  /// @note Mix13 maintains divisibility so we still have to subtract 1 from the
  /// bucket count
  size_t pre_hash(K key, size_t randomizer) {
    std::hash<K> to_num;
    size_t hashed = to_num(key) ^ randomizer;
    // mix13
    hashed ^= (hashed >> 33);
    hashed *= 0xff51afd7ed558ccd;
    hashed ^= (hashed >> 33);
    hashed *= 0xc4ceb9fe1a85ec53;
    hashed ^= (hashed >> 33);
    return hashed;
  }

  /// Acquire a lock on the bucket. Will prevent others from modifying it
  bool acquire(std::shared_ptr<rdma_capability> pool, remote_lock lock) {
    // Spin while trying to acquire the lock
    while (true) {
      // Can this be a CAS on an address within a PList?
      lock_type v = pool->CompareAndSwap<lock_type>(lock, E_UNLOCKED, E_LOCKED);

      // Permanent unlock
      if (v == P_UNLOCKED) {
        return false;
      }
      // If we can switch from unlock to lock status
      if (v == E_UNLOCKED) {
        return true;
      }
    }
  }

  /// @brief Unlock a lock ==> the reverse of acquire
  /// @param lock the lock to unlock
  /// @param unlock_status what should the end lock status be.
  inline void unlock(std::shared_ptr<rdma_capability> pool, remote_lock lock, uint64_t unlock_status) {
    remote_lock temp = pool->Allocate<lock_type>();
    pool->Write<lock_type>(lock, unlock_status, temp);
    // Have to deallocate "8" of them to account for alignment
    // [esl] This "deallocate 8" is a hack to get around a remus memory leak.
    // (must fix remus to fix this)
    pool->Deallocate<lock_type>(temp, 8);
  }

  template <typename T> inline bool is_null(rdma_ptr<T> ptr) { return ptr == nullptr; }

  /// @brief Change the baseptr for a given bucket to point to a different EList
  /// or a different PList
  /// @param list_start the start of the plist (bucket list)
  /// @param bucket the bucket to manipulate
  /// @param baseptr the new pointer that bucket should have
  //
  // [mfs] I don't really understand this
  // [esl] Is supposed to be a short-hand for manipulating the EList/PList
  // pointer within a bucket.
  //       I tried to update my documentation to make that more clear
  inline void change_bucket_pointer(std::shared_ptr<rdma_capability> pool, remote_plist list_start, uint64_t bucket,
                                    remote_baseptr baseptr) {
    rdma_ptr<remote_baseptr> bucket_ptr = get_baseptr(list_start, bucket);
    // [mfs] Can this address manipulation be hidden?
    // [esl] todo: I think Remus needs to support for the [] operator in the
    // remote ptr...
    //             Otherwise I am forced to manually calculate the pointer of a
    //             bucket
    if (!pool->is_local(bucket_ptr)) {
      // Have to use a temp variable to account for alignment. Remote pointer is
      // 8 bytes!
      // [esl] This "deallocate 8" is a hack to get around a remus memory leak.
      // todo: Must be fixed in remus
      auto temp = pool->Allocate<remote_baseptr>(); // ? because of alignment, this
                                                    // might be 8 bytes. which is why
                                                    // the leak happens...
      pool->Write<remote_baseptr>(bucket_ptr, baseptr, temp);
      pool->Deallocate<remote_baseptr>(temp, 8);
    } else {
      *bucket_ptr = baseptr;
    }
  }

  /// @brief Hashing function to decide bucket size
  /// @param key the key to hash
  /// @param level the level in the iht
  /// @param count the number of buckets to hash into
  inline uint64_t level_hash(const K &key, size_t level, size_t count) {
    // 1) pre_hash will first map the type K to size_t
    //    then pre_hash will help distribute non-uniform inputs evenly by
    //    applying a finalizer
    // 2) We use count-1 to ensure the bucket count is co-prime with the other
    // plist bucket counts
    //    B/C of the property: A key maps to a suboptimal set of values when
    //    modding by 2A given "k mod A = Y" (where Y becomes the parent bucket)
    //    This happens because the hashing function maintains divisibility.
    return (level ^ pre_hash(key, level)) % (count - 1);
  }

  /// Rehash function
  /// @param pool The memory pool capability
  /// @param parent The P-List whose bucket needs rehashing
  /// @param pcount The number of elements in `parent`
  /// @param pdepth The depth of `parent`
  /// @param pidx   The index in `parent` of the bucket to rehash
  remote_plist rehash(std::shared_ptr<rdma_capability> pool, remote_plist parent, size_t pcount, size_t pdepth,
                      size_t pidx) {
    // pow(2, pdepth);
    pcount = pcount * 2;
    // how much bigger than original size we are
    int plist_size_factor = (pcount / PLIST_SIZE);

    // 2 ^ (depth) ==> in other words (depth:factor). 0:1, 1:2, 2:4, 3:8, 4:16,
    // 5:32.
    remote_plist new_p = pool->Allocate<PList>(plist_size_factor);
    InitPList(new_p, plist_size_factor);

    // hash everything from the full elist into it
    remote_elist parent_bucket = static_cast<remote_elist>(parent->buckets[pidx].base);
    remote_elist source = pool->is_local(parent_bucket) ? parent_bucket : pool->Read<EList>(parent_bucket);
    for (size_t i = 0; i < source->count; i++) {
      uint64_t b = level_hash(source->pairs[i].key, pdepth + 1, pcount);
      if (is_null(new_p->buckets[b].base)) {
        remote_elist e = pool->Allocate<EList>();
        new_p->buckets[b].base = static_cast<remote_baseptr>(e);
      }
      remote_elist dest = static_cast<remote_elist>(new_p->buckets[b].base);
      dest->elist_insert(source->pairs[i]);
    }
    // Deallocate the old elist
    // TODO replace for a remote deallocation at some point
    pool->Deallocate<EList>(source);
    return new_p;
  }

public:
  RdmaIHT(Peer self) : self_(self) {
    // I want to make sure we are choosing PLIST_SIZE and ELIST_SIZE to best use
    // the space (b/c of alignment)
    if ((PLIST_SIZE * sizeof(plist_pair_t)) % 64 != 0) {
      REMUS_WARN("Suboptimal PLIST_SIZE b/c PList aligned to 64 bytes");
    }
    if (((ELIST_SIZE * sizeof(pair_t)) + sizeof(size_t)) % 64 < 60) {
      REMUS_WARN("Suboptimal ELIST_SIZE b/c EList aligned to 64 bytes");
    }
  };

  /// @brief Create a fresh iht
  /// @param pool the capability to init the IHT with
  /// @return the iht root pointer
  rdma_ptr<anon_ptr> InitAsFirst(std::shared_ptr<rdma_capability> pool) {
    remote_plist iht_root = pool->Allocate<PList>();
    InitPList(iht_root, 1);
    this->root = iht_root;
    return static_cast<rdma_ptr<anon_ptr>>(iht_root);
  }

  /// @brief Initialize an IHT from the pointer of another IHT
  /// @param root_ptr the root pointer of the other iht from InitAsFirst();
  void InitFromPointer(rdma_ptr<anon_ptr> root_ptr) { this->root = static_cast<remote_plist>(root_ptr); }

  /// @brief Gets a value at the key.
  /// @param pool the capability providing one-sided RDMA
  /// @param key the key to search on
  /// @return an optional containing the value, if the key exists
  std::optional<V> contains(std::shared_ptr<rdma_capability> pool, K key) {
    // start at root
    remote_plist curr = pool->Read<PList>(root);
    remote_plist parent_ptr = root;
    size_t depth = 1, count = PLIST_SIZE;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      // Normal descent
      if (curr->buckets[bucket].lock == P_UNLOCKED) {
        remote_plist bucket_base = static_cast<remote_plist>(curr->buckets[bucket].base);
        pool->Deallocate<PList>(curr, 1 << (depth - 1)); // deallocate if curr was not ours
        curr = pool->ExtendedRead<PList>(bucket_base, 1 << depth);
        parent_ptr = bucket_base;
        depth++;
        count *= 2;
        continue;
      }

      // Erroneous descent into EList (Think we are at an EList, but it turns
      // out its a PList)
      if (!acquire(pool, get_lock(parent_ptr, bucket))) {
        // We must re-fetch the PList to ensure freshness of our pointers (1 <<
        // depth-1 to adjust size of read with customized ExtendedRead)
        pool->Deallocate<PList>(curr, 1 << (depth - 1)); // deallocate if curr was not ours
        curr = pool->ExtendedRead<PList>(parent_ptr, 1 << (depth - 1));
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base = static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e =
        pool->is_local(bucket_base) || is_null(bucket_base) ? bucket_base : pool->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // empty elist
        unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
        // deallocate plist that brought us to the empty elist
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        // Note: we don't deallocate e because it is a nullptr!
        return std::nullopt;
      }
      // Get elist and linear search
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the key
        pair_t kv = e->pairs[i];
        if (kv.key == key) {
          V result = kv.val;
          unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
          // Deallocate the elist and the plist used in descent
          if (!pool->is_local(bucket_base))
            pool->Deallocate<EList>(e);
          pool->Deallocate<PList>(curr, 1 << (depth - 1));
          return std::make_optional<V>(result);
        }
      }

      // Can't find, unlock and return false
      unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
      // Deallocate elist and plist used in descent
      if (!pool->is_local(bucket_base))
        pool->Deallocate<EList>(e);
      pool->Deallocate<PList>(curr, 1 << (depth - 1));
      return std::nullopt;
    }
  }

  /// @brief Insert a key and value into the iht. Result will become the value
  /// at the key if already present.
  /// @param pool the capability providing one-sided RDMA
  /// @param key the key to insert
  /// @param value the value to associate with the key
  /// @return an empty optional if the insert was successful. Otherwise it's the
  /// value at the key.
  std::optional<V> insert(std::shared_ptr<rdma_capability> pool, K key, V value) {
    // start at root
    remote_plist curr = pool->Read<PList>(root);
    remote_plist parent_ptr = root;
    size_t depth = 1, count = PLIST_SIZE;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      // Normal descent
      if (curr->buckets[bucket].lock == P_UNLOCKED) {
        remote_plist bucket_base = static_cast<remote_plist>(curr->buckets[bucket].base);
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        curr = pool->ExtendedRead<PList>(bucket_base, 1 << depth);
        parent_ptr = bucket_base;
        depth++;
        count *= 2;
        continue;
      }

      // Erroneous descent into EList (Think we are at an EList, but it turns
      // out its a PList)
      if (!acquire(pool, get_lock(parent_ptr, bucket))) {
        // We must re-fetch the PList to ensure freshness of our pointers (1 <<
        // depth-1 to adjust size of read with customized ExtendedRead)
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        curr = pool->ExtendedRead<PList>(parent_ptr, 1 << (depth - 1));
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base = static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e =
        pool->is_local(bucket_base) || is_null(bucket_base) ? bucket_base : pool->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // If we are we need to allocate memory for our elist
        remote_elist e_new = pool->Allocate<EList>();
        e_new->count = 0;
        e_new->elist_insert(key, value);
        remote_baseptr e_base = static_cast<remote_baseptr>(e_new);
        // modify the parent's bucket's pointer and unlock
        change_bucket_pointer(pool, parent_ptr, bucket, e_base);
        unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
        pool->Deallocate<PList>(curr,
                                1 << (depth - 1)); // deallocate current plist
        // successful insert
        return std::nullopt;
      }

      // We have recursed to an non-empty elist
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the key
        pair_t kv = e->pairs[i];
        if (kv.key == key) {
          V result = kv.val;
          // Contains the key => unlock and return false
          unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
          // Deallocate elist and plist used in descent
          if (!pool->is_local(bucket_base))
            pool->Deallocate<EList>(e);
          pool->Deallocate<PList>(curr, 1 << (depth - 1));
          return std::make_optional<V>(result);
        }
      }

      // Check for enough insertion room
      if (e->count < ELIST_SIZE) {
        // insert, unlock, return
        e->elist_insert(key, value);
        // If we are modifying a local copy, we need to write to the remote at
        // the end
        if (!pool->is_local(bucket_base))
          pool->Write<EList>(static_cast<remote_elist>(bucket_base), *e);
        // unlock and return true
        unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
        if (!pool->is_local(bucket_base))
          pool->Deallocate<EList>(e);
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        return std::nullopt;
      }

      // Need more room so rehash into plist and perma-unlock
      // todo? Make it possible to prevent the loop by inserting with the
      // current key/value in rehash function call
      remote_plist p = rehash(pool, curr, count, depth, bucket);
      // modify the bucket's pointer, keeping local curr updated with remote
      // curr
      curr->buckets[bucket].base = static_cast<remote_baseptr>(p);
      change_bucket_pointer(pool, parent_ptr, bucket, static_cast<remote_baseptr>(p));
      // unlock bucket
      curr->buckets[bucket].lock = P_UNLOCKED;
      unlock(pool, get_lock(parent_ptr, bucket), P_UNLOCKED);
      if (!pool->is_local(bucket_base))
        pool->Deallocate<EList>(e);
      // repeat from top, inserting into the bucket we just rehashed
    }
  }

  /// @brief Will remove a value at the key. Will stored the previous value in
  /// result.
  /// @param pool the capability providing one-sided RDMA
  /// @param key the key to remove at
  /// @return an optional containing the old value if the remove was successful.
  /// Otherwise an empty optional.
  std::optional<V> remove(std::shared_ptr<rdma_capability> pool, K key) {
    // start at root
    remote_plist curr = pool->Read<PList>(root);
    remote_plist parent_ptr = root;
    size_t depth = 1, count = PLIST_SIZE;
    while (true) {
      uint64_t bucket = level_hash(key, depth, count);
      // Normal descent
      if (curr->buckets[bucket].lock == P_UNLOCKED) {
        remote_plist bucket_base = static_cast<remote_plist>(curr->buckets[bucket].base);
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        curr = pool->ExtendedRead<PList>(bucket_base, 1 << depth);
        parent_ptr = bucket_base;
        depth++;
        count *= 2;
        continue;
      }

      // Erroneous descent into EList (Think we are at an EList, but it turns
      // out its a PList)
      if (!acquire(pool, get_lock(parent_ptr, bucket))) {
        // We must re-fetch the PList to ensure freshness of our pointers (1 <<
        // depth-1 to adjust size of read with customized ExtendedRead)
        pool->Deallocate<PList>(curr, 1 << (depth - 1)); // deallocate if curr was not ours
        curr = pool->ExtendedRead<PList>(parent_ptr, 1 << (depth - 1));
        continue;
      }

      // We locked an elist, we can read the baseptr and progress
      remote_elist bucket_base = static_cast<remote_elist>(curr->buckets[bucket].base);
      remote_elist e =
        pool->is_local(bucket_base) || is_null(bucket_base) ? bucket_base : pool->Read<EList>(bucket_base);

      // Past this point we have recursed to an elist
      if (is_null(e)) {
        // empty elist, can just unlock and return false
        unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
        // Deallocate data used for descending
        pool->Deallocate<PList>(curr, 1 << (depth - 1));
        return std::nullopt;
      }

      // Get elist and linear search
      for (size_t i = 0; i < e->count; i++) {
        // Linear search to determine if elist already contains the value
        pair_t kv = e->pairs[i];
        if (kv.key == key) {
          V result = kv.val; // saving the previous value at key
          if (e->count > 1) {
            // Edge swap if not count=0|1
            e->pairs[i] = e->pairs[e->count - 1];
          }
          e->count -= 1;
          // If we are modifying the local copy, we need to write to the remote
          if (!pool->is_local(bucket_base))
            pool->Write<EList>(static_cast<remote_elist>(bucket_base), *e);
          // Unlock and return
          unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
          // Deallocate data used for descending
          if (!pool->is_local(bucket_base))
            pool->Deallocate<EList>(e);
          pool->Deallocate<PList>(curr, 1 << (depth - 1));
          return std::make_optional<V>(result);
        }
      }

      // Can't find, unlock and return false
      unlock(pool, get_lock(parent_ptr, bucket), E_UNLOCKED);
      // Deallocate data used for descending
      if (!pool->is_local(bucket_base))
        pool->Deallocate<EList>(e);
      pool->Deallocate<PList>(curr, 1 << (depth - 1));
      return std::nullopt;
    }
  }

  /// @brief Populate only works when we have numerical keys. Will add data
  /// @param pool the capability providing one-sided RDMA
  /// @param count the number of values to insert. Recommended in total to do
  /// key_range / 2
  /// @param key_lb the lower bound for the key range
  /// @param key_ub the upper bound for the key range
  /// @param value the value to associate with each key. Currently, we have
  /// asserts for result to be equal to the key. Best to set value equal to key!
  void populate(std::shared_ptr<rdma_capability> pool, int op_count, K key_lb, K key_ub, std::function<K(V)> value) {
    // Populate only works when we have numerical keys
    K key_range = key_ub - key_lb;

    // Create a random operation generator that is
    // - evenly distributed among the key range
    std::uniform_real_distribution<double> dist = std::uniform_real_distribution<double>(0.0, 1.0);
    std::default_random_engine gen((unsigned)std::time(NULL));
    for (int c = 0; c < op_count; c++) {
      int k = (dist(gen) * key_range) + key_lb;
      insert(pool, k, value(k));
      // Wait some time before doing next insert...
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
  }
};
