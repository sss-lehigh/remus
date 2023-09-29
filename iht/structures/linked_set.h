#pragma once

#include "../../rdma/memory_pool/memory_pool.h"
#include "../common.h"

using ::rome::rdma::MemoryPool;
using ::rome::rdma::remote_nullptr;
using ::rome::rdma::remote_ptr;

#define NODE_BUNDLE_SIZE 7

template <class K, class V> class alignas(64) LinkedSet {
private:
  // "Poor-mans" enum to represent the state of a node.
  // LOCKED = 1, UNLOCKED = 2, DELETED = 3
  // The deleted state allows us to mark linkedsets as deleted to allow checking
  // if my value
  uint64_t LOCKED = 1, UNLOCKED = 2, DELETED = 3;

  // "Super class" for the elist and plist structs
  typedef uint64_t lock_type;
  typedef remote_ptr<lock_type> remote_lock;

  struct alignas(64) Node {
    remote_ptr<Node> next;
    int length;
    int total_length;
    K key[NODE_BUNDLE_SIZE];
    V value[NODE_BUNDLE_SIZE];
  };

  typedef remote_ptr<Node> remote_node;

  // Lock on the data structure
  remote_lock lock;
  remote_node first;

  /// Acquire a lock on the bucket. Will prevent others from modifying it
  bool acquire(MemoryPool *pool, remote_lock lock) {
    // Spin while trying to acquire the lock
    while (true) {
      lock_type v = pool->CompareAndSwap<lock_type>(lock, UNLOCKED, LOCKED);

      // If we can switch from unlock to lock status
      if (v == UNLOCKED)
        return true;
      if (v == DELETED)
        return false;
    }
  }

  /// @brief Unlock a lock ==> the reverse of acquire
  /// @param lock the lock to unlock
  /// @param unlock_status what should the end lock status be.
  inline void unlock(MemoryPool *pool, remote_lock lock) {
    remote_lock temp = pool->Allocate<lock_type>();
    pool->Write<lock_type>(lock, UNLOCKED, temp);
    // Have to deallocate "8" of them to account for alignment (this is why we
    // prealloc the data)
    pool->Deallocate<lock_type>(temp, 8);
  }

  /// Edit the total count of the linkedlist to be either 1 greater or 1 less
  void changeCount(MemoryPool *pool, bool isIncrement) {
    remote_node node = pool->Read<Node>(first);
    Node node_pulled = *std::to_address(node);
    node_pulled.total_length += isIncrement ? 1 : -1;
    pool->Write<Node>(first, node_pulled);
    pool->Deallocate<Node>(node);
  }

public:
  LinkedSet(MemoryPool *pool) {
    lock = pool->Allocate<lock_type>();
    *std::to_address(lock) = UNLOCKED;
    first = pool->Allocate<Node>();
    Node node;
    node.next = remote_nullptr;
    node.length = 0;
    node.total_length = 0;
    *std::to_address(first) = node;
  };

  /// @brief Get the length of the list
  /// @param pool the pool to use as a resource
  /// @return the length
  int list_length(MemoryPool *pool) {
    remote_node node = pool->Read<Node>(first);
    Node node_pulled = *std::to_address(node);
    int total_length = node_pulled.total_length;
    pool->Deallocate<Node>(node);
    return total_length;
  }

  /// Freeze a LinkedSet and prevent changes to it
  void freeze(MemoryPool *pool) {
    // Spin while trying to acquire the lock
    while (true) {
      lock_type v = pool->CompareAndSwap<lock_type>(lock, UNLOCKED, LOCKED);

      // If we can switch from unlock to lock status
      if (v == UNLOCKED)
        return;
    }
  }

  /// Notify clients that the linked lists are detached completely
  /// Also, in the future, we might want to quaratine the "deleted" LinkedSets
  /// and deallocate them on next rehash
  void melt(MemoryPool *pool) {
    remote_lock temp = pool->Allocate<lock_type>();
    pool->Write<lock_type>(lock, DELETED, temp);
    // Have to deallocate "8" of them to account for alignment (this is why we
    // prealloc the data)
    pool->Deallocate<lock_type>(temp, 8);
  }

  /// A function to run an operation on each value in the linked list
  /// - Doesn't acquire a lock
  void foreach (MemoryPool *pool, std::function<void(K k, V v)> func) {
    remote_node node = first;
    while (node != remote_nullptr) {
      remote_node red_node = pool->Read<Node>(node);
      Node node_data = *std::to_address(red_node);

      // Iterate through bundles
      // (can turn on when printing to get a better idea of separation)
      // ROME_INFO("{}/{} agg={} ->", node_data.length, NODE_BUNDLE_SIZE,
      // node_data.total_length);
      for (int i = 0; i < node_data.length; i++) {
        func(node_data.key[i], node_data.value[i]);
      }
      node = node_data.next;
      pool->Deallocate<Node>(red_node);
    }
  }

  /// @brief Will insert a value if it doesn't exist
  /// @param value the value to insert
  /// @return if the insert was successful
  HT_Res<V> insert(MemoryPool *pool, K key, V value) {
    if (!acquire(pool, lock))
      return HT_Res<V>(REHASH_DELETED, value);
    remote_node node = first;

    Node node_data;
    remote_node red_node;
    while (node != remote_nullptr) {
      red_node = pool->Read<Node>(node);
      node_data = *std::to_address(red_node);

      // Check if the key already exists
      for (int i = 0; i < node_data.length; i++) {
        if (node_data.key[i] == key) {
          unlock(pool, lock);
          pool->Deallocate<Node>(red_node);
          return HT_Res<V>(FALSE_STATE, node_data.value[i]);
        }
      }
      if (node_data.length == NODE_BUNDLE_SIZE) {
        if (node_data.next == remote_nullptr) {
          // Full but needing more memory allocated
          remote_node new_node = pool->Allocate<Node>();
          Node new_node_data;
          new_node_data.next = remote_nullptr;
          new_node_data.key[0] = key;
          new_node_data.value[0] = value;
          new_node_data.length = 1;
          *std::to_address(new_node) = new_node_data;
          // Attach new node
          node_data.next = new_node;
          break;
        } else {
          // Everything is full and can continue onward
          node = node_data.next;
          pool->Deallocate<Node>(red_node);
        }
      } else {
        // Adding data into node
        node_data.key[node_data.length] = key;
        node_data.value[node_data.length] = value;
        node_data.length++;
        break;
      }
    }

    pool->Write<Node>(node, node_data);
    changeCount(pool, true);
    unlock(pool, lock);
    pool->Deallocate<Node>(red_node);
    return HT_Res<V>(TRUE_STATE, 0);
  }

  /// @brief Check if a key is contained in the linked set
  /// @param key the key to check
  /// @return if it exists
  HT_Res<V> contains(MemoryPool *pool, K key) {
    if (!acquire(pool, lock))
      return HT_Res<V>(REHASH_DELETED, 0);
    remote_node node = first;
    while (node != remote_nullptr) {
      remote_node red_node = pool->Read<Node>(node);
      Node node_data = *std::to_address(red_node);
      // Check if the value already exists
      for (int i = 0; i < node_data.length; i++) {
        if (node_data.key[i] == key) {
          unlock(pool, lock);
          pool->Deallocate<Node>(red_node);
          return HT_Res<V>(TRUE_STATE, node_data.value[i]);
        }
      }
      node = node_data.next;
      pool->Deallocate<Node>(red_node);
    }
    unlock(pool, lock);
    return HT_Res<V>(FALSE_STATE, 0);
  }

  /// @brief remove a value
  /// @param value the value to remove
  /// @return if it was successful
  HT_Res<V> remove(MemoryPool *pool, K key) {
    if (!acquire(pool, lock))
      return HT_Res<V>(REHASH_DELETED, 0);
    remote_node node = first;

    while (node != remote_nullptr) {
      remote_node red_node = pool->Read<Node>(node);
      Node node_data = *std::to_address(red_node);

      // Check if the value doesn't exist in this unit
      for (int i = 0; i < node_data.length; i++) {
        if (node_data.key[i] == key) {
          V old_value = node_data.value[i];
          // Delete
          node_data.key[i] = node_data.key[node_data.length - 1];
          node_data.value[i] = node_data.value[node_data.length - 1];
          node_data.length--;
          pool->Write<Node>(node, node_data);
          changeCount(pool, false);
          unlock(pool, lock);
          pool->Deallocate<Node>(red_node);
          return HT_Res<V>(TRUE_STATE, old_value);
        }
      }
      node = node_data.next;
      pool->Deallocate<Node>(red_node);
    }
    unlock(pool, lock);
    return HT_Res<V>(FALSE_STATE, 0);
  }
};
