#pragma once

#include <memory>
#include <remus/remus.h>

using namespace remus;

/// A lock-free sorted list-based map
///
/// @tparam K The type for keys in the map
/// @tparam V The type for values in the map
template <typename K, typename V> class LockFreeList {
  struct Node {
    Atomic<K> key;       // The key stored in this node
    Atomic<V> value;     // The value stored in this node
    Atomic<Node *> next; // Pointer to the next node

    /// Initialize a node.  Assume that it is called with `this` being a remote
    /// pointer
    ///
    /// @param k  The key to store
    /// @param v  The value to store
    /// @param ct The calling thread's Remus context
    void init(const K &k, const V &v, std::shared_ptr<ComputeThread> ct) {
      key.store(k, ct);
      value.store(v, ct);
      next.store(nullptr, ct);
    }
  };

  LockFreeList *This;  // The "this" pointer to use for data accesses
  Atomic<Node *> head; // The head pointer (ALWAYS USE `This->head`!)

public:
  /// Allocate a LockFreeList in remote memory and initialize it
  ///
  /// @param ct The calling thread's Remus context
  ///
  /// @return An rdma_ptr to the allocated/initialized list
  static rdma_ptr<LockFreeList> New(std::shared_ptr<ComputeThread> ct) {
    auto list = ct->New<LockFreeList>();
    list->head.store(nullptr, ct);
    return rdma_ptr<LockFreeList>((uintptr_t)list);
  }

  /// Construct a LockFreeList, setting its `This` to a remote memory location.
  /// Note that every thread in the program could have a unique LockFreeList,
  /// but if they all use the same `This`, they'll all access the same remote
  /// memory.
  ///
  /// @param This A LockFreeList produced by a preceding call to `New()`.
  LockFreeList(const remus::rdma_ptr<LockFreeList> &This)
      : This((LockFreeList *)((uintptr_t)This)) {}

  /// Insert a key/value pair into the list if the key doesn't already exist
  ///
  /// @param key The key to try and insert
  /// @param value The value to associate with that key
  /// @param ct The calling thread's Remus context
  /// @return True on success, false if the key was already present
  bool insert(const K &key, const V &value, std::shared_ptr<ComputeThread> ct) {
    // Make a new node, in the hope that we'll succeed
    Node *new_node = ct->New<Node>();
    new_node->init(key, value, ct);
    // On CAS failure we'll return to here
    while (true) {
      Node *prev = nullptr;
      Node *curr = This->head.load(ct);
      // Find insertion point
      // [mfs] This isn't correct, because it doesn't clean marked nodes
      while (curr && curr->key.load(ct) < key) {
        prev = curr;
        curr = curr->next.load(ct);
      }
      // If the key is found, reclaim immediately and return
      if (curr && curr->key.load(ct) == key) {
        ct->Delete(new_node);
        return false;
      }
      // Try to insert the node.  Insert at head if !prev
      new_node->next.store(curr, ct);
      if (!prev && This->head.compare_exchange_weak(curr, new_node, ct)) {
        return true;
      } else if (prev && prev->next.compare_exchange_weak(curr, new_node, ct)) {
        return true;
      }
      // CAS failed; retry
    }
  }

  /// Remove a key/value pair from the list
  ///
  /// @param key The key to remove
  /// @param ct  The calling thread's Remus context
  /// @return True on success, false if the key was not present
  bool remove(const K &key, std::shared_ptr<ComputeThread> ct) {
    // On CAS failure we'll return to here
    while (true) {
      Node *prev = nullptr;
      Node *curr = This->head.load(ct);
      // Find node to remove
      // [mfs] This isn't correct, because it doesn't clean marked nodes
      while (curr && curr->key.load(ct) < key) {
        prev = curr;
        curr = curr->next.load(ct);
      }
      // If the key is not found, return
      if (!curr || curr->key.load(ct) != key) {
        return false; // Not found
      }
      // [mfs] This should be marking the node (via its next pointer's low bit)
      Node *next = curr->next.load(ct);
      // Try to unstitch the node.  Unstitch at head if !prev
      if (!prev && This->head.compare_exchange_weak(curr, next, ct)) {
        // [mfs] Need to implement Reclaim
        ct->SchedReclaim(curr);
        return true;
      } else if (prev && prev->next.compare_exchange_weak(curr, next, ct)) {
        // [mfs] Important!!!
        ct->SchedReclaim(curr); // Use EBR here in production!
        return true;
      }
      // CAS failed; retry
    }
  }

  /// Return the value associated with the provided key, or NONE if the key is
  /// not found
  ///
  /// @param key The key to get
  /// @param ct  The calling thread's Remus context
  /// @return The found key's associated value, or NONE
  std::optional<K> get(const K &key, std::shared_ptr<ComputeThread> ct) {
    Node *curr = This->head.load(ct);
    // [mfs] This loop needs to clean marked nodes
    while (curr) {
      if (curr->key.load(ct) == key) {
        return curr->value.load(ct);
      } else if (curr->key.load(ct) > key) {
        return {};
      }
      curr = curr->next.load(ct);
    }
    return {};
  }

  /// "Destruct" the list by reclaiming all of its nodes, and then reclaiming
  /// its `This` pointer.  It is assumed that this runs in a single-threaded
  /// context, so that memory can be immediately reclaimed.
  ///
  /// @param ct The calling thread's Remus context
  void destroy(std::shared_ptr<ComputeThread> ct) {
    Node *curr = This->head.load(ct);
    while (curr) {
      Node *next = curr->next.load(ct);
      ct->Delete(curr);
      curr = next;
    }
    ct->Delete(This);
  }
};