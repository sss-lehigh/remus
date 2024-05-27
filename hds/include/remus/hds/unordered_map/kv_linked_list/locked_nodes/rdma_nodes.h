#include <ranges>

#include <protos/workloaddriver.pb.h>
#include <remus/logging/logging.h>
#include <remus/rdma/memory_pool.h>
#include <remus/rdma/rdma.h>

#include "node_t.h"

#pragma once

namespace remus::hds::kv_linked_list::locked_nodes {

template <typename K, typename V, int N> struct rdma_node_t {
  alignas(128) uint64_t lock_ = 0;
  alignas(128) utility::uint32_bitset<N> present = 0;
  alignas(128) K keys[N];
  remus::rdma::rdma_ptr<rdma_node_t<K, V, N>> next_ = nullptr;
  alignas(128) V values[N];
};

template <typename K, typename V, int N> struct rdma_node_pointer;

template <typename K, typename V, int N, typename Group> struct rdma_node_reference {

  static_assert(Group::size == 1, "Only support single thread group for now");

  using node = rdma_node_t<K, V, N>;

  template <typename U> using rdma_ptr = remus::rdma::rdma_ptr<U>;

  using rdma_capability = remus::rdma::rdma_capability;

  constexpr static size_t size_of_group = Group::size;

  HDS_HOST_DEVICE int index_of_key(K x, Group &group) {
    bool found = false;
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] == x) {
        return i;
      }
    }
    return -1;
  }

  HDS_HOST_DEVICE V get(int idx) { return local->values[idx]; }

  HDS_HOST_DEVICE bool has_key(K x, Group &group) {
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] == x) {
        return true;
      }
    }
    return false;
  }

  HDS_HOST_DEVICE int empty_index(Group &group) {
    for (int i = 0; i < N; ++i) {
      if (!local->present.get(i)) {
        return i;
      }
    }
    return -1;
  }

  HDS_HOST_DEVICE bool has_any_lt_key(K x, Group &group) {
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] < x) {
        return true;
      }
    }

    return false;
  }

  HDS_HOST_DEVICE bool has_any_gt_key(K x, Group &group) {
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] > x) {
        return true;
      }
    }

    return false;
  }

  HDS_HOST_DEVICE bool has_any_gte_key(K x, Group &group) {
    for (int i = 0; i < N / size_of_group; ++i) {
      if (local->present.get(i) && local->keys[i] >= x) {
        return true;
      }
    }

    return false;
  }

  HDS_HOST_DEVICE K max_key(Group &group) {
    K max_key_ = std::numeric_limits<K>::min();
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] >= max_key_) {
        max_key_ = local->keys[i];
      }
    }

    return max_key_;
  }

  HDS_HOST_DEVICE K min_key(Group &group) {
    K min_key_ = std::numeric_limits<K>::max();
    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] <= min_key_) {
        min_key_ = local->keys[i];
      }
    }

    return min_key_;
  }

  HDS_HOST_DEVICE void partition_by_and_insert(K k, V v, rdma_node_pointer<K, V, N> left, Group &group) {

    // TODO check left is local and assume it is

    assert(left != nullptr);

    utility::uint32_bitset<N> bits = 0;

    // src, dst
    std::ranges::copy(std::span<K, N>(local->keys), left.ptr->keys);
    std::ranges::copy(std::span<V, N>(local->values), left.ptr->values);

    for (int i = 0; i < N; ++i) {
      if (local->present.get(i) && local->keys[i] < k) {
        // tell leader to store present bit set at i
        bits.set(i);
      }
    }

    left.ptr->present = bits;

    uint32_t xored_bits = static_cast<uint32_t>(bits) ^ static_cast<uint32_t>(local->present);

    auto new_present = utility::uint32_bitset<N>(xored_bits);
    self.ptr->present = new_present;

    for (int i = 0; i < N; ++i) {
      if (!bits.get(i)) {
        left.set(i, k, v);
        break;
      }
      if (!new_present.get(i)) {
        self.set(i, k, v);
        break;
      }
    }
  }

  HDS_HOST_DEVICE rdma_node_pointer<K, V, N> next(Group &group) {
    return rdma_node_pointer<K, V, N>(local->next_, ctx);
  }

  template <typename S, typename T, int O> friend struct rdma_node_pointer;

private:
  rdma_node_pointer<K, V, N> self;
  rdma_ptr<node> local;
  rdma_capability *ctx;
};

struct rdma_pointer_constructor {

  rdma_pointer_constructor(remus::rdma::rdma_capability *ctx_) : ctx(ctx_) {}

  template <typename K, typename V, int N>
  rdma_node_pointer<K, V, N> operator()(remus::rdma::rdma_ptr<rdma_node_t<K, V, N>> ptr) {
    return rdma_node_pointer<K, V, N>(ptr, ctx);
  }

  remus::rdma::rdma_capability *ctx;
};

template <typename K, typename V, int N> struct rdma_node_pointer {

  template <typename S, typename T, int O, typename Group> friend struct rdma_node_reference;

  using node = rdma_node_t<K, V, N>;

  template <typename U> using rdma_ptr = remus::rdma::rdma_ptr<U>;

  using rdma_capability = remus::rdma::rdma_capability;

  HDS_HOST_DEVICE constexpr rdma_node_pointer(std::nullptr_t ptr_) : ptr(ptr_), ctx(nullptr) {}

  HDS_HOST_DEVICE constexpr rdma_node_pointer(rdma_ptr<node> ptr_, rdma_capability *ctx_) : ptr(ptr_), ctx(ctx_) {}

  constexpr rdma_node_pointer() = default;
  constexpr rdma_node_pointer(const rdma_node_pointer &) = default;
  constexpr rdma_node_pointer(rdma_node_pointer &&) = default;
  constexpr rdma_node_pointer &operator=(const rdma_node_pointer &) = default;
  constexpr rdma_node_pointer &operator=(rdma_node_pointer &&) = default;

  HDS_HOST_DEVICE constexpr bool operator==(const rdma_node_pointer<K, V, N> &rhs) const { return ptr == rhs.ptr; }

  HDS_HOST_DEVICE constexpr bool operator==(std::nullptr_t rhs) const { return ptr == rhs; }

  HDS_HOST_DEVICE constexpr explicit operator rdma_ptr<node>() { return ptr; }

  template <typename Group>
  HDS_HOST_DEVICE rdma_node_reference<K, V, N, std::remove_cvref_t<Group>> load(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    static_assert(N % std::remove_cvref_t<Group>::size == 0);
    static_assert(std::remove_cvref_t<Group>::size <= N);

    rdma_node_reference<K, V, N, std::remove_cvref_t<Group>> result;
    result.self = *this;
    result.local = ctx->Read<node>(ptr);
    local_copy = result.local;
    result.ctx = ctx;

    return result;
  }

  HDS_HOST_DEVICE void set(int offset, K k, V v) {

    if (local_copy == nullptr) {
      local_copy = ctx->Read<node>(ptr);
    }

    auto keys_ptr = rdma_ptr<char>(ptr) + offsetof(node, keys) + sizeof(K) * offset;
    auto values_ptr = rdma_ptr<char>(ptr) + offsetof(node, values) + sizeof(V) * offset;
    ctx->Write(rdma_ptr<K>(keys_ptr), k);
    ctx->Write(rdma_ptr<V>(values_ptr), v);

    auto tmp = local_copy->present;
    tmp.set(offset, true);

    auto present_ptr = rdma_ptr<char>(ptr) + offsetof(node, present);
    ctx->Write(rdma_ptr<utility::uint32_bitset<N>>(present_ptr), tmp);
  }

  HDS_HOST_DEVICE void remove(int offset) {
    if (local_copy == nullptr) {
      local_copy = ctx->Read<node>(ptr);
    }
    auto tmp = local_copy->present;
    tmp.set(offset, false);

    auto present_ptr = rdma_ptr<char>(ptr) + offsetof(node, present);
    ctx->Write(rdma_ptr<utility::uint32_bitset<N>>(present_ptr), tmp);
  }

  HDS_HOST_DEVICE bool is_empty() {

    if (local_copy == nullptr) {
      local_copy = ctx->Read<node>(ptr);
    }
    auto tmp = local_copy->present;

    for (int i = 0; i < N; ++i) {
      if (tmp.get(i)) {
        return false;
      }
    }

    return true;
  }

  template <typename Group> HDS_HOST_DEVICE rdma_node_pointer next(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    auto next_ptr = rdma_ptr<rdma_ptr<node>>(rdma_ptr<char>(ptr) + offsetof(node, next_));
    rdma_ptr<rdma_ptr<node>> local_copy_of_next = ctx->Read<rdma_ptr<node>>(next_ptr);
    rdma_node_pointer next_node_pointer(*local_copy_of_next, ctx);
    ctx->Deallocate(local_copy_of_next, 1);
    return next_node_pointer;
  }

  HDS_HOST_DEVICE rdma_node_pointer unsafe_next() {
    auto next_ptr = rdma_ptr<rdma_ptr<node>>(rdma_ptr<char>(ptr) + offsetof(node, next_));
    rdma_ptr<rdma_ptr<node>> local_copy_of_next = ctx->Read<rdma_ptr<node>>(next_ptr);
    rdma_node_pointer next_node_pointer(*local_copy_of_next, ctx);
    ctx->Deallocate(local_copy_of_next, 1);
    return next_node_pointer;
  }

  HDS_HOST_DEVICE void store_next(rdma_node_pointer next) {
    auto next_ptr = rdma_ptr<char>(ptr) + offsetof(node, next_);
    ctx->Write<rdma_ptr<node>>(rdma_ptr<rdma_ptr<node>>(next_ptr), next.ptr);
  }

  HDS_HOST_DEVICE static rdma_node_pointer init_locked_node(rdma_node_pointer n) {
    // For now assume node is allocated locally

    atomic_ref<uint64_t>(n.ptr->lock_).store(1, memory_order_relaxed);
    atomic_ref<utility::uint32_bitset<N>>(n.ptr->present).store(0, memory_order_relaxed);
    n.ptr->next_ = nullptr;
    return n;
  }

  template <typename Group> HDS_HOST_DEVICE void lock_unsync(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    if (group.is_leader()) {
      lock_unsync_leader();
    }
  }

  HDS_HOST_DEVICE void lock_unsync_leader() {
    auto lock_ptr = rdma_ptr<char>(ptr) + offsetof(node, lock_);
    while (!ctx->CompareAndSwap<uint64_t>(rdma_ptr<uint64_t>(lock_ptr), 0, 1))
      ;
  }

  template <typename Group> HDS_HOST_DEVICE void lock(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    lock_unsync(group);
    group.sync();
  }

  template <typename Group> HDS_HOST_DEVICE void unlock_unsync(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    if (group.is_leader()) {
      unlock_unsync_leader();
    }
  }

  HDS_HOST_DEVICE void unlock_unsync_leader() {
    auto lock_ptr = rdma_ptr<char>(ptr) + offsetof(node, lock_);
    ctx->Write<uint64_t>(rdma_ptr<uint64_t>(lock_ptr), 0);
  }

  template <typename Group> HDS_HOST_DEVICE void unlock(Group &group) {
    static_assert(std::remove_cvref_t<Group>::size == 1, "Only support single thread group for now");
    group.sync();
    unlock_unsync(group);
  }

  HDS_HOST_DEVICE void print() {

    if (local_copy == nullptr) {
      local_copy = ctx->Read<node>(ptr);
    }
    printf("Node: %p\n", this);

    bool have_min = false;
    K min_key;

    bool have_max = false;
    K max_key;
    printf("Present : %x\n", static_cast<uint32_t>(ptr->present));
    for (int i = 0; i < N; ++i) {
      if (local_copy->present.get(i)) {
        printf("Key : %d Value : %d\n", static_cast<int>(local_copy->keys[i]), static_cast<int>(local_copy->values[i]));
        if (!have_min || local_copy->keys[i] < min_key) {
          min_key = local_copy->keys[i];
          have_min = true;
        }
        if (!have_max || local_copy->keys[i] > max_key) {
          max_key = local_copy->keys[i];
          have_max = true;
        }
      }
    }

    printf("Min : %d | Max : %d\n", min_key, max_key);
    printf("Next : %p\n", local_copy->next_);
    printf("\n");
  }

private:
  rdma_ptr<node> ptr = nullptr;
  rdma_ptr<node> local_copy = nullptr; // TODO make safe shared with reference
  rdma_capability *ctx = nullptr;
};

}; // namespace remus::hds::kv_linked_list::locked_nodes