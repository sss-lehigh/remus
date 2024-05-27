#include <concepts>
#include <numeric>
#include <type_traits>

#include "../../../utility/array.h"
#include "../../../utility/atomic.h"
#include "../../../utility/bitset.h"

#include "node_t.h"

#pragma once

namespace remus::hds::kv_linked_list::locked_nodes {

template <typename K, typename V, int N> struct reg_cached_node_pointer;

template <size_t size> struct tiling {
  HDS_HOST_DEVICE constexpr int operator()(int idx, int thread) { return thread + idx * size; }
};

template <typename K, typename V, int N, typename Group> struct reg_cached_node_reference {

  constexpr static size_t size_of_group = Group::size;

  HDS_HOST_DEVICE int index_of_key(K x, Group &group) {
    int i = 0;
    bool found = false;
    for (; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] == x) {
        found = true;
        break;
      }
    }

    int index = group.ballot_index(found);

    if (index != -1) {
      tiling<size_of_group> tile{};
      int offset = tile(i, group.thread_rank());
      return group.shfl(offset, index);
    } else {
      return -1;
    }
  }

  HDS_HOST_DEVICE V get(int idx) { return atomic_ref<V>(self.ptr->values[idx]).load(memory_order_relaxed); }

  HDS_HOST_DEVICE bool has_key(K x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] == x) {
        found = true;
        break;
      }
    }
    return group.any(found);
  }

  HDS_HOST_DEVICE int empty_index(Group &group) {
    bool found = false;
    int idx = 0;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (!present.get(i)) {
        found = true;
        idx = i;
        break;
      }
    }

    int index = group.ballot_index(found);

    if (index != -1) {
      tiling<size_of_group> tile{};
      int offset = tile(idx, group.thread_rank());
      return group.shfl(offset, index);
    }

    return -1;
  }

  HDS_HOST_DEVICE bool has_any_lt_key(K x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] < x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE bool has_any_gt_key(K x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] > x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE bool has_any_gte_key(K x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] >= x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE K max_key(Group &group) {
    K max_key = std::numeric_limits<K>::min();
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] >= max_key) {
        max_key = k_regs[i];
      }
    }

    return group.reduce_max(max_key);
  }

  HDS_HOST_DEVICE K min_key(Group &group) {
    K min_key = std::numeric_limits<K>::max();
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] <= min_key) {
        min_key = k_regs[i];
      }
    }

    return group.reduce_min(min_key);
  }

  HDS_HOST_DEVICE void partition_by_and_insert(K k, V v, reg_cached_node_pointer<K, V, N> left, Group &group) {

    tiling<size_of_group> tile{};

    utility::uint32_bitset<N> my_bits = 0;

    // Store all keys in left
    for (int i = 0; i < N / size_of_group; ++i) {
      int offset = tile(i, group.thread_rank());
      atomic_ref<K>(left.ptr->keys[offset]).store(k_regs[i], memory_order_relaxed);
    }

    // load values
    array<K, N / Group::size> v_regs;
    for (int i = 0; i < N / size_of_group; ++i) {
      int offset = tile(i, group.thread_rank());
      v_regs[i] = atomic_ref<V>(self.ptr->values[offset]).load(memory_order_relaxed);
    }

    // store values
    for (int i = 0; i < N / size_of_group; ++i) {
      int offset = tile(i, group.thread_rank());
      atomic_ref<V>(left.ptr->values[offset]).store(v_regs[i], memory_order_relaxed);
    }

    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && k_regs[i] < k) {
        int offset = tile(i, group.thread_rank());
        // tell leader to store present bit set at i
        my_bits.set(offset);
      }
    }

    auto reduction = group.reduce_or(static_cast<uint32_t>(my_bits));

    // reduce or the bits
    auto bits = utility::uint32_bitset<N>(reduction);

    if (group.is_leader()) {

      atomic_ref<utility::uint32_bitset<N>>(left.ptr->present).store(bits, memory_order_relaxed);
      uint32_t xored_bits =
        static_cast<uint32_t>(bits) ^
        static_cast<uint32_t>(atomic_ref<utility::uint32_bitset<N>>(self.ptr->present).load(memory_order_relaxed));

      auto new_present = utility::uint32_bitset<N>(xored_bits);
      atomic_ref<utility::uint32_bitset<N>>(self.ptr->present).store(new_present, memory_order_seq_cst);

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
  }

  HDS_HOST_DEVICE reg_cached_node_pointer<K, V, N> next(Group &group) { return next_; }

  template <typename S, typename R, int O> friend struct reg_cached_node_pointer;

private:
  reg_cached_node_pointer<K, V, N> self;
  utility::uint32_bitset<N / Group::size> present;
  array<K, N / Group::size> k_regs;
  reg_cached_node_pointer<K, V, N> next_;
};

template <typename K, typename V, int N> struct reg_cached_node_pointer {

  template <typename S, typename R, int O, typename Group> friend struct reg_cached_node_reference;

  using node = node_t<K, V, N>;

  HDS_HOST_DEVICE constexpr reg_cached_node_pointer(node *ptr_) : ptr(ptr_) {}

  constexpr reg_cached_node_pointer() = default;
  constexpr reg_cached_node_pointer(const reg_cached_node_pointer &) = default;
  constexpr reg_cached_node_pointer(reg_cached_node_pointer &&) = default;
  constexpr reg_cached_node_pointer &operator=(const reg_cached_node_pointer &) = default;
  constexpr reg_cached_node_pointer &operator=(reg_cached_node_pointer &&) = default;

  HDS_HOST_DEVICE constexpr bool operator==(const reg_cached_node_pointer<K, V, N> &rhs) const {
    return ptr == rhs.ptr;
  }

  HDS_HOST_DEVICE constexpr explicit operator node *() { return ptr; }

  template <typename Group>
  HDS_HOST_DEVICE reg_cached_node_reference<K, V, N, std::remove_cvref_t<Group>> load(Group &group) {
    static_assert(N % std::remove_cvref_t<Group>::size == 0);
    static_assert(std::remove_cvref_t<Group>::size <= N);

    tiling<std::remove_cvref_t<Group>::size> tile{};
    reg_cached_node_reference<K, V, N, std::remove_cvref_t<Group>> result;
    result.self = *this;

    auto tmp = atomic_ref<utility::uint32_bitset<N>>(ptr->present).load(memory_order_relaxed);

    if constexpr (tile(N / std::remove_cvref_t<Group>::size - 1, std::remove_cvref_t<Group>::size - 1) < N) {
      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        result.k_regs[i] = atomic_ref<K>(ptr->keys[tile(i, group.thread_rank())]).load(memory_order_relaxed);
      }

      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        result.present.set(i, tmp.get(tile(i, group.thread_rank())));
      }

    } else {

      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if (tiling(i, group.thread_rank()) < N) {
          result.k_regs[i] = atomic_ref<K>(ptr->keys[tile(i, group.thread_rank())]).load(memory_order_relaxed);
        }
      }

      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if (tiling(i, group.thread_rank()) < N) {
          result.present.set(i, tmp.get(tile(i, group.thread_rank())));
        }
      }
    }

    result.next_ = next(group);

    return result;
  }

  HDS_HOST_DEVICE void set(int offset, K k, V v) {
    atomic_ref<K>(ptr->keys[offset]).store(k, memory_order_relaxed);
    atomic_ref<V>(ptr->values[offset]).store(v, memory_order_relaxed);

    auto tmp = atomic_ref<utility::uint32_bitset<N>>(ptr->present).load(memory_order_relaxed);
    tmp.set(offset, true);

    atomic_ref<utility::uint32_bitset<N>>(ptr->present).store(tmp, memory_order_relaxed);
  }

  HDS_HOST_DEVICE void remove(int offset) {
    auto tmp = atomic_ref<utility::uint32_bitset<N>>(ptr->present).load(memory_order_relaxed);
    tmp.set(offset, false);
    atomic_ref<utility::uint32_bitset<N>>(ptr->present).store(tmp, memory_order_relaxed);
  }

  HDS_HOST_DEVICE bool is_empty() {
    auto tmp = atomic_ref<utility::uint32_bitset<N>>(ptr->present).load(memory_order_relaxed);

    for (int i = 0; i < N; ++i) {
      if (tmp.get(i)) {
        return false;
      }
    }

    return true;
  }

  template <typename Group> HDS_HOST_DEVICE reg_cached_node_pointer next(Group &group) {
    return reg_cached_node_pointer(atomic_ref<node *>(ptr->next_).load(memory_order_relaxed));
  }

  HDS_HOST_DEVICE reg_cached_node_pointer unsafe_next() {
    return reg_cached_node_pointer(atomic_ref<node *>(ptr->next_).load(memory_order_relaxed));
  }

  HDS_HOST_DEVICE void store_next(reg_cached_node_pointer next) {
    atomic_ref<node *>(ptr->next_).store(next.ptr, memory_order_relaxed);
  }

  HDS_HOST_DEVICE static reg_cached_node_pointer init_locked_node(reg_cached_node_pointer n) {
    atomic_ref<uint64_t>(n.ptr->lock_).store(1, memory_order_relaxed);
    atomic_ref<utility::uint32_bitset<N>>(n.ptr->present).store(0, memory_order_relaxed);
    atomic_ref<node *>(n.ptr->next_).store(nullptr, memory_order_relaxed);
    return n;
  }

  template <typename Group> HDS_HOST_DEVICE void lock_unsync(Group &group) {
    if (group.is_leader()) {
      lock_unsync_leader();
    }
    hds::atomic_thread_fence(memory_order_acquire);
  }

  HDS_HOST_DEVICE void lock_unsync_leader() {
    while (true) {
      uint64_t expected = 0;
      if (atomic_ref<uint64_t>(ptr->lock_).compare_exchange_strong(expected, 1, memory_order_acq_rel)) {
        return;
      }
      while (atomic_ref<uint64_t>(ptr->lock_).load(memory_order_relaxed) != 0) {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__)
        __builtin_ia32_pause();
#endif
      }
    }
  }

  template <typename Group> HDS_HOST_DEVICE void lock(Group &group) {
    lock_unsync(group);
    group.sync();
  }

  template <typename Group> HDS_HOST_DEVICE void unlock_unsync(Group &group) {
    hds::atomic_thread_fence(memory_order_release);
    if (group.is_leader()) {
      unlock_unsync_leader();
    }
  }

  HDS_HOST_DEVICE void unlock_unsync_leader() { atomic_ref<uint64_t>(ptr->lock_).store(0, memory_order_release); }

  template <typename Group> HDS_HOST_DEVICE void unlock(Group &group) {
    group.sync();
    unlock_unsync(group);
  }

  HDS_HOST_DEVICE void print() {
    printf("Node: %p\n", this);

    bool have_min = false;
    K min_key;

    bool have_max = false;
    K max_key;
    printf("Present : %x\n", static_cast<uint32_t>(ptr->present));
    for (int i = 0; i < N; ++i) {
      if (ptr->present.get(i)) {
        printf("Key : %d Value : %d\n", static_cast<int>(ptr->keys[i]), static_cast<int>(ptr->values[i]));
        if (!have_min || ptr->keys[i] < min_key) {
          min_key = ptr->keys[i];
          have_min = true;
        }
        if (!have_max || ptr->keys[i] > max_key) {
          max_key = ptr->values[i];
          have_max = true;
        }
      }
    }

    printf("Min : %d | Max : %d\n", static_cast<int>(min_key), static_cast<int>(max_key));
    printf("Next : %p\n", ptr->next_);
    printf("\n");
  }

private:
  node *ptr = nullptr;
};

} // namespace remus::hds::kv_linked_list::locked_nodes