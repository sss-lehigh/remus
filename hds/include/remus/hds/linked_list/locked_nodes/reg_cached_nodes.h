#include <concepts>
#include <numeric>
#include <type_traits>

#include "../../utility/array.h"
#include "../../utility/atomic.h"
#include "../../utility/bitset.h"

#include "node_t.h"

#pragma once

namespace remus::hds::locked_nodes {

template <typename T, int N> struct reg_cached_node_pointer;

template <size_t size> struct tiling {
  HDS_HOST_DEVICE constexpr int operator()(int idx, int thread) { return thread + idx * size; }
};

template <typename T, int N, typename Group> struct reg_cached_node_reference {

  constexpr static size_t size_of_group = Group::size;

  HDS_HOST_DEVICE int index_of_key(T x, Group &group) {
    bool found = false;
    int i = 0;
    for (; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] == x) {
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

  HDS_HOST_DEVICE bool has_key(T x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] == x) {
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

  HDS_HOST_DEVICE bool has_any_lt_key(T x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] < x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE bool has_any_gt_key(T x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] > x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE bool has_any_gte_key(T x, Group &group) {
    bool found = false;
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] >= x) {
        found = true;
        break;
      }
    }

    return group.any(found);
  }

  HDS_HOST_DEVICE T max_key(Group &group) {
    T max_key = std::numeric_limits<T>::min();
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] >= max_key) {
        max_key = regs[i];
      }
    }

    return group.reduce_max(max_key);
  }

  HDS_HOST_DEVICE T min_key(Group &group) {
    T min_key = std::numeric_limits<T>::max();
    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] <= min_key) {
        min_key = regs[i];
      }
    }

    return group.reduce_min(min_key);
  }

  HDS_HOST_DEVICE void partition_by_and_insert(T x, reg_cached_node_pointer<T, N> left, Group &group) {

    tiling<size_of_group> tile{};

    utility::uint32_bitset<N> my_bits = 0;

    // Store all keys in left
    for (int i = 0; i < N / size_of_group; ++i) {
      int offset = tile(i, group.thread_rank());
      atomic_ref<T>(left.ptr->values[offset]).store(regs[i], memory_order_relaxed);
    }

    for (int i = 0; i < N / size_of_group; ++i) {
      if (present.get(i) && regs[i] < x) {
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
          left.set(i, x);
          break;
        }
        if (!new_present.get(i)) {
          self.set(i, x);
          break;
        }
      }
    }
  }

  HDS_HOST_DEVICE reg_cached_node_pointer<T, N> next(Group &group) { return next_; }

  template <typename S, int O> friend struct reg_cached_node_pointer;

private:
  reg_cached_node_pointer<T, N> self;
  utility::uint32_bitset<N / Group::size> present;
  array<T, N / Group::size> regs;
  reg_cached_node_pointer<T, N> next_;
};

template <typename T, int N> struct reg_cached_node_pointer {

  template <typename S, int O, typename Group> friend struct reg_cached_node_reference;

  using node = node_t<T, N>;

  HDS_HOST_DEVICE constexpr reg_cached_node_pointer(node *ptr_) : ptr(ptr_) {}

  constexpr reg_cached_node_pointer() = default;
  constexpr reg_cached_node_pointer(const reg_cached_node_pointer &) = default;
  constexpr reg_cached_node_pointer(reg_cached_node_pointer &&) = default;
  constexpr reg_cached_node_pointer &operator=(const reg_cached_node_pointer &) = default;
  constexpr reg_cached_node_pointer &operator=(reg_cached_node_pointer &&) = default;

  HDS_HOST_DEVICE constexpr bool operator==(const reg_cached_node_pointer<T, N> &rhs) const { return ptr == rhs.ptr; }

  HDS_HOST_DEVICE constexpr explicit operator node *() { return ptr; }

  template <typename Group>
  HDS_HOST_DEVICE reg_cached_node_reference<T, N, std::remove_cvref_t<Group>> load(Group &group) {
    static_assert(N % std::remove_cvref_t<Group>::size == 0);
    static_assert(std::remove_cvref_t<Group>::size <= N);

    tiling<std::remove_cvref_t<Group>::size> tile{};
    reg_cached_node_reference<T, N, std::remove_cvref_t<Group>> result;
    result.self = *this;

    auto tmp = atomic_ref<utility::uint32_bitset<N>>(ptr->present).load(memory_order_relaxed);

    if constexpr (tile(N / std::remove_cvref_t<Group>::size - 1, std::remove_cvref_t<Group>::size - 1) < N) {
      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        result.regs[i] = atomic_ref<T>(ptr->values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
      }

      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        result.present.set(i, tmp.get(tile(i, group.thread_rank())));
      }

    } else {

      for (int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if (tiling(i, group.thread_rank()) < N) {
          result.regs[i] = atomic_ref<T>(ptr->values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
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

  HDS_HOST_DEVICE void set(int offset, T x) {
    atomic_ref<T>(ptr->values[offset]).store(x, memory_order_relaxed);

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
    T min_key;

    bool have_max = false;
    T max_key;
    printf("Present : %x\n", static_cast<uint32_t>(ptr->present));
    for (int i = 0; i < N; ++i) {
      if (ptr->present.get(i)) {
        printf("Key : %d\n", ptr->values[i]);
        if (!have_min || ptr->values[i] < min_key) {
          min_key = ptr->values[i];
          have_min = true;
        }
        if (!have_max || ptr->values[i] > max_key) {
          max_key = ptr->values[i];
          have_max = true;
        }
      }
    }

    printf("Min : %d | Max : %d\n", min_key, max_key);
    printf("Next : %p\n", ptr->next_);
    printf("\n");
  }

private:
  node *ptr = nullptr;
};

} // namespace remus::hds::locked_nodes
