#include "../allocator/allocator.h"
#include "kv_linked_list/lock_linked_list.h"
#include "kv_linked_list/utility.h"

#pragma once

namespace remus::hds {

template <typename T> struct BasicHash {

  HDS_HOST_DEVICE constexpr unsigned operator()(T x) const { return static_cast<unsigned>(x); }
};

struct fast_div_mod {

  // Based on "Division by Invariant Integers using Multiplication"

  HDS_HOST_DEVICE fast_div_mod(uint32_t divisor_) : divisor(divisor_) {
    // l is ceil [ log divisor ]

    for (l = 0; l < 32; ++l) {
      if ((1 << l) >= divisor) {
        break;
      }
    }

    uint64_t one = 1;

    // 2 ^ 32 * ((2 ^ l) - divisor) / divisor + 1
    // gives us 2 ^ 32 * (2 ^ ceil [log divisor] - divisor) / divisor + 1
    m = static_cast<uint32_t>((one << 32ul) * ((one << l) - divisor) / divisor + 1ul);
  }

  HDS_HOST_DEVICE uint32_t div(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return (divisor != 1) ? (__umulhi(m, x) + x) >> l : x;
#else
    return (divisor != 1) ? (((static_cast<uint64_t>(m) * x) >> 32) + x) >> l : x;
#endif
  }

  HDS_HOST_DEVICE uint32_t mod(uint32_t x) { return x - div(x) * divisor; }

  uint32_t divisor;
  uint32_t m;
  uint32_t l;
};

template <typename K, typename V, int N,
          template <typename, typename, int, template <typename, typename, int> typename, typename, typename>
          typename linked_list_,
          template <typename, typename, int> typename node_pointer_, typename Allocator,
          typename Constructor = kv_linked_list::kv_inplace_construct<K, V, N, node_pointer_>,
          typename ArrayAllocator = Allocator, typename Hash = BasicHash<K>>
class unordered_map {
private:
  using kv_ll_t = linked_list_<K, V, N, node_pointer_, Allocator, Constructor>;

public:
  HDS_HOST_DEVICE unordered_map(uint32_t buckets_) : unordered_map(fast_div_mod(buckets_), Allocator{}) {}

  HDS_HOST_DEVICE unordered_map(fast_div_mod divmod_) : unordered_map(divmod_, Allocator{}) {}

  template <typename A>
    requires std::convertible_to<A, Allocator>
  HDS_HOST_DEVICE unordered_map(uint32_t buckets_, A &&alloc_)
    : unordered_map(fast_div_mod(buckets_), std::forward<A>(alloc)) {}

  template <typename A>
    requires std::convertible_to<A, Allocator>
  HDS_HOST_DEVICE unordered_map(fast_div_mod divmod_, A &&alloc_)
    : divmod(divmod_), alloc(std::forward<A>(alloc_)), aalloc(alloc) {
    buckets_array = aalloc.template allocate<kv_ll_t>(divmod.divisor);
    for (uint32_t i = 0; i < divmod.divisor; ++i) {
      new (buckets_array + i) kv_ll_t(alloc);
    }
  }

  template <typename A, typename B, typename C>
    requires std::convertible_to<A, Allocator> and std::convertible_to<B, ArrayAllocator> and
             std::convertible_to<C, Constructor>
  HDS_HOST_DEVICE unordered_map(uint32_t buckets_, A &&alloc_, B &&aalloc_, C &&construct)
    : unordered_map(fast_div_mod(buckets_), std::forward<A>(alloc_), std::forward<B>(aalloc_),
                    std::forward<C>(construct)) {}

  template <typename A, typename B, typename C>
    requires std::convertible_to<A, Allocator> and std::convertible_to<B, ArrayAllocator> and
               std::convertible_to<C, Constructor>
  HDS_HOST_DEVICE unordered_map(fast_div_mod divmod_, A &&alloc_, B &&aalloc_, C &&construct)
    : divmod(divmod_), alloc(std::forward<A>(alloc_)), aalloc(std::forward<B>(aalloc_)) {
    buckets_array = aalloc.template allocate<kv_ll_t>(divmod.divisor);
    for (uint32_t i = 0; i < divmod.divisor; ++i) {
      new (buckets_array + i) kv_ll_t(alloc, std::forward<C>(construct));
    }
  }

  HDS_HOST_DEVICE ~unordered_map() {

    for (uint32_t i = 0; i < divmod.divisor; ++i) {
      (buckets_array + i)->~kv_ll_t();
    }

    aalloc.deallocate(buckets_array, divmod.divisor);
  }

  template <typename Group> HDS_HOST_DEVICE optional<V> get(K k, Group &group) {
    auto idx = divmod.mod(Hash{}(k));
    return buckets_array[idx].get(k, group);
  }

  template <typename Group> HDS_HOST_DEVICE bool insert(K k, V v, Group &group) {
    auto idx = divmod.mod(Hash{}(k));
    return buckets_array[idx].insert(k, v, group);
  }

  template <typename Group> HDS_HOST_DEVICE bool remove(K k, Group &group) {
    auto idx = divmod.mod(Hash{}(k));
    return buckets_array[idx].remove(k, group);
  }

private:
  kv_ll_t *buckets_array;
  fast_div_mod divmod;
  Allocator alloc;
  ArrayAllocator aalloc;
};

}; // namespace remus::hds
