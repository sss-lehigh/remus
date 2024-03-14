#include "../../../utility/bitset.h"

#pragma once

namespace remus::hds::kv_linked_list::locked_nodes {

template <typename K, typename V, int N> struct node_t {
  alignas(128) uint64_t lock_ = 0;
  alignas(128) utility::uint32_bitset<N> present = 0;
  alignas(128) K keys[N];
  node_t *next_ = nullptr;
  alignas(128) V values[N];
};

} // namespace remus::hds::kv_linked_list::locked_nodes
