#include <concepts>
#include <numeric>
#include <type_traits>

#include "../../utility/annotations.h"

#pragma once

namespace remus::hds::kv_linked_list {

template <typename K, typename V, int N, template <typename, typename, int> typename node_pointer_>
struct kv_inplace_construct {

  using node_pointer = node_pointer_<K, V, N>;
  using node = typename node_pointer::node;

  constexpr HDS_HOST_DEVICE node_pointer operator()(node *ptr) const { return new (ptr) node(); }
};

}; // namespace remus::hds::kv_linked_list
