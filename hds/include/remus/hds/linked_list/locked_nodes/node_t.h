#include "../../utility/bitset.h"

#pragma once

namespace remus::hds::locked_nodes {

template<typename T, int N>
struct node_t {
  alignas(128) uint64_t lock_ = 0;
  alignas(128) utility::uint32_bitset<N> present = 0;
  alignas(128) T values[N];
  node_t* next_ = nullptr;
};

}

