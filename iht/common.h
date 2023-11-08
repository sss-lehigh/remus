#pragma once

#include "../util/tcp/tcp.h"

/// @brief a type used for templating remote pointers as anonymous (for
/// exchanging over the network where the types are "lost")
struct anon_ptr {};

/// @brief  IHT_Op is used by the Client Adaptor to pass in operations to Apply,
///         by forming a stream of IHT_Ops.
template <typename K, typename V> struct IHT_Op {
  int op_type;
  K key;
  V value;
  IHT_Op(int op_type_, K key_, V value_)
      : op_type(op_type_), key(key_), value(value_){};
};

#define CONTAINS 0
#define INSERT 1
#define REMOVE 2
#define CNF_ELIST_SIZE 7   // 7
#define CNF_PLIST_SIZE 128 // 128