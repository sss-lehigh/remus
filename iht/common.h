#pragma once

#define CONTAINS 0
#define INSERT 1
#define REMOVE 2
#define CNF_ELIST_SIZE 7   // 7
#define CNF_PLIST_SIZE 128 // 128

/// @brief  IHT_Op is used by the Client Adaptor to pass in operations to Apply,
///         by forming a stream of IHT_Ops.
template <typename K, typename V> struct IHT_Op {
  int op_type;
  K key;
  V value;
  IHT_Op(int op_type_, K key_, V value_)
      : op_type(op_type_), key(key_), value(value_){};
};

// [mfs] This should be an enum, but I don't really see why it's even needed.
typedef uint64_t state_value;
// Value states
// [mfs] What is "REHASH_DELETED"?
state_value FALSE_STATE = 1, TRUE_STATE = 2, REHASH_DELETED = 3;

/// @brief Output for IHT that includes a status and a value
// [mfs] An Optional would work better here...
template <typename T> struct HT_Res {
  state_value status;
  T result;

  HT_Res(state_value status, T result) {
    this->status = status;
    this->result = result;
  }
};
