#pragma once

#include <algorithm>

#include "macros.h"

namespace remus {

class EmptyValue {
public:
  typedef void *data_type;
  inline void set_value(data_type) {}
  data_type value() const { return nullptr; }
};

namespace internal {

// A wrapper for concrete value types to avoid requiring that the user define
// the API themselves. Types can be simple types (e.g., int, double, pointers)
// or they can be structs and classes.
template <typename T> class ROME_PACKED Value {
public:
  typedef T data_type;
  explicit Value(const data_type &value) : value_(value) {}
  Value(const Value &copy) : value_(copy.value_) {}
  Value(Value &&src) : value_(std::move(src.value_)) {}

  inline data_type value() const { return value_; }
  inline void set_value(const data_type &value) { value_ = value; }

private:
  data_type value_;
};

} // namespace internal

} // namespace remus
