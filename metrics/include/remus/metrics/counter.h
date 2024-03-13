#pragma once

#include <cstdint>

#include "abstract_metric.h"

namespace remus::metrics {

template <typename T> class Counter : public Metric, Accumulator<Counter<T>> {
public:
  ~Counter() = default;
  explicit Counter(std::string_view name) : Metric(name), counter_(0) {}
  Counter(std::string_view name, T counter) : Metric(name), counter_(counter) {}

  T GetCounter() const { return counter_; }

  // Assignment.
  Counter &operator=(T c);

  // Addition and subtraction.
  Counter &operator+=(T rhs);
  Counter &operator-=(T rhs);

  // Pre- and postfix increment.
  Counter &operator++();
  Counter operator++(int);

  // Pre- and postfix decrement.
  Counter &operator--();
  Counter operator--(int);

  // Equlity operator.
  bool operator==(T c) const;
  bool operator==(const Counter &c) const;

  // For printing.
  std::string ToString() override;

  MetricProto ToProto() override;

  Metrics ToMetrics() override;

  remus::util::Status Accumulate(const remus::util::StatusVal<Counter<T>> &other) override;

private:
  T counter_;
};

// ---------------|
// IMPLEMENTATION |
// ---------------|

template <typename T> Counter<T> &Counter<T>::operator=(T c) {
  counter_ = c;
  return *this;
}

template <typename T> Counter<T> &Counter<T>::operator+=(T rhs) {
  counter_ += rhs;
  return *this;
}

template <typename T> Counter<T> &Counter<T>::operator-=(T rhs) {
  counter_ -= rhs;
  return *this;
}

template <typename T> Counter<T> &Counter<T>::operator++() {
  ++counter_;
  return *this;
}

template <typename T> Counter<T> Counter<T>::operator++(int) {
  Counter old = *this;
  operator++();
  return old;
}

template <typename T> Counter<T> &Counter<T>::operator--() {
  --counter_;
  return *this;
}

template <typename T> Counter<T> Counter<T>::operator--(int) {
  Counter old = *this;
  operator--();
  return old;
}

template <typename T> bool Counter<T>::operator==(T c) const {
  return counter_ == c;
}

template <typename T> bool Counter<T>::operator==(const Counter<T> &c) const {
  return name_ == c.name_ && operator==(c.counter_);
}

template <typename T> std::string Counter<T>::ToString() {
  return "count: " + std::to_string(counter_) + "";
}

template <typename T> MetricProto Counter<T>::ToProto() {
  MetricProto proto;
  proto.set_name(name_);
  proto.mutable_counter()->set_count(counter_);
  return proto;
}

template <typename T> Metrics Counter<T>::ToMetrics() {
  Metrics result = Metrics(MetricType::Counter);
  result.name = name_;
  result.try_get_counter()->counter = counter_;
  return result;
}

template <typename T>
remus::util::Status Counter<T>::Accumulate(const remus::util::StatusVal<Counter<T>> &other) {
  if (other.status.t != remus::util::Ok)
    return other.status;
  if (!(name_ == other.val.value().name_)) {
    return {remus::util::FailedPrecondition,
            "Counter name does not match: " + other.val.value().name_};
  }
  operator+=(other.val.value().counter_);
  return remus::util::Status::Ok();
}

} // namespace remus::metrics
