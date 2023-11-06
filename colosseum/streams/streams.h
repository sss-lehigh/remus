#pragma once
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <valarray>
#include <vector>

#include "../../logging/logging.h"
#include "../../util/distribution_util.h"
#include "stream.h"

namespace rome {

template <typename T> class TestStream : public Stream<T> {
public:
  TestStream(const std::vector<T> &input)
      : output_(input), iter_(output_.begin()) {}

private:
  sss::StatusVal<T> NextInternal() override {
    auto curr = iter_;
    if (curr == output_.end()) {
      return {StreamTerminatedStatus(), {}};
    }
    iter_++; // Only advance `iter_` if not at the end.
    return {sss::Status::Ok(), *curr};
  }
  std::vector<T> output_;
  typename std::vector<T>::iterator iter_;
};

// [mfs] This is unused?
template <typename DistributionType, typename... Args>
class RandomDistributionStream
    : public Stream<typename DistributionType::result_type> {
public:
  RandomDistributionStream(const Args &...args)
      : mt_(rd_()), distribution_(std::forward<decltype(args)>(args)...) {}

private:
  sss::StatusVal<typename DistributionType::result_type>
  NextInternal() override {
    return distribution_(mt_);
  }
  std::random_device rd_;
  std::mt19937 mt_;
  DistributionType distribution_;
};

// [mfs] This is unused?
template <typename... Args>
using UniformDoubleStream =
    RandomDistributionStream<std::uniform_real_distribution<double>, Args...>;

// [mfs] This is unused?
template <typename... Args>
using UniformIntStream =
    RandomDistributionStream<std::uniform_int_distribution<int>, Args...>;

// A `WeightedStream` randomly picks values from a given enum `E` with a
// distribution following the provided weights. The index of each weight in the
// `weights` vector maps to the corresponding enum value. We use a simple
// algorithm in which we insert the enum value into `output_` a number of times
// equivalent to its weight. Then, when sampling we sample uniformly into the
// `output_` and return the enum value.
//
// [mfs] This is unused?
template <typename E> class WeightedStream : public Stream<E> {
  static_assert(std::is_enum<E>::value);

public:
  WeightedStream(const std::vector<uint32_t> &weights)
      : distribution_(0,
                      std::accumulate(weights.begin(), weights.end(), 0) - 1) {
    for (uint32_t i = 0; i < weights.size(); ++i) {
      for (int j = 0; j < weights[i]; ++j) {
        output_.push_back(E(i));
      }
    }
  }

private:
  sss::StatusVal<E> NextInternal() override {
    auto next = distribution_(mt_);
    return output_[next];
  }

  std::mt19937 mt_;
  std::uniform_int_distribution<uint32_t> distribution_;
  std::vector<E> output_;
};

// [mfs] This is unused?
template <typename T> class MonotonicStream : public Stream<T> {
public:
  MonotonicStream(const T &init, const T &step) : step_(step), value_(init) {}

private:
  inline sss::StatusVal<T> NextInternal() override { return value_ += step_; }

  const T step_;
  T value_;
};

// [mfs] This is unused?
template <typename T> class CircularStream : public Stream<T> {
public:
  CircularStream(const T &start, const T &end, const T &step)
      : step_(step), start_(start), end_(end), curr_() {}

private:
  inline sss::StatusVal<T> NextInternal() override {
    auto temp = curr_;
    curr_ += step_;
    return (temp % end_) + start_;
  }

  const T step_;
  const T start_;
  const T end_;
  T curr_;
};

// [mfs] This is unused?
template <typename T, typename... U> class MappedStream : public Stream<T> {
public:
  static std::unique_ptr<MappedStream>
  Create(std::function<sss::StatusVal<T>(const std::unique_ptr<Stream<U>> &...)>
             generator,
         std::unique_ptr<Stream<U>>... streams) {
    return std::unique_ptr<MappedStream>(
        new MappedStream(std::move(generator), std::move(streams)...));
  }

protected:
  MappedStream(
      std::function<sss::StatusVal<T>(const std::unique_ptr<Stream<U>> &...)>
          generator,
      std::unique_ptr<Stream<U>>... streams)
      : generator_(generator), streams_({std::move(streams)...}) {}

  sss::StatusVal<T> NextInternal() override {
    return std::apply(generator_, streams_);
  }

  std::function<sss::StatusVal<T>(const std::unique_ptr<Stream<U>> &...)>
      generator_;
  std::tuple<std::unique_ptr<Stream<U>>...> streams_;
};

// [mfs] This is unused?
struct NoOp {};
class NoOpStream : public Stream<NoOp> {
public:
private:
  sss::StatusVal<NoOp> NextInternal() { return {sss::Status::Ok(), NoOp{}}; }
};

template <typename T> class EndlessStream : public Stream<T> {
public:
  EndlessStream(std::function<T(void)> generator) : generator_(generator) {}

private:
  std::function<T(void)> generator_;
  inline sss::StatusVal<T> NextInternal() override {
    return {sss::Status::Ok(), generator_()};
  }
};

template <typename T> class FixedLengthStream : public Stream<T> {
public:
  FixedLengthStream(std::function<T(void)> generator, int length) : generator_(generator), length_(length), count_(0) {}

private:
  std::function<T(void)> generator_;
  int length_;
  int count_;
  inline sss::StatusVal<T> NextInternal() override {
    count_++;
    if (length_ < count_){
      return {StreamTerminatedStatus(), {}};
    }
    return {sss::Status::Ok(), generator_()};
  }
};

} // namespace rome