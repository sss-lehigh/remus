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

template <typename T> class EndlessStream : public Stream<T> {
public:
  EndlessStream(std::function<T(void)> generator) : generator_(generator) {}

private:
  std::function<T(void)> generator_;
  inline sss::StatusVal<T> NextInternal() override {
    return {sss::Status::Ok(), generator_()};
  }
};

} // namespace rome