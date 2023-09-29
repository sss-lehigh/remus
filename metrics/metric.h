#pragma once
#include <ostream>
#include <vector>

#include "absl/status/statusor.h"
#include "protos/metrics.pb.h"

namespace rome::metrics {

class Metric {
 public:
  Metric(std::string_view name) : name_(name) {}
  virtual ~Metric() = default;
  virtual std::string ToString() = 0;
  virtual MetricProto ToProto() = 0;
  friend std::ostream& operator<<(std::ostream& os, Metric& metric) {
    return os << "name: \"" << metric.name_ << "\", " << metric.ToString();
  };

 protected:
  const std::string name_;
};

template <typename T>
class Accumulator {
 public:
  virtual ~Accumulator() = default;
  virtual absl::Status Accumulate(const absl::StatusOr<T>& other) = 0;
};

}  // namespace rome::metrics