#pragma once

#include "metrics_object.h"
#include <ostream>
#include <vector>

#include "remus/util/status.h"
#include <protos/metrics.pb.h>

namespace remus::metrics {

class Metric {
public:
  Metric(std::string_view name) : name_(name) {}
  virtual ~Metric() = default;
  virtual std::string ToString() = 0;
  virtual MetricProto ToProto() = 0;
  virtual Metrics ToMetrics() = 0;
  friend std::ostream &operator<<(std::ostream &os, Metric &metric) {
    return os << "name: \"" << metric.name_ << "\", " << metric.ToString();
  };

protected:
  const std::string name_;
};

template <typename T> class Accumulator {
public:
  virtual ~Accumulator() = default;
  virtual remus::util::Status Accumulate(const remus::util::StatusVal<T> &other) = 0;
};

} // namespace remus::metrics
