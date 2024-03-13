#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <nlohmann/json.hpp>

#include "remus/logging/logging.h"

using namespace std;
using namespace nlohmann;

namespace remus::metrics {

class CounterMetric {
public:
  uint64_t counter;
  
  /// Serialize to a raw json object
  json as_json(){
    json j;
    j["counter"] = counter;
    return j;
  }
  
  /// Serialize to a string
  std::string serialize(){
    return as_json().dump(2);
  }

  /// Deserialize a string to fill in the fields
  void deserialize(std::string data){
    json j = json::parse(data);
    counter = j["counter"];
  }
};

class StopwatchMetric {
public:
  uint64_t runtime_ns;

  /// Serialize to a raw json object
  json as_json(){
    json j;
    j["runtime_ns"] = runtime_ns;
    return j;
  }

  /// Serialize to a string
  std::string serialize(){
    return as_json().dump(2);
  }

  /// Deserialize a string to fill in the fields
  void deserialize(std::string data){
    json j = json::parse(data);
    runtime_ns = j["runtime_ns"];
  }
};

class SummaryMetric {
public:
    string units;
    double mean;
    double stddev;
    double min;
    double p50;
    double p90;
    double p95;
    double p99;
    double p999;
    double max;
    uint64_t count;

    SummaryMetric() = default;

    /// Serialize to a raw json object
    json as_json(){
      json j;
      j["units"] = units;
      j["mean"] = mean;
      j["stddev"] = stddev;
      j["min"] = min;
      j["p50"] = p50;
      j["p90"] = p90;
      j["p95"] = p95;
      j["p99"] = p99;
      j["p999"] = p999;
      j["max"] = max;
      j["count"] = count;
      return j;
    }

    /// Serialize to a string
    std::string serialize(){
      return as_json().dump(2);
    }

    /// Deserialize a string to fill in the fields
    void deserialize(std::string data){
      json j = json::parse(data);
      units = j["units"];
      mean = j["mean"];
      stddev = j["stddev"];
      min = j["min"];
      p50 = j["p50"];
      p90 = j["p90"];
      p95 = j["p95"];
      p99 = j["p99"];
      p999 = j["p999"];
      max = j["max"];
      count = j["count"];
    }

};

namespace MetricType {
  enum MetricType {
    Counter, Stopwatch, Summary
  };

  std::string stringify(MetricType t){
    switch(t){
      case Counter:
        return "Counter";
      case Stopwatch:
        return "Stopwatch";
      case Summary:
        return "Summary";
      default:
        return "Unknown";
    }
  }
}

/// A oneof Summary, Counter, and Stopwatch
class Metrics {
private:
  // the type of the data
  MetricType::MetricType type;
  // a pointer to the object
  shared_ptr<void> metric_data;
public:
  /// the name of the metric
  string name;

  /// Construct a metric with a certain type
  Metrics(MetricType::MetricType type){
    this->type = type;
    if (type == MetricType::Summary)
      metric_data = make_shared<SummaryMetric>();
    else if (type == MetricType::Counter)
      metric_data = make_shared<CounterMetric>();
    else if (type == MetricType::Stopwatch)
      metric_data = make_shared<StopwatchMetric>();
  }

  /// If we are a summary metric
  bool has_summary(){
    return type == MetricType::Summary;
  }

  /// If we are a stopwatch metric
  bool has_stopwatch(){
    return type == MetricType::Stopwatch;
  }

  /// If we are a counter metric
  bool has_counter(){
    return type == MetricType::Counter;
  }

  /// Try to unwrap a summary
  /// Will return a nullptr if the metric is not a summary
  shared_ptr<SummaryMetric> try_get_summary(){
    if (type != MetricType::Summary) return nullptr;
    return static_pointer_cast<SummaryMetric>(metric_data);
  }

  /// Try to unwrap a stopwatch
  /// Will return a nullptr if the metric is not a stopwatch
  shared_ptr<StopwatchMetric> try_get_stopwatch(){
    if (type != MetricType::Stopwatch) return nullptr;
    return static_pointer_cast<StopwatchMetric>(metric_data);
  }

  /// Try to unwrap a counter
  /// Will return a nullptr if the metric is not a counter
  shared_ptr<CounterMetric> try_get_counter(){
    if (type != MetricType::Counter) return nullptr;
    return static_pointer_cast<CounterMetric>(metric_data);
  }

  /// Serialize to a raw json object
  json as_json(){
    json j;
    j["name"] = name;
    j["type"] = type;
    if (type == MetricType::Summary)
      j["data"] = try_get_summary()->as_json();
    else if (type == MetricType::Counter)
      j["data"] = try_get_counter()->as_json();
    else if (type == MetricType::Stopwatch)
      j["data"] = try_get_stopwatch()->as_json();
    return j;
  }

  /// Serialize to a string
  std::string serialize(){
    return as_json().dump(2);
  }

  /// Deserialize a string to fill in the fields
  void deserialize(std::string data){
    json j = json::parse(data);
    name = j["name"];
    if (type != j["type"]){
      ROME_WARN("Overriding type in deserialize old={} new={}", MetricType::stringify(type), MetricType::stringify(j["type"]));
    }
    type = j["type"];
    if (type == MetricType::Summary){
      shared_ptr<SummaryMetric> metric_data = make_shared<SummaryMetric>();
      metric_data->deserialize(j["data"].dump());
      this->metric_data = metric_data;
    } else if (type == MetricType::Counter){
      shared_ptr<CounterMetric> metric_data = make_shared<CounterMetric>();
      metric_data->deserialize(j["data"].dump());
      this->metric_data = metric_data;
    } else if (type == MetricType::Stopwatch){
      shared_ptr<StopwatchMetric> metric_data = make_shared<StopwatchMetric>();
      metric_data->deserialize(j["data"].dump());
      this->metric_data = metric_data;
    }
  }
};

}
