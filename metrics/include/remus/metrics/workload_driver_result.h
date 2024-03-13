#pragma once

#include <nlohmann/json.hpp>

#include "metrics_object.h"

namespace remus::metrics {

/// Create a result from the client adaptor
class WorkloadDriverResult {
public:
    Metrics ops{Metrics(MetricType::Counter)};
    Metrics runtime{Metrics(MetricType::Stopwatch)};
    Metrics qps{Metrics(MetricType::Summary)};
    Metrics latency{Metrics(MetricType::Summary)};

    /// Serialize to a raw json object
    json as_json(){
        json j;
        j["ops"] = ops.as_json();
        j["runtime"] = runtime.as_json();
        j["qps"] = qps.as_json();
        j["latency"] = latency.as_json();
        return j;
    }

    /// Serialize to a string
    std::string serialize(){
        return as_json().dump(2);
    }

    /// Deserialize a string to fill in the fields
    void deserialize(std::string data){
        json j = json::parse(data);
        ops.deserialize(j["ops"].dump());
        runtime.deserialize(j["runtime"].dump());
        qps.deserialize(j["qps"].dump());
        latency.deserialize(j["latency"].dump());
    }
};
}
