#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <thread>

#include "../metrics/counter.h"
#include "../metrics/metric.h"
#include "../metrics/stopwatch.h"
#include "../metrics/summary.h"
#include "../util/duration_util.h"
#include "client_adaptor.h"
#include "protos/colosseum.pb.h"
#include "qps_controller.h"
#include "streams/stream.h"

namespace rome {

// For reference, a `WorkloadDriver` with a simple `MappedStream` (i.e.,
// consisting of one sub-stream) can achieve roughly 1M QPS. This was measured
// using a client adaptor that does nothing. As the number of constituent
// streams increases, we expect the maximum throughput to decrease but it is not
// likely to be the limiting factor in performance.
template <typename OpType> class WorkloadDriver {
public:
  ~WorkloadDriver();

  // Creates a new `WorkloadDriver` from the constiuent client adaptor and
  // stream. If not `nullptr`, the QPS controller is used to limit the
  // throughput of operations fed to the client.
  static std::unique_ptr<WorkloadDriver>
  Create(std::unique_ptr<ClientAdaptor<OpType>> client,
         std::unique_ptr<Stream<OpType>> stream, QpsController *qps_controller,
         std::optional<std::chrono::milliseconds> qps_sampling_rate =
             std::nullopt) {
    return std::unique_ptr<WorkloadDriver>(new WorkloadDriver(
        std::move(client), std::move(stream), qps_controller,
        qps_sampling_rate.value_or(std::chrono::milliseconds(0))));
  }

  // Calls the client's `Start` method before starting the workload driver,
  // returning its error if there is one. Operations are then pulled from
  // `stream_` and passed to `client_`'s `Apply` method. The client will handle
  // operations until either the given stream is exhausted or `Stop` is called.
  sss::Status Start();

  // Stops the workload driver so no new requests are passed to the client.
  // Then, the client's `Stop` method is called so that any pending operations
  // can be finalized.
  sss::Status Stop();

  metrics::Stopwatch *GetStopwatch() { return stopwatch_.get(); }

  std::string ToString() {
    std::stringstream ss;
    ss << ops_ << std::endl;
    ss << lat_summary_ << std::endl;
    ss << qps_summary_ << std::endl;
    ss << *stopwatch_;
    return ss.str();
  }

  WorkloadDriverProto ToProto() {
    WorkloadDriverProto proto;
    proto.mutable_ops()->CopyFrom(ops_.ToProto());
    proto.mutable_runtime()->CopyFrom(stopwatch_->ToProto());
    proto.mutable_qps()->CopyFrom(qps_summary_.ToProto());
    proto.mutable_latency()->CopyFrom(lat_summary_.ToProto());
    return proto;
  }

private:
  sss::Status Run();

  WorkloadDriver(std::unique_ptr<ClientAdaptor<OpType>> client,
                 std::unique_ptr<Stream<OpType>> stream,
                 QpsController *qps_controller,
                 std::chrono::milliseconds qps_sampling_rate)
      : terminated_(false), running_(false), client_(std::move(client)),
        stream_(std::move(stream)), qps_controller_(qps_controller),
        ops_("total_ops"), stopwatch_(nullptr), prev_ops_(0),
        qps_sampling_rate_(qps_sampling_rate),
        qps_summary_("sampled_qps", "ops/s", 1000), lat_sampling_rate_(10),
        lat_summary_("sampled_lat", "ns", 1000) {}

  std::atomic<bool> terminated_;
  std::atomic<bool> running_;

  std::unique_ptr<ClientAdaptor<OpType>> client_;
  std::unique_ptr<Stream<OpType>> stream_;

  //! Not owned.
  QpsController *qps_controller_;

  metrics::Counter<uint64_t> ops_;
  std::unique_ptr<metrics::Stopwatch> stopwatch_;

  uint64_t prev_ops_;
  std::chrono::milliseconds qps_sampling_rate_;
  metrics::Summary<double> qps_summary_;

  std::chrono::milliseconds lat_sampling_rate_;
  metrics::Summary<double> lat_summary_;

  std::future<sss::Status> run_status_;
  std::unique_ptr<std::thread> run_thread_;
};

// ---------------|
// Implementation |
// ---------------|

template <typename OpType> WorkloadDriver<OpType>::~WorkloadDriver() {
  terminated_ = true;
  run_thread_->join();
}

template <typename OpType> sss::Status WorkloadDriver<OpType>::Start() {
  if (terminated_) {
    return {sss::Unavailable, "Cannot restart a terminated workload driver."};
  }

  auto task =
      std::packaged_task<sss::Status()>(std::bind(&WorkloadDriver::Run, this));
  run_status_ = task.get_future();
  run_thread_ = std::make_unique<std::thread>(std::move(task));
  while (!running_)
    ;
  return sss::Status::Ok();
}

template <typename OpType> sss::Status WorkloadDriver<OpType>::Stop() {
  ROME_INFO("Stopping Workload Driver...");
  if (terminated_) {
    return {sss::Unavailable, "Workload driver was already terminated"};
  }

  // Signal `run_thread_` to stop. After joining the thread, it is guaranteed
  // that no new operations are passed to the client for handling.
  terminated_ = true;
  run_status_.wait();

  if (run_status_.get().t != sss::Ok)
    return run_status_.get();
  return sss::Status::Ok();
}

template <typename OpType> sss::Status WorkloadDriver<OpType>::Run() {
  auto status = client_->Start();
  if (status.t != sss::Ok)
    return status;
  stopwatch_ = metrics::Stopwatch::Create("driver_stopwatch");
  running_ = true;

  while (!terminated_) {
    if (qps_controller_ != nullptr) {
      qps_controller_->Wait();
    }

    auto next_op = stream_->Next();
    if (next_op.status.t != sss::Ok) {
      if (!IsStreamTerminated(next_op.status)) {
        status = next_op.status;
      }
      break;
    }

    auto curr_lap = stopwatch_->GetLapSplit();
    auto curr_lap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        curr_lap.GetRuntimeNanoseconds());

    auto client_status = client_->Apply(next_op.val.value());
    if (curr_lap_ms > lat_sampling_rate_) {
      lat_summary_
          << (stopwatch_->GetLapSplit().GetRuntimeNanoseconds().count() -
              curr_lap.GetRuntimeNanoseconds().count());
    }

    if (client_status.t != sss::Ok) {
      status = client_status;
      break;
    }

    ++ops_;

    if (curr_lap_ms > qps_sampling_rate_) {
      auto curr_ops = ops_.GetCounter();
      auto sample =
          (curr_ops - prev_ops_) /
          (stopwatch_->GetLap().GetRuntimeNanoseconds().count() * 1e-9);
      qps_summary_ << sample;
      prev_ops_ = curr_ops;
    }
  }
  // The client's `Stop` may block while there are any outstanding operations.
  // After this call, it is assumed that the client is no longer active.
  status = client_->Stop();
  stopwatch_->Stop();
  return status;
}

} // namespace rome