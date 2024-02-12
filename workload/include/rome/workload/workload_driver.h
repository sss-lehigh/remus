/**
 * This file effectively reduces all of colosseum to three pieces:
 * - A concept that describes what a `Client` ought to look like.
 * - A `Stream` type.  This is essentially a generator of values to use in
 *   experiments
 * - A `WorkloadDriver` type.  It kicks off a *single* client and monitors it.
 *   This could be simplified quite a bit.
 */

#pragma once

#include <future>
#include <protos/workloaddriver.pb.h>

#include "rome/metrics/counter.h"
#include "rome/metrics/stopwatch.h"
#include "rome/metrics/summary.h"

namespace rome {

// A wrapper for clients, which are effectively the interface to some system to
// be tested. It can wrap a data structure or it can wrap a client in a
// distributed system. The point is that it is an abstract way to represent
// applying an operation produced by a `Stream` and the entity performing the
// operation.
//
// As an example, consider a data structure with a simple set API (i.e., Set,
// Get, and Delete). An operation in this scenario would be some struct defining
// the operation type and the target key.
//
// [mfs]  It seems like this should be a concept, to avoid virtual dispatch
//        overheads.  Or are there workloads with different ClientAdapters at
//        the same time?
//
// [mfs]  I can't yet do static_assert() that the role_client.h <==>
//        IsClientAdapter... Keep working on it?
template <template <typename> typename ClientAdapter, typename Operation>
concept IsClientAdapter =
    requires(ClientAdapter<Operation> c, const Operation o, bool b) {
      { c.Start() } -> std::same_as<util::Status>;
      { c.Apply(o) } -> std::same_as<util::Status>;
      { c.Stop() } -> std::same_as<util::Status>;
      { c.Operations(b) } -> std::same_as<util::Status>;
    };

namespace {
inline util::Status StreamTerminatedStatus() {
  return {util::StreamTerminated, "Stream terminated"};
}

inline bool IsStreamTerminated(const util::Status &status) {
  return status.t == util::StreamTerminated;
}
} // namespace

// Represents a stream of input for benchmarking a system. The common use is
// to define the template parameter to be some struct that contains the
// necessary information for a given operation. Streams can represent an
// infinite sequence of random numbers, sequential values, or can be backed by
// a trace file. By encapsulating any input in a stream, workload driver code
// can abstract out the work of generating values.
//
// Calling `Next` on a stream of numbers returns the next value in the stream,
// or `StreamTerminatedStatus`. If a user calls `Terminate` then all future
// calls to next will produce `StreamTerminatedStatus`.
//
// [mfs] Rename "InfiniteStream" or something like that?
template <typename T> class Stream {
public:
  Stream() : terminated_(false) {}
  virtual ~Stream() = default;

  // Movable but not copyable.
  Stream(const Stream &) = delete;
  Stream(Stream &&) = default;

  util::StatusVal<T> Next() __attribute__((always_inline)) {
    if (!terminated_) {
      return NextInternal();
    } else {
      return {StreamTerminatedStatus(), {}};
    }
  }

  void Terminate() { terminated_ = true; }

private:
  virtual util::StatusVal<T> NextInternal() __attribute__((always_inline)) = 0;

  bool terminated_;
};

template <typename T> class TestStream : public Stream<T> {
public:
  TestStream(const std::vector<T> &input)
      : output_(input), iter_(output_.begin()) {}

private:
  util::StatusVal<T> NextInternal() override {
    auto curr = iter_;
    if (curr == output_.end()) {
      return {StreamTerminatedStatus(), {}};
    }
    iter_++; // Only advance `iter_` if not at the end.
    return {util::Status::Ok(), *curr};
  }
  std::vector<T> output_;
  typename std::vector<T>::iterator iter_;
};

template <typename T> class EndlessStream : public Stream<T> {
public:
  EndlessStream(std::function<T(void)> generator) : generator_(generator) {}

private:
  std::function<T(void)> generator_;
  inline util::StatusVal<T> NextInternal() override {
    return {util::Status::Ok(), generator_()};
  }
};

template <typename T> class FixedLengthStream : public Stream<T> {
public:
  FixedLengthStream(std::function<T(void)> generator, int length)
      : generator_(generator), length_(length), count_(0) {}

private:
  std::function<T(void)> generator_;
  int length_;
  int count_;
  inline util::StatusVal<T> NextInternal() override {
    count_++;
    if (length_ < count_) {
      return {StreamTerminatedStatus(), {}};
    }
    return {util::Status::Ok(), generator_()};
  }
};

/// A simple workload driver
///
/// This workload driver
// For reference, a `WorkloadDriver` with a simple `MappedStream` (i.e.,
// consisting of one sub-stream) can achieve roughly 1M QPS. This was measured
// using a client adapter that does nothing. As the number of constituent
// streams increases, we expect the maximum throughput to decrease but it is
// not likely to be the limiting factor in performance.
template <template <typename> typename ClientAdapter, typename OpType>
class WorkloadDriver {

  std::atomic<bool> terminated_;
  std::atomic<bool> running_;

  std::unique_ptr<ClientAdapter<OpType>> client_;
  std::unique_ptr<Stream<OpType>> stream_;

  metrics::Counter<uint64_t> ops_;
  std::unique_ptr<metrics::Stopwatch> stopwatch_;

  uint64_t prev_ops_;
  std::chrono::milliseconds qps_sampling_rate_;
  metrics::Summary<double> qps_summary_;

  std::chrono::milliseconds lat_sampling_rate_;
  metrics::Summary<double> lat_summary_;

  std::future<util::Status> run_status_;
  std::unique_ptr<std::thread> run_thread_;

  WorkloadDriver(std::unique_ptr<ClientAdapter<OpType>> client,
                 std::unique_ptr<Stream<OpType>> stream,
                 std::chrono::milliseconds qps_sampling_rate)
      : terminated_(false), running_(false), client_(std::move(client)),
        stream_(std::move(stream)), ops_("total_ops"), stopwatch_(nullptr),
        prev_ops_(0), qps_sampling_rate_(qps_sampling_rate),
        qps_summary_("sampled_qps", "ops/s", 1000), lat_sampling_rate_(10),
        lat_summary_("sampled_lat", "ns", 1000) {}

public:
  ~WorkloadDriver() {
    terminated_ = true;
    run_thread_->join();
  }

  // Creates a new `WorkloadDriver` from the constituent client adapter and
  // stream
  static std::unique_ptr<WorkloadDriver>
  Create(std::unique_ptr<ClientAdapter<OpType>> client,
         std::unique_ptr<Stream<OpType>> stream,
         std::optional<std::chrono::milliseconds> qps_sampling_rate =
             std::nullopt) {
    return std::unique_ptr<WorkloadDriver>(new WorkloadDriver(
        std::move(client), std::move(stream),
        qps_sampling_rate.value_or(std::chrono::milliseconds(0))));
  }

  // Calls the client's `Start` method before starting the workload driver,
  // returning its error if there is one. Operations are then pulled from
  // `stream_` and passed to `client_`'s `Apply` method. The client will
  // handle operations until either the given stream is exhausted or `Stop` is
  // called.
  util::Status Start() {
    if (terminated_) {
      return {util::Unavailable, "Cannot restart a terminated workload driver."};
    }

    auto task = std::packaged_task<util::Status()>(
        std::bind(&WorkloadDriver::Run, this));
    run_status_ = task.get_future();
    run_thread_ = std::make_unique<std::thread>(std::move(task));
    while (!running_)
      ;
    return util::Status::Ok();
  }

  // Stops the workload driver so no new requests are passed to the client.
  // Then, the client's `Stop` method is called so that any pending operations
  // can be finalized.
  util::Status Stop() {
    ROME_INFO("Stopping Workload Driver...");
    if (terminated_) {
      return {util::Unavailable, "Workload driver was already terminated"};
    }

    // Signal `run_thread_` to stop. After joining the thread, it is guaranteed
    // that no new operations are passed to the client for handling.
    terminated_ = true;
    run_status_.wait();

    if (run_status_.get().t != util::Ok)
      return run_status_.get();
    return util::Status::Ok();
  }

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
  util::Status Run() {
    auto status = client_->Start();
    if (status.t != util::Ok)
      return status;
    stopwatch_ = metrics::Stopwatch::Create("driver_stopwatch");
    running_ = true;

    while (!terminated_) {
      auto next_op = stream_->Next();
      if (next_op.status.t != util::Ok) {
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

      if (client_status.t != util::Ok) {
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
};

} // namespace rome
