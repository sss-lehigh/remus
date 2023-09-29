#pragma once
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace rome {

inline absl::Status StreamTerminatedStatus() {
  return absl::OutOfRangeError("Stream terminated.");
}

inline bool IsStreamTerminated(const absl::Status &status) {
  return status.code() == absl::StatusCode::kOutOfRange &&
         status.message() == "Stream terminated.";
}

// Represents a stream of input for benchmarking a system. The common use is to
// define the template parameter to be some struct that contains the necessary
// information for a given operation. Streams can represent an infinite sequence
// of random numbers, sequential values, or can be backed by a trace file. By
// encapsulating any input in a stream, workload driver code can abstact out the
// work of generating values.
//
// Calling `Next` on a stream of numbers returns the next value in the stream,
// or `StreamTerminatedStatus`. If a user calls `Terminate` then all future
// calls to next will produce `StreamTerminatedStatus`.
template <typename T>
class Stream {
 public:
  Stream() : terminated_(false) {}
  virtual ~Stream() = default;

  // Movable but not copyable.
  Stream(const Stream &) = delete;
  Stream(Stream &&) = default;

  absl::StatusOr<T> Next() __attribute__((always_inline)) {
    if (!terminated_) {
      return NextInternal();
    } else {
      return StreamTerminatedStatus();
    }
  }

  void Terminate() { terminated_ = true; }

 private:
  virtual absl::StatusOr<T> NextInternal() __attribute__((always_inline)) = 0;

  bool terminated_;
};

}  // namespace rome