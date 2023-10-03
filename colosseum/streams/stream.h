#pragma once

#include "../../vendor/sss/status.h"

namespace rome {

// [mfs] I don't understand why we wouldn't just use a StreamTerminated type?

inline sss::Status StreamTerminatedStatus() {
  return {sss::OutOfRange, "Stream terminated."};
}

inline bool IsStreamTerminated(const sss::Status &status) {
  return status.t == sss::OutOfRange &&
         status.message.value() == "Stream terminated.";
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
//
// [mfs] Rename "InfiniteStream" or something like that?
template <typename T> class Stream {
public:
  Stream() : terminated_(false) {}
  virtual ~Stream() = default;

  // Movable but not copyable.
  Stream(const Stream &) = delete;
  Stream(Stream &&) = default;

  sss::StatusVal<T> Next() __attribute__((always_inline)) {
    if (!terminated_) {
      return NextInternal();
    } else {
      return {StreamTerminatedStatus(), {}};
    }
  }

  void Terminate() { terminated_ = true; }

private:
  virtual sss::StatusVal<T> NextInternal() __attribute__((always_inline)) = 0;

  bool terminated_;
};

} // namespace rome