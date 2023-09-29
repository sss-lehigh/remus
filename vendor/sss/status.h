#pragma once

#include <optional>
#include <string>

namespace sss {
enum StatusType {
  Ok,
  InternalError,
  Unavailable,
  Cancelled,
  NotFound,
  Unknown,
  AlreadyExists,
  FailedPrecondition,
  InvalidArgument,
  ResourceExhausted,
  Aborted
};

struct Status {
  const StatusType t;
  const std::optional<std::string> message;
};

template <class T> struct StatusVal {
  const Status status;
  std::optional<T> val;
};
} // namespace sss