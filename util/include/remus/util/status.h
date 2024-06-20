#pragma once

#include <optional>
#include <sstream>
#include <string>

namespace remus::util {
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
  Aborted,
  OutOfRange,
  StreamTerminated
};

struct Status {
  StatusType t;
  std::optional<std::string> message;

  static Status Ok() { return {StatusType::Ok, {}}; }

  template <typename T> Status operator<<(T t) {
    std::string curr = message ? message.value() : "";
    std::stringstream s;
    s << curr;
    s << t;
    message = s.str();
    return *this;
  }
};

template <class T> struct StatusVal {
  Status status;
  std::optional<T> val;
};
} // namespace remus::util
