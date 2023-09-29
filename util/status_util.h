#pragma once

#include <sstream>
#include <string_view>

#include "absl/status/status.h"

#define VALUE_OR_DIE(status_or)                         \
  [&](const auto& __s) __attribute__((always_inline)) { \
    if (!(__s).ok()) {                                  \
      ROME_FATAL(__s.status().ToString());              \
    }                                                   \
    return __s.value();                                 \
  }                                                     \
  ((status_or))

namespace util {

template <absl::StatusCode ErrorCode>
class StatusBuilder {
 public:
  StatusBuilder() = default;
  StatusBuilder(const StatusBuilder& other) : ss_(other.ss_.str()) {}
  StatusBuilder(StatusBuilder&& other) : ss_(other.ss_.str()) {}

  template <typename T>
  StatusBuilder<ErrorCode>& operator<<(T t) {
    ss_ << t;
    return *this;
  }

  // Conversion function enabling the builder to be used exactly like an
  // `absl::Status` object.
  operator absl::Status() const { return absl::Status(ErrorCode, ss_.str()); }

 private:
  std::stringstream ss_;
};

using UnavailableErrorBuilder = StatusBuilder<absl::StatusCode::kUnavailable>;
using CancelledErrorBuilder = StatusBuilder<absl::StatusCode::kCancelled>;
using NotFoundErrorBuilder = StatusBuilder<absl::StatusCode::kNotFound>;
using UnknownErrorBuilder = StatusBuilder<absl::StatusCode::kUnknown>;
using AlreadyExistsErrorBuilder =
    StatusBuilder<absl::StatusCode::kAlreadyExists>;
using FailedPreconditionErrorBuilder =
    StatusBuilder<absl::StatusCode::kFailedPrecondition>;
using InternalErrorBuilder = StatusBuilder<absl::StatusCode::kInternal>;
using InvalidArgumentErrorBuilder =
    StatusBuilder<absl::StatusCode::kInvalidArgument>;
using ResourceExhaustedErrorBuilder =
    StatusBuilder<absl::StatusCode::kResourceExhausted>;
using AbortedErrorBuilder = StatusBuilder<absl::StatusCode::kAborted>;

}  // namespace util