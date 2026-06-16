#pragma once

#include <cstdio>
#include <format>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace remus {
/// [mfs] This doesn't need so much engineering.  Why isn't a Variant good
///       enough?  Does the type of error really matter?
enum StatusType {
  Ok,
  InternalError,
  Unavailable,
  Aborted,
};

/// [mfs] I'm wondering why this can't just be a variant... it's either OK or
/// Some(string)...
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

/// [mfs] I'm not convinced we need this in the workload folder.  It is also
///       used in the RDMA code, where we might be able to get by with a
///       std::variant.
template <class T> struct StatusVal {
  Status status;
  std::optional<T> val;
};
} // namespace remus

namespace remus {
#define RELEASE 0
#define DEBUG 1

// Make sure we have a log level, even if the build tools didn't define one
#ifndef REMUS_LOG_LEVEL
#warning "REMUS_LOG_LEVEL is not defined... defaulting to DEBUG"
#define REMUS_LOG_LEVEL DEBUG
#endif
#if REMUS_LOG_LEVEL != RELEASE && REMUS_LOG_LEVEL != DEBUG
#warning "Invalid value for REMUS_LOG_LEVEL.  Defaulting to DEBUG"
#define REMUS_LOG_LEVEL DEBUG
#endif

/// Print a message
///
/// @param msg The message to print
///
inline void print_debug(std::string_view msg, const char *file, uint32_t line) {
  // NB: for thread-safety, we use printf
  std::printf("[DEBUG] %.*s (%s:%u)\n", (int)msg.length(), msg.data(), file,
              line);
  std::fflush(stdout);
}
/// Print an info message
///
/// @param msg The message to print
inline void print_info(std::string_view msg) {
  // NB: for thread-safety, we use printf
  std::printf("[INFO] %.*s\n", (int)msg.length(), msg.data());
  std::fflush(stdout);
}
/// Print a fatal message
///
/// @param msg The message to print
inline void print_fatal(std::string_view msg) {
  // NB: for thread-safety, we use printf
  std::printf("[FATAL] %.*s\n", (int)msg.length(), msg.data());
  std::fflush(stdout);
}

/// Print a debug message only if REMUS_DEBUG_MSGS is defined
#if REMUS_LOG_LEVEL == DEBUG
#define REMUS_DEBUG(...)                                                       \
  remus::print_debug(std::format(__VA_ARGS__), __FILE__, __LINE__)
#else
#define REMUS_DEBUG(...)
#endif

/// Print an information message
#define REMUS_INFO(...) remus::print_info(std::format(__VA_ARGS__))

/// Terminate with a message on a fatal error
#define REMUS_FATAL(...)                                                       \
  {                                                                            \
    remus::print_fatal(std::format(__VA_ARGS__));                              \
    std::_Exit(1);                                                             \
  }

/// Assert, and print a fatal message if it fails
///
/// [mfs] ASSERT doesn't print a line number or file number.  We should add that
///       (see DEBUG).
#define REMUS_ASSERT(check, ...)                                               \
  if (!(check)) [[unlikely]] {                                                 \
    remus::print_fatal(std::format(__VA_ARGS__));                              \
    std::_Exit(1);                                                             \
  }

/// Terminate if status is not OK
#define OK_OR_FAIL(status)                                                     \
  if (auto __s = status; (__s.t != remus::util::Ok)) [[unlikely]] {            \
    REMUS_FATAL("{}", __s.message.value());                                    \
  }

/// Fail if func does not return 0
#define RDMA_CM_ASSERT(func, ...)                                              \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    REMUS_ASSERT(ret == 0, "{}{}{}", #func, "(): ", strerror(errno));          \
  }

inline void INIT() {
#if REMUS_LOG_LEVEL == DEBUG
  std::printf("REMUS::DEBUG is true\n");
#else
  std::printf("REMUS::DEBUG is false\n");
#endif
}
} // namespace remus
