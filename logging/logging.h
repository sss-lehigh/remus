#pragma once

#include "../vendor/sss/status.h"

#define TRACE 0
#define DEBUG 1
#define INFO 2
#define WARN 3
#define ERROR 4
#define CRITICAL 5
#define OFF 6

// [mfs]  This gets us a guaranteed log level, even if the build tools didn't
//        define one.
#ifndef ROME_LOG_LEVEL
#warning "ROME_LOG_LEVEL is not defined... defaulting to TRACE"
#define ROME_LOG_LEVEL TRACE
#endif

// [mfs]  The *entire* spdlog infrastructure seems to be in use only for the
//        sake of getting coloring for messages on stdout.  I think we can
//        remove it without really losing anything.

//! Must be set before including `spdlog/spdlog.h`
#define SPDLOG_ACTIVE_LEVEL ROME_LOG_LEVEL

#include <memory>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

inline void __rome_init_log__() {
  auto __rome_log__ = ::spdlog::get("rome");
  if (!__rome_log__) {
#if defined(ROME_ASYNC_LOG)
    ::spdlog::init_thread_pool(8192, 1);
    __rome_log__ =
        ::spdlog::create_async<::spdlog::sinks::stdout_color_sink_mt>("rome");
#else
    __rome_log__ =
        ::spdlog::create<::spdlog::sinks::stdout_color_sink_mt>("rome");
#endif
  }
  static_assert(ROME_LOG_LEVEL < spdlog::level::level_enum::n_levels,
                "Invalid logging level.");
  __rome_log__->set_level(
      static_cast<spdlog::level::level_enum>(ROME_LOG_LEVEL));
  __rome_log__->set_pattern("[%Y-%m-%d %H:%M%S thread:%t] [%^%l%$] [%@] %v");
  ::spdlog::set_default_logger(std::move(__rome_log__));
  SPDLOG_INFO("Logging level: {}", ::spdlog::level::to_string_view(
                                       ::spdlog::default_logger()->level()));
}

#if ROME_LOG_LEVEL == OFF
#define ROME_INIT_LOG []() {}
#else
#define ROME_INIT_LOG __rome_init_log__
#endif

#if ROME_LOG_LEVEL == OFF
#define ROME_DEINIT_LOG() ((void)0)
#else
#define ROME_DEINIT_LOG [&]() { ::spdlog::drop("rome"); }
#endif

#if ROME_LOG_LEVEL <= TRACE
#define ROME_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#else
#define ROME_TRACE(...) ((void)0)
#endif

#if ROME_LOG_LEVEL <= DEBUG
#define ROME_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#else
#define ROME_DEBUG(...) ((void)0)
#endif

#if ROME_LOG_LEVEL <= INFO
#define ROME_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#else
#define ROME_INFO(...) ((void)0)
#endif

#if ROME_LOG_LEVEL <= WARN
#define ROME_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#else
#define ROME_WARN(...) ((void)0)
#endif

#if ROME_LOG_LEVEL <= ERROR
#define ROME_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#else
#define ROME_ERROR(...) ((void)0)
#endif

#if ROME_LOG_LEVEL <= CRITICAL
#define ROME_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
#else
#define ROME_CRITICAL(...) ((void)0)
#endif

// [mfs]  This looks like it is incorrect.  LOG_LEVEL==OFF will lead to an
//        undefined symbol, when we probably still want a call to
//        std::terminate()?
#if ROME_LOG_LEVEL != OFF
#define ROME_FATAL(...)                                                        \
  SPDLOG_CRITICAL(__VA_ARGS__);                                                \
  std::terminate();
#endif

#define ROME_ASSERT(check, ...)                                                \
  if (!(check)) [[unlikely]] {                                                 \
    SPDLOG_CRITICAL(__VA_ARGS__);                                              \
    std::terminate();                                                          \
  }

#define OK_OR_FAIL(status)                                                     \
  if (auto __s = status; !(__s.t == sss::Ok)) [[unlikely]] {                   \
    SPDLOG_CRITICAL(__s.message.value());                                      \
    std::terminate();                                                          \
  }

#define RETURN_STATUS_ON_ERROR(status)                                         \
  if (!(status.t == sss::Ok)) [[unlikely]] {                                   \
    SPDLOG_ERROR(status.message.value());                                      \
    return status;                                                             \
  }

#define RETURN_STATUSVAL_ON_ERROR(sv)                                          \
  if (!(sv.status.t == sss::Ok)) [[unlikely]] {                                \
    SPDLOG_ERROR(sv.status.message.value());                                   \
    return sv.status;                                                          \
  }

// Specific checks for debugging. Can be turned off by commenting out
// `ROME_DO_DEBUG_CHECKS` in config.h
#ifndef ROME_NDEBUG
#define ROME_ASSERT_DEBUG(func, ...)                                           \
  if (!(func)) [[unlikely]] {                                                  \
    SPDLOG_ERROR(__VA_ARGS__);                                                 \
    std::terminate();                                                          \
  }
#else
#define ROME_ASSERT_DEBUG(...) ((void)0)
#endif

#define STATUSVAL_OR_DIE(__s)                                                  \
  if (!(__s.status.t == sss::Ok)) {                                            \
    ROME_FATAL(__s.status.message.value());                                    \
  }

#define RDMA_CM_CHECK(func, ...)                                               \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    if (ret != 0) {                                                            \
      sss::Status err = {sss::InternalError, ""};                              \
      err << #func << "(): " << strerror(errno);                               \
      return err;                                                              \
    }                                                                          \
  }

#define RDMA_CM_CHECK_TOVAL(func, ...)                                         \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    if (ret != 0) {                                                            \
      sss::Status err = {sss::InternalError, ""};                              \
      err << #func << "(): " << strerror(errno);                               \
      return {err, {}};                                                        \
    }                                                                          \
  }

#define RDMA_CM_ASSERT(func, ...)                                              \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    ROME_ASSERT(ret == 0, "{}{}{}", #func, "(): ", strerror(errno));           \
  }
