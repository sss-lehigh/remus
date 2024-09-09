#pragma once

#include "remus/util/status.h"

#define TRACE 0
#define DEBUG 1
#define INFO 2
#define WARN 3
#define ERROR 4
#define CRITICAL 5
#define OFF 6

// [mfs]  This gets us a guaranteed log level, even if the build tools didn't
//        define one.
#ifndef REMUS_LOG_LEVEL
  #warning "REMUS_LOG_LEVEL is not defined... defaulting to TRACE"
  #define REMUS_LOG_LEVEL TRACE
#endif

// [mfs]  The *entire* spdlog infrastructure seems to be in use only for the
//        sake of getting coloring for messages on stdout.  I think we can
//        remove it without really losing anything.

//! Must be set before including `spdlog/spdlog.h`
#define SPDLOG_ACTIVE_LEVEL REMUS_LOG_LEVEL

#include <memory>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

inline void __remus_init_log__() {
  auto __remus_log__ = ::spdlog::get("remus");
  if (!__remus_log__) {
#if defined(REMUS_ASYNC_LOG)
    ::spdlog::init_thread_pool(8192, 1);
    __remus_log__ = ::spdlog::create_async<::spdlog::sinks::stdout_color_sink_mt>("remus");
#else
    __remus_log__ = ::spdlog::create<::spdlog::sinks::stdout_color_sink_mt>("remus");
#endif
  }
  static_assert(REMUS_LOG_LEVEL < spdlog::level::level_enum::n_levels, "Invalid logging level.");
  __remus_log__->set_level(static_cast<spdlog::level::level_enum>(REMUS_LOG_LEVEL));
  __remus_log__->set_pattern("[%Y-%m-%d %H:%M%S thread:%t] [%^%l%$] [%@] %v");
  ::spdlog::set_default_logger(std::move(__remus_log__));
  SPDLOG_INFO("Logging level: {}", ::spdlog::level::to_string_view(::spdlog::default_logger()->level()));
}

#if REMUS_LOG_LEVEL == OFF
  #define REMUS_INIT_LOG []() {}
#else
  #define REMUS_INIT_LOG __remus_init_log__
#endif

#if REMUS_LOG_LEVEL == OFF
  #define REMUS_DEINIT_LOG() ((void)0)
#else
  #define REMUS_DEINIT_LOG [&]() { ::spdlog::drop("remus"); }
#endif

#if REMUS_LOG_LEVEL <= TRACE
  #define REMUS_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#else
  #define REMUS_TRACE(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL <= DEBUG
  #define REMUS_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#else
  #define REMUS_DEBUG(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL <= INFO
  #define REMUS_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#else
  #define REMUS_INFO(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL <= WARN
  #define REMUS_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#else
  #define REMUS_WARN(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL <= ERROR
  #define REMUS_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#else
  #define REMUS_ERROR(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL <= CRITICAL
  #define REMUS_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
#else
  #define REMUS_CRITICAL(...) ((void)0)
#endif

#if REMUS_LOG_LEVEL != OFF
  #define REMUS_FATAL(...)                                                                                             \
    {                                                                                                                  \
      SPDLOG_CRITICAL(__VA_ARGS__);                                                                                    \
      std::_Exit(1);                                                                                                   \
    }
#endif

#if REMUS_LOG_LEVEL == OFF
  #define REMUS_FATAL(...)                                                                                             \
    {                                                                                                                  \
      std::_Exit(1);                                                                                                   \
    }
#endif

#define REMUS_ASSERT(check, ...)                                                                                       \
  if (!(check)) [[unlikely]] {                                                                                         \
    SPDLOG_CRITICAL(__VA_ARGS__);                                                                                      \
    std::_Exit(1);                                                                                                  \
  }

#define OK_OR_FAIL(status)                                                                                             \
  if (auto __s = status; !(__s.t == remus::util::Ok)) [[unlikely]] {                                                   \
    SPDLOG_CRITICAL(__s.message.value());                                                                              \
    std::_Exit(1);                                                                                                  \
  }

#define RETURN_STATUS_ON_ERROR(status)                                                                                 \
  if (!(status.t == remus::util::Ok)) [[unlikely]] {                                                                   \
    SPDLOG_ERROR(status.message.value());                                                                              \
    return status;                                                                                                     \
  }

#define RETURN_STATUSVAL_ON_ERROR(sv)                                                                                  \
  if (!(sv.status.t == remus::util::Ok)) [[unlikely]] {                                                                \
    SPDLOG_ERROR(sv.status.message.value());                                                                           \
    return sv.status;                                                                                                  \
  }

// Specific checks for debugging. Can be turned off by commenting out
// `REMUS_DO_DEBUG_CHECKS` in config.h
#ifndef REMUS_NDEBUG
  #define REMUS_ASSERT_DEBUG(func, ...)                                                                                \
    if (!(func)) [[unlikely]] {                                                                                        \
      SPDLOG_ERROR(__VA_ARGS__);                                                                                       \
      std::_Exit(1);                                                                                                \
    }
#else
  #define REMUS_ASSERT_DEBUG(...) ((void)0)
#endif

#define STATUSVAL_OR_DIE(__s)                                                                                          \
  if (!(__s.status.t == remus::util::Ok)) {                                                                            \
    REMUS_FATAL(__s.status.message.value());                                                                           \
  }

#define RDMA_CM_CHECK(func, ...)                                                                                       \
  {                                                                                                                    \
    int ret = func(__VA_ARGS__);                                                                                       \
    if (ret != 0) {                                                                                                    \
      remus::util::Status err = {remus::util::InternalError, ""};                                                      \
      err << #func << "(): " << strerror(errno);                                                                       \
      return err;                                                                                                      \
    }                                                                                                                  \
  }

#define RDMA_CM_CHECK_TOVAL(func, ...)                                                                                 \
  {                                                                                                                    \
    int ret = func(__VA_ARGS__);                                                                                       \
    if (ret != 0) {                                                                                                    \
      remus::util::Status err = {remus::util::InternalError, ""};                                                      \
      err << #func << "(): " << strerror(errno);                                                                       \
      return {err, {}};                                                                                                \
    }                                                                                                                  \
  }

#define RDMA_CM_ASSERT(func, ...)                                                                                      \
  {                                                                                                                    \
    int ret = func(__VA_ARGS__);                                                                                       \
    REMUS_ASSERT(ret == 0, "{}{}{}", #func, "(): ", strerror(errno));                                                  \
  }
