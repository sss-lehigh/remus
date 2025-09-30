#pragma once

#include <coroutine>
#include <exception>
#include <optional>

namespace remus {

/// Define the abort decision enum at namespace level
struct CoroutineState {
  bool ready;
};

/// @brief A simple async result type that can be used to co_yield a value
///
/// @tparam T The type of value that this AsyncResult will hold
template <typename T>
struct AsyncResult {
  /// @brief The promise_type for AsyncResult
  /// @details
  /// Since we co_yield a AsyncResult, C++ requires that it have a promise_type
  struct promise_type {
    T val;  // return value,
    CoroutineState state = {false};

    /// Ensure that co_yield suspends the coroutine and makes the result
    /// available to the caller
    std::suspend_always yield_value(std::suspend_always) { return {}; }

    /// Ensure that co_return makes the result available to the caller
    void return_value(const auto &value) {
      val = value;
      state.ready = true;
    }

    /// Produce the "future"... a AsyncResult that enables resuming the
    /// coroutine
    auto get_return_object() {
      return AsyncResult{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    /// Don't suspend when the coroutine starts
    auto initial_suspend() { return std::suspend_never{}; }

    /// We're not really interested in exceptions, so we'll terminate on an
    /// unhandled exception
    void unhandled_exception() { std::terminate(); }

    /// Suspend on a coroutine return.  This means someone else needs to call
    /// destroy on the handle... it won't be done automatically.
    auto final_suspend() noexcept { return std::suspend_always{}; }
  };

 private:
  std::coroutine_handle<promise_type> handle;  // Save the coroutine handle

 public:
  /// Construct by copying the handle
  AsyncResult(auto h) : handle(h) {}

  /// Destruct by destroying the handle
  ~AsyncResult() {
    if (handle) handle.destroy();
  }

  /// Forbid copying AsyncResults
  AsyncResult(const AsyncResult &) = delete;

  /// Forbid assigning AsyncResults
  AsyncResult &operator=(const AsyncResult &) = delete;

  /// It's OK to move a AsyncResult
  AsyncResult(AsyncResult &&c) noexcept : handle{std::move(c.handle)} {
    c.handle = nullptr;
  }

  /// It's OK to move-assign a AsyncResult
  AsyncResult &operator=(AsyncResult &&c) noexcept {
    // If it's a new handle that we're copying in, destroy whatever we have
    // locally and ensure that `c` gives up responsibility for destroying the
    // handle
    if (this != &c) {
      if (handle) handle.destroy();
      handle = std::move(c.handle);
      c.handle = nullptr;
    }
    return *this;
  }

  /// Resume execution of the coroutine, but only if it isn't done.  We use this
  /// to do all coroutine cleanups at the end of a ccds operation.
  ///
  /// NB: We don't need the result of handle for now
  void resume() { handle.resume(); }

  /// Unpack the return value from this AsyncResult
  T get_value() { return handle.promise().val; }

  bool get_ready() const { return handle.promise().state.ready; }
};

/// @brief A simple async result type that does **not** co_yield a value
struct AsyncResultVoid {
  /// @brief The promise_type for AsyncResultVoid
  /// @details
  /// Since we co_yield a AsyncResultVoid, C++ requires that it have a
  /// promise_type
  struct promise_type {
    CoroutineState state = {false};
    /// Ensure that co_yield suspends the coroutine and makes the result
    /// available to the caller
    std::suspend_always yield_value(std::suspend_always) { return {}; }

    /// Ensure that co_return makes the result available to the caller
    void return_void() { state.ready = true; }

    /// Produce the "future"... a AsyncResult that enables resuming the
    /// coroutine
    auto get_return_object() {
      return AsyncResultVoid{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    /// Don't suspend when the coroutine starts
    auto initial_suspend() { return std::suspend_never{}; }

    /// We're not really interested in exceptions, so we'll terminate on an
    /// unhandled exception
    void unhandled_exception() { std::terminate(); }

    /// Suspend on a coroutine return.  This means someone else needs to call
    /// destroy on the handle... it won't be done automatically.
    auto final_suspend() noexcept { return std::suspend_always{}; }
  };

 private:
  std::coroutine_handle<promise_type> handle;  // Save the coroutine handle

 public:
  /// Construct by copying the handle
  AsyncResultVoid(auto h) : handle(h) {}

  /// Destruct by destroying the handle
  ~AsyncResultVoid() {
    if (handle) handle.destroy();
  }

  /// Forbid copying AsyncResults
  AsyncResultVoid(const AsyncResultVoid &) = delete;

  /// Forbid assigning AsyncResults
  AsyncResultVoid &operator=(const AsyncResultVoid &) = delete;

  /// It's OK to move a AsyncResult
  AsyncResultVoid(AsyncResultVoid &&c) noexcept : handle{std::move(c.handle)} {
    c.handle = nullptr;
  }

  /// It's OK to move-assign a AsyncResult
  AsyncResultVoid &operator=(AsyncResultVoid &&c) noexcept {
    // If it's a new handle that we're copying in, destroy whatever we have
    // locally and ensure that `c` gives up responsibility for destroying the
    // handle
    if (this != &c) {
      if (handle) handle.destroy();
      handle = std::move(c.handle);
      c.handle = nullptr;
    }
    return *this;
  }

  /// Resume execution of the coroutine, but only if it isn't done.  We use this
  /// to do all coroutine cleanups at the end of a ccds operation.
  ///
  /// NB: We don't need the result of handle for now
  void resume() { handle.resume(); }

  bool get_ready() const { return handle.promise().state.ready; }
};

}  // namespace remus
