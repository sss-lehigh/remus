#pragma once

#include <atomic>
#include <barrier>
#include <coroutine>
#include <exception>
#include <memory>

#include "../logging/logging.h"
#include "../util/coroutine.h"
#include "absl/status/status.h"

namespace util {

// Forward declaration necessary so that `from_promise()` is defined for our
// coroutine handle. There may be a cleaner way to accomplish this, but this how
// its done here, https://en.cppreference.com/w/cpp/language/coroutines.
class Promise;

// For our purposes, the return object of a coroutine is just a wrapper for the
// `promise_type`. Any coroutine that is used with the scheduler must return a
// `Task`.
class Coro : public std::coroutine_handle<Promise> {
public:
  using promise_type = Promise;
  using handler_type = std::coroutine_handle<Promise>;
};

// The promise object of a coroutine dictates behavior when the coroutine first
// starts, and when it returns. It also can save some state to be queried later.
class Promise {
public:
  Coro get_return_object() {
    return {std::coroutine_handle<Promise>::from_promise(*this)};
  }

  std::suspend_always initial_suspend() { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void unhandled_exception() {
    std::rethrow_exception(std::current_exception());
  }
  void return_void() {}
};

// The interface for all coroutine schedulers. A scheduler can add new
// coroutines, start running, and cancel running. So
template <typename PromiseT> class Scheduler {
public:
  // Adds a new coroutine to the runner to be run with a given policy.
  virtual void Schedule(std::coroutine_handle<PromiseT> task) = 0;

  // Starts running the coroutines until `Stop()` is called or some other
  // temination condition is reached.
  virtual void Run() = 0;

  // Cancels running the coroutines.
  virtual void Cancel() = 0;
};

using Cancelation = std::atomic<bool>;

template <typename PromiseT>
class RoundRobinScheduler : public Scheduler<PromiseT> {
public:
  ~RoundRobinScheduler() { ROME_TRACE("Task count: {}", task_count_); }
  RoundRobinScheduler() : task_count_(0), curr_(nullptr), canceled_(false) {}

  // Getters.
  int task_count() const { return task_count_; }

  // Inserts the given task as the next task to run.
  void Schedule(std::coroutine_handle<PromiseT> task) override {
    if (canceled_)
      return;
    auto coro = new CoroWrapper{task, nullptr, nullptr};
    if (curr_ == nullptr) {
      coro->prev = coro;
      coro->next = coro;
      curr_ = coro;
      last_ = curr_;
    } else {
      coro->prev = last_;
      coro->next = last_->next;
      last_->next->prev = coro;
      last_->next = coro;
      last_ = coro;
    }
    ++task_count_;
  }

  void Run() override {
    ROME_ASSERT(curr_ != nullptr,
                "You must schedule at least one task before running");
    while (curr_ != nullptr) {
      if (curr_->handle.done()) {
        if (curr_->next == curr_) {
          // Only one coroutine was left...
          --task_count_;
          delete curr_;
          last_ = nullptr;
          std::atomic_thread_fence(std::memory_order_release);
          curr_ = nullptr;
          continue;
        }

        // Unlink `curr_` by setting its predecessor's next to `curr_`'s next.
        // And setting the successors's prev to `curr_`'s prev. Finally, update
        // the tail of the list to point to `curr_`'s predecessor.
        curr_->prev->next = curr_->next;
        curr_->next->prev = curr_->prev;
        if (last_ == curr_) {
          last_ = curr_->prev;
        }

        // Cleanup the removed `Task` by deleting the `TaskWrapper`, which in
        // turn takes care of destroying the underlying coroutine handle.
        auto *temp = curr_;
        curr_ = curr_->next;
        --task_count_;
        delete temp;
      } else {
        curr_->handle.resume();
        curr_ = curr_->next;
      }
    }
  }

  // Cancels the scheduler and then waits for all currently scheduled tasks to
  // complete. Coroutines can obtain a pointer to the cancelation flag using
  // `Cancelation()` and then check if it has been canceled.
  void Cancel() override {
    canceled_ = true;
    while (curr_ != nullptr)
      ;
  }

  const Cancelation &Cancelation() const { return canceled_; }

private:
  struct CoroWrapper {
    ~CoroWrapper() { handle.destroy(); }
    std::coroutine_handle<PromiseT> handle;
    CoroWrapper *prev;
    CoroWrapper *next;
  };

  int task_count_;
  CoroWrapper *curr_;
  CoroWrapper *last_;
  std::atomic<bool> canceled_;
};

} // namespace util