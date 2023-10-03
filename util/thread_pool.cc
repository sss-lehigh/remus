#include <chrono>

#include "../logging/logging.h"

#include "thread_pool.h"

namespace util {

namespace {
constexpr std::chrono::nanoseconds kInitialBackoff(100);
constexpr std::chrono::nanoseconds kMaxBackoff(10000000);
} // namespace

ThreadPool::~ThreadPool() {
  for (auto &t : threads_) {
    t.join();
  }
}

ThreadPool::ThreadPool() : drain_(false), stopped_(false) {
  ROME_INFO("Creating thread pool (size={})",
            std::thread::hardware_concurrency());
  for (uint32_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
    threads_.emplace_back([this]() { RunThread(); });
  }
}

sss::Status ThreadPool::Enqueue(Task task) /*LOCKS_EXCLUDED(mu_)*/ {
  if (drain_) {
    return {sss::Unavailable, "Thread pool is draining"};
  }
  std::lock_guard<std::mutex> lock(mu_);
  tasks_.push_back(task);
  return sss::Status::Ok();
}

void ThreadPool::Stop() {
  if (!stopped_)
    stopped_ = true;
}

void ThreadPool::Drain() {
  while (!tasks_.empty())
    ;
  Stop();
}

void ThreadPool::RunThread() /*LOCKS_EXCLUDED(mu_)*/ {
  auto backoff = kInitialBackoff;
  bool tasks_empty;
  Task task;
  while (!stopped_) {
    {
      // Anonymous scope so the lock is released before execution.
      std::lock_guard<std::mutex> lock(mu_);
      tasks_empty = tasks_.empty();
      if (!tasks_empty) {
        backoff = kInitialBackoff;
        task = tasks_.front();
        tasks_.pop_front();
      }
    }

    if (tasks_empty) {
      std::this_thread::sleep_for(backoff);
      backoff = std::min(backoff * 2, kMaxBackoff);
      continue;
    } else {
      task.second();
    }
  }
}

} // namespace util