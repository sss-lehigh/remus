#pragma once

#include <deque>
#include <functional>
#include <string>
#include <thread>
#include <utility>

#include "../util/thread_util.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"

namespace util {

using Task = std::pair<std::string, std::function<void()>>;

class ThreadPool {
public:
  ~ThreadPool();
  ThreadPool();

  absl::Status Enqueue(Task task) LOCKS_EXCLUDED(mu_);

  void Stop();
  void Drain();

private:
  void RunThread();

  std::vector<std::thread> threads_;

  std::atomic<bool> drain_;
  std::atomic<bool> stopped_;

  // TODO: Replace with lockfree queue.
  std::mutex mu_;
  std::deque<Task> tasks_ GUARDED_BY(mu_);
};

} // namespace util