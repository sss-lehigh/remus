#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

namespace testutil {

class FakeClock {
 public:
  typedef std::chrono::nanoseconds duration;
  typedef duration::rep rep;
  typedef duration::period period;
  typedef std::chrono::time_point<FakeClock, duration> time_point;

  static constexpr bool is_steady = false;

  static time_point now() { return curr_time_; }

  static std::atomic<time_point> curr_time_;
  static void init() { curr_time_ = time_point(duration(0)); }
  static void advance() {
    time_point now, next;
    do {
      now = curr_time_.load();
      next = now + duration(1);
    } while (!curr_time_.compare_exchange_strong(now, next));
  }
  template <typename DurationType>
  static void advance(const DurationType &by) {
    time_point now, next;
    do {
      now = curr_time_.load();
      next = now + std::chrono::duration_cast<duration>(by);
    } while (!curr_time_.compare_exchange_strong(now, next));
  }
};

// Initializes `FakeClock`'s current time to be 0.
inline std::atomic<FakeClock::time_point> FakeClock::curr_time_(
    FakeClock::time_point(FakeClock::duration(0)));

}  // namespace testutil