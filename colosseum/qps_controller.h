#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <mutex>
#include <numeric>

#include "../util/clocks.h"
#include "../util/duration_util.h"

// [mfs] This needs more documentation

namespace rome {

// A `QpsController` is responsible for modulating the number of operations per
// second that is produced by a given workload. Controlled components call
// `Wait` before every operation and are stalled appropriately to keep the QPS
// at the expected rate.
class QpsController {
public:
  virtual ~QpsController() = default;
  virtual void Wait() = 0;
};

template <typename Clock>
class LeakyTokenBucketQpsController : public QpsController {
public:
  static std::unique_ptr<LeakyTokenBucketQpsController>
  Create(int64_t max_qps) {
    return std::unique_ptr<LeakyTokenBucketQpsController>(
        new LeakyTokenBucketQpsController(max_qps));
  }

  void Wait() /*LOCKS_EXCLUDED(mu_)*/ override {
    std::lock_guard<std::mutex> lock(mu_);
    do {
      TryRefreshTokens();
    } while (tokens_ == 0);
    --tokens_;
    return;
  }

protected:
  explicit LeakyTokenBucketQpsController(int64_t max_qps)
      : max_qps_(max_qps), tokens_(max_qps) {
    last_refill_ = Clock::now();
  }

  void TryRefreshTokens() /*EXCLUSIVE_LOCKS_REQUIRED(mu_)*/ {
    auto now = Clock::now();
    int64_t deposit =
        std::chrono::duration_cast<std::chrono::seconds>(now - last_refill_)
            .count() *
        max_qps_;
    if (deposit > 0) {
      tokens_ = std::max(tokens_ + deposit, max_qps_);
      last_refill_ = now;
    }
  }

  std::mutex mu_;
  int64_t max_qps_ /*GUARDED_BY(mu_)*/;
  int64_t tokens_ /*GUARDED_BY(mu_)*/;
  std::chrono::time_point<Clock> last_refill_ /*GUARDED_BY(mu_)*/;
};

template <typename Clock>
class CyclingLeakyTokenBucketQpsController
    : public LeakyTokenBucketQpsController<Clock> {
public:
  void Wait() override /*EXCLUSIVE_LOCKS_REQUIRED(this->mu_)*/ {
    std::lock_guard<std::mutex> lock(this->mu_);
    do {
      TryUpdateQps();
      this->TryRefreshTokens();
    } while (this->tokens_ == 0);
    --this->tokens_;
    return;
  }

private:
  CyclingLeakyTokenBucketQpsController(int64_t min_qps, int64_t max_qps)
      : LeakyTokenBucketQpsController<Clock>(max_qps), curr_qps_(min_qps) {}

  void TryUpdateQps() {}

  int64_t curr_qps_;
};

} // namespace rome