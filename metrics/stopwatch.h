#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include "metric.h"

namespace rome::metrics {

// All of these are KHz
inline static constexpr char kTscFreqKhzFilePath[] =
    "/sys/devices/system/cpu/cpu0/tsc_freq_khz";
inline static constexpr char kMaxFreqFilePath[] =
    "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency";
inline static constexpr int kDefaultCpuFreqKhz = 2300000;

class Stopwatch : public Metric {
public:
  // A split is used to measure a period of time. The period starts when the
  // `Split` is first created and completes whenever a call to `Stop` is made.
  // The purpose of having a specific object to measure each period is to enable
  // monitoring more complex control flows. A `Split` object can be moved
  //
  class Split {
  public:
    Split(uint32_t tsc_freq_khz, uint64_t start);
    Split(uint32_t tsc_freq_khz, uint64_t start, uint64_t end);

    // Default move and copy.
    Split(const Split &) = default;
    Split(Split &&other) = default;

    // Returns the length of the split in nanoseconds.
    std::chrono::nanoseconds GetRuntimeNanoseconds();

  private:
    uint32_t tsc_freq_khz_;
    uint64_t start_;
    uint64_t end_;
  };

  // Initializes the tsc frequency from either a known file or a compile time
  // variable. Then, starts the stopwatch running by initializing the internal
  // `split_`, which tracks total runtime.
  static std::unique_ptr<Stopwatch> Create(std::string_view name);

  // Returns a new `Split` object that is used to measure a portion of the
  // overall runtime.
  Split GetSplit();

  Split GetLap();

  Split GetLapSplit();

  // Stops the stopwatch. Future calls to `Split` will still generate new splits
  // that can be used for timing, but the internal split maintained by the
  // stopwatch is fixed.
  void Stop();

  // Returns the total runtime measured by the stopwatch. The return value is
  // only valid after `Stop` has been called.
  std::chrono::nanoseconds GetRuntimeNanoseconds();

  std::string ToString() override;

  MetricProto ToProto() override;

private:
  Stopwatch(std::string_view name, uint64_t tsc_freq_khz);

  uint64_t tsc_freq_khz_;
  uint64_t start_;
  uint64_t end_;
  uint64_t lap_;
};

} // namespace rome::metrics