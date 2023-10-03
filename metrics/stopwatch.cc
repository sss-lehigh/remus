#include "stopwatch.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string_view>

#include "../logging/logging.h"

// [mfs] These assembly instructions are becoming full function calls.  Consider
// consolidating into the .h file?

namespace rome::metrics {

namespace {

inline uint64_t RdtscpAcquire() {
  uint32_t lo, hi;
  asm volatile("MFENCE\n\t"
               "RDTSCP\n\t"
               : "=a"(lo), "=d"(hi)::"%ecx");
  return ((static_cast<uint64_t>(hi) << 32) + lo);
}

inline uint64_t RdtscpRelease() {
  uint32_t lo, hi;
  asm volatile("RDTSCP\n\t"
               "LFENCE\n\t"
               : "=a"(lo), "=d"(hi)::"%ecx");
  return ((static_cast<uint64_t>(hi) << 32) + lo);
}

} // namespace

Stopwatch::Stopwatch(std::string_view name, uint64_t tsc_freq_khz)
    : Metric(name), tsc_freq_khz_(tsc_freq_khz), start_(RdtscpAcquire()),
      end_(0), lap_(start_) {}

/*static*/ std::unique_ptr<Stopwatch> Stopwatch::Create(std::string_view name) {
  // Attempt to read the TSC frequency from the tsc_freq_khz file first, then
  // try to read the maximum cpu frequency. The latter is not perfect for modern
  // Intel machines but it is better than nothing. If there is no max frequency
  // then use whatever is provided at compile time.
  uint64_t tsc_freq_khz;
  std::ifstream file;
  file.open(kTscFreqKhzFilePath);
  if (file.is_open()) {
    ROME_TRACE("Loading tsc_freq from tsc_freq_khz");
    file >> tsc_freq_khz;
  } else {
    file.clear();
    file.open(kMaxFreqFilePath);
    if (file.is_open()) {
      ROME_TRACE("Loading tsc_freq from max_cpu_freq");
      file >> tsc_freq_khz;
    } else {
      ROME_WARN(
          "Could not determine CPU frequency. Using compile time value: {} KHz "
          "[RESULTS MAY BE INACCURATE]",
          kDefaultCpuFreqKhz);
      tsc_freq_khz = kDefaultCpuFreqKhz;
    }
  }
  ROME_DEBUG("Using tsc_freq: {}", tsc_freq_khz);
  return std::unique_ptr<Stopwatch>(new Stopwatch(name, tsc_freq_khz));
}

Stopwatch::Split::Split(uint32_t tsc_freq_khz, uint64_t start, uint64_t end)
    : tsc_freq_khz_(tsc_freq_khz), start_(start), end_(end) {}

Stopwatch::Split::Split(uint32_t tsc_freq_khz, uint64_t start)
    : Split(tsc_freq_khz, start, RdtscpAcquire()) {}

namespace {
constexpr double KhzToGhz(int khz) { return double(khz) / 1e6; }
} // namespace

// # cycles / Ghz = # cycles / (cycles / nanosecond) = nanoseconds
std::chrono::nanoseconds Stopwatch::Split::GetRuntimeNanoseconds() {
  return std::chrono::nanoseconds(
      uint64_t((end_ - start_) / KhzToGhz(tsc_freq_khz_)));
}

Stopwatch::Split Stopwatch::GetSplit() { return Split(tsc_freq_khz_, start_); }

Stopwatch::Split Stopwatch::GetLap() {
  auto split = Split(tsc_freq_khz_, lap_);
  lap_ = RdtscpAcquire();
  return split;
}

Stopwatch::Split Stopwatch::GetLapSplit() { return Split(tsc_freq_khz_, lap_); }

void Stopwatch::Stop() { end_ = RdtscpRelease(); }

std::chrono::nanoseconds Stopwatch::GetRuntimeNanoseconds() {
  return Split(tsc_freq_khz_, start_, end_).GetRuntimeNanoseconds();
}

std::string Stopwatch::ToString() {
  std::stringstream ss;
  ss << "runtime: " << GetRuntimeNanoseconds().count() << " ns";
  return ss.str();
}

MetricProto Stopwatch::ToProto() {
  MetricProto proto;
  proto.set_name(name_);
  proto.mutable_stopwatch()->set_runtime_ns(GetRuntimeNanoseconds().count());
  return proto;
}

} // namespace rome::metrics