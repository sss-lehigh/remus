#pragma once

#include <chrono>

// [mfs] This seems like not enough code to justify a file

namespace util {

using SystemClock = std::chrono::system_clock;
using SteadyClock = std::chrono::steady_clock;

} // namespace util