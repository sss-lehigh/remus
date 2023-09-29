#pragma once
#include <chrono>

#include "../testutil/fake_clock.h"

namespace util {

using SystemClock = std::chrono::system_clock;
using SteadyClock = std::chrono::steady_clock;

} // namespace util