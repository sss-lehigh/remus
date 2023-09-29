#pragma once

#if defined(__clang__)
#include <experimental/memory_resource>
namespace util {
using namespace std::experimental;
#elif defined(__GNUC__) || defined(__GNUG__)
#include <memory_resource>
namespace util {
using namespace std;
#else
#error "Unknown compiler"
#endif
}