#pragma once
#include <chrono>

namespace util {

namespace internal {

// Converts a `std::duration` of `InputType` to `OutputType` and returns a
// double representation.
template <typename InputType, typename OutputType>
inline double ToDoubleInternal(const InputType &in) {
  return double(std::chrono::duration_cast<OutputType>(in).count());
}

} // namespace internal

template <typename DurationType>
inline double ToDoubleNanoseconds(const DurationType &in) {
  return internal::ToDoubleInternal<DurationType, std::chrono::nanoseconds>(in);
}

template <typename DurationType>
inline double ToDoubleMicroseconds(const DurationType &in) {
  return internal::ToDoubleInternal<DurationType, std::chrono::microseconds>(
      in);
}

template <typename DurationType>
inline double ToDoubleMilliseconds(const DurationType &in) {
  return internal::ToDoubleInternal<DurationType, std::chrono::milliseconds>(
      in);
}

template <typename DurationType>
inline double ToDoubleSeconds(const DurationType &in) {
  return internal::ToDoubleInternal<DurationType, std::chrono::seconds>(in);
}

} // namespace rome::timing