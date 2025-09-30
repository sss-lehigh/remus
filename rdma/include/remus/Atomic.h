#pragma once

#include <atomic>

#include "compute_thread.h"

namespace remus {
/// @brief A class for atomic operations on a type T
///
/// @tparam T The type of value to be atomically manipulated
template <typename T> class Atomic {
private:
  T value;

public:
  /// TODO: Document this
  ///
  /// @param compute_thread
  /// @param fence
  /// @return
  T load(std::shared_ptr<typename remus::ComputeThread> compute_thread,
         bool fence = true) const {
    return compute_thread->Read(
        rdma_ptr<T>((uintptr_t)this + offsetof(Atomic, value)), fence);
  }

  /// TODO: Document this
  ///
  /// @param value
  /// @param compute_thread
  /// @param size
  /// @param fence
  void store(T value,
             std::shared_ptr<typename remus::ComputeThread> compute_thread,
             size_t size = sizeof(T), bool fence = true) {
    compute_thread->Write(
        rdma_ptr<T>((uintptr_t)this + offsetof(Atomic, value)), value, fence,
        size);
  }

  /// TODO: Document this
  ///
  /// @param expected
  /// @param desired
  /// @param compute_thread
  /// @param fence
  /// @return
  bool compare_exchange_weak(
      T expected, T desired,
      std::shared_ptr<typename remus::ComputeThread> compute_thread,
      bool fence = true) {
    auto res = compute_thread->CompareAndSwap(
        rdma_ptr<T>((uintptr_t)this + offsetof(Atomic, value)), expected,
        desired, fence);
    return res == expected;
  }

  /// TODO: Document this
  ///
  /// @param expected
  /// @param desired
  /// @param compute_thread
  /// @param fence
  /// @return
  bool compare_exchange_strong(
      T expected, T desired,
      std::shared_ptr<typename remus::ComputeThread> compute_thread,
      bool fence = true) {
    auto res = compute_thread->CompareAndSwap(
        rdma_ptr<T>((uintptr_t)this + offsetof(Atomic, value)), expected,
        desired, fence);
    return res == expected;
  }

  /// TODO: Document this
  ///
  /// @param value
  /// @param compute_thread
  /// @param signal
  /// @param fence
  /// @return
  T fetch_add(T value,
              std::shared_ptr<typename remus::ComputeThread> compute_thread,
              bool signal = true, bool fence = true) {
    return compute_thread->FetchAndAdd(
        rdma_ptr<T>((uintptr_t)this + offsetof(Atomic, value)), value, signal,
        fence);
  }
};
} // namespace remus