#include <cassert>
#include <utility>
#include <atomic>

#include "arch.h"

#pragma once

namespace hds::arch {

template<typename T>
HDS_HOST_DEVICE_INLINE T atomic_load(const T* ptr, memory_order order, Arch<Architecture::CPU>) {
  switch (order) {
    case memory_order::memory_order_weak:
      return load(ptr, Arch<Architecture::CPU>{});
    case memory_order::memory_order_relaxed:
      return std::atomic_ref<T>(*ptr).load(std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return std::atomic_ref<T>(*ptr).load(std::memory_order_acquire);
    case memory_order::memory_order_release:
      return std::atomic_ref<T>(*ptr).load(std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return std::atomic_ref<T>(*ptr).load(std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return std::atomic_ref<T>(*ptr).load(std::memory_order_seq_cst);
    default:
      assert(false);
  }
}

template<typename T, typename U>
HDS_HOST_DEVICE_INLINE void atomic_store(T* ptr, U&& x, memory_order order, Arch<Architecture::CPU>) {
  switch (order) {
    case memory_order::memory_order_weak:
      return store(ptr, std::forward<U>(x), Arch<Architecture::CPU>{});
    case memory_order::memory_order_relaxed:
      return std::atomic_ref<T>(*ptr).store(std::forward<U>(x), std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return std::atomic_ref<T>(*ptr).store(std::forward<U>(x), std::memory_order_acquire);
    case memory_order::memory_order_release:
      return std::atomic_ref<T>(*ptr).store(std::forward<U>(x), std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return std::atomic_ref<T>(*ptr).store(std::forward<U>(x), std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return std::atomic_ref<T>(*ptr).store(std::forward<U>(x), std::memory_order_seq_cst);
    default:
      assert(false);
  }
}

template<typename T, typename U, typename V, Architecture arch>
HDS_HOST_DEVICE_INLINE void atomic_cas_val(T* ptr, U&& expected, V&& desired, memory_order order, Arch<Architecture::CPU>) {
  T tmp = std::forward<U>(expected);
  switch (order) {
    case memory_order::memory_order_relaxed:
      std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_relaxed);
      return tmp;
    case memory_order::memory_order_acquire:
      std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_acquire);
      return tmp;
    case memory_order::memory_order_release:
      std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_release);
      return tmp;
    case memory_order::memory_order_acq_rel:
      std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_acq_rel);
      return tmp;
    case memory_order::memory_order_seq_cst:
      std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_seq_cst);
      return tmp;
    default:
      assert(false);
  }
}

template<typename T, typename U, typename V, Architecture arch>
HDS_HOST_DEVICE_INLINE bool atomic_cas_bool(T* ptr, U&& expected, V&& desired, memory_order order, Arch<Architecture::CPU>) {
  T tmp = std::forward<U>(expected);
  switch (order) {
    case memory_order::memory_order_relaxed:
      return std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_acquire);
    case memory_order::memory_order_release:
      return std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), std::memory_order_seq_cst);
    default:
      assert(false);
  }
}

}
