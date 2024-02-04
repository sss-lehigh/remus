#include <cassert>
#include <utility>
#include <cuda/std/atomic>

#include "arch.h"

#pragma once

namespace hds::arch {

template<typename T>
HDS_HOST_DEVICE_INLINE T atomic_load(T* ptr, memory_order order, Arch<Architecture::CPU>) {
  switch (order) {
    case memory_order::memory_order_weak:
      return load(ptr, Arch<Architecture::CPU>{});
    case memory_order::memory_order_relaxed:
      return cuda::std::atomic_ref<T>(*ptr).load(cuda::std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return cuda::std::atomic_ref<T>(*ptr).load(cuda::std::memory_order_acquire);
    case memory_order::memory_order_release:
      return cuda::std::atomic_ref<T>(*ptr).load(cuda::std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return cuda::std::atomic_ref<T>(*ptr).load(cuda::std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return cuda::std::atomic_ref<T>(*ptr).load(cuda::std::memory_order_seq_cst);
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
      return cuda::std::atomic_ref<T>(*ptr).store(std::forward<U>(x), cuda::std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return cuda::std::atomic_ref<T>(*ptr).store(std::forward<U>(x), cuda::std::memory_order_acquire);
    case memory_order::memory_order_release:
      return cuda::std::atomic_ref<T>(*ptr).store(std::forward<U>(x), cuda::std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return cuda::std::atomic_ref<T>(*ptr).store(std::forward<U>(x), cuda::std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return cuda::std::atomic_ref<T>(*ptr).store(std::forward<U>(x), cuda::std::memory_order_seq_cst);
    default:
      assert(false);
  }
}

template<typename T, typename U, typename V, Architecture arch>
HDS_HOST_DEVICE_INLINE T atomic_cas_val(T* ptr, U&& expected, V&& desired, memory_order order, Arch<Architecture::CPU>) {
  T tmp = std::forward<U>(expected);
  switch (order) {
    case memory_order::memory_order_relaxed:
      cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_relaxed);
      return tmp;
    case memory_order::memory_order_acquire:
      cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_acquire);
      return tmp;
    case memory_order::memory_order_release:
      cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_release);
      return tmp;
    case memory_order::memory_order_acq_rel:
      cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_acq_rel);
      return tmp;
    case memory_order::memory_order_seq_cst:
      cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_seq_cst);
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
      return cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_relaxed);
    case memory_order::memory_order_acquire:
      return cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_acquire);
    case memory_order::memory_order_release:
      return cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_release);
    case memory_order::memory_order_acq_rel:
      return cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_acq_rel);
    case memory_order::memory_order_seq_cst:
      return cuda::std::atomic_ref<T>(*ptr).compare_exchange_strong(tmp, std::forward<V>(desired), cuda::std::memory_order_seq_cst);
    default:
      assert(false);
  }
}

}
