#include "annotations.h"
#include <atomic>

#if defined(GPU)
#include <cuda/std/atomic>
#endif

#pragma once

namespace rome::hds {
#if defined(GPU)

template<typename T>
using atomic = cuda::std::atomic<T>;

template<typename T>
using atomic_ref = cuda::std::atomic_ref<T>;

using memory_order = cuda::std::memory_order;

inline constexpr memory_order memory_order_relaxed = cuda::std::memory_order::relaxed;
inline constexpr memory_order memory_order_consume = cuda::std::memory_order::consume;
inline constexpr memory_order memory_order_acquire = cuda::std::memory_order::acquire;
inline constexpr memory_order memory_order_release = cuda::std::memory_order::release;
inline constexpr memory_order memory_order_acq_rel = cuda::std::memory_order::acq_rel;
inline constexpr memory_order memory_order_seq_cst = cuda::std::memory_order::seq_cst;

HDS_HOST_DEVICE void atomic_thread_fence(memory_order order) noexcept {
  ::cuda::std::atomic_thread_fence(order);
}

#else

template<typename T>
using atomic = std::atomic<T>;

template<typename T>
using atomic_ref = std::atomic_ref<T>;

using memory_order = std::memory_order;

inline constexpr memory_order memory_order_relaxed = std::memory_order::relaxed;
inline constexpr memory_order memory_order_consume = std::memory_order::consume;
inline constexpr memory_order memory_order_acquire = std::memory_order::acquire;
inline constexpr memory_order memory_order_release = std::memory_order::release;
inline constexpr memory_order memory_order_acq_rel = std::memory_order::acq_rel;
inline constexpr memory_order memory_order_seq_cst = std::memory_order::seq_cst;

HDS_HOST_DEVICE void atomic_thread_fence(memory_order order) noexcept {
  ::std::atomic_thread_fence(order);
}

#endif
}

