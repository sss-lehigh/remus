#include <cassert>
#include <utility>
#include "../utility/annotations.h"

#pragma once

namespace hds::arch {

enum Architecture {
  CPU,
  PTX
};

HDS_HOST_DEVICE Architecture current_arch() {
#if defined(__CUDA_ARCH__)
  return PTX;
#else
  return CPU;
#endif
};

template<Architecture a>
struct Arch {
  static constexpr Architecture value = a;
};

template<Architecture a>
constexpr inline Architecture Arch_v = Arch<a>::value;

template<typename T, Architecture arch>
HDS_HOST_DEVICE_INLINE T load(const T* ptr, Arch<arch>) {
  return *ptr;
}

template<typename T, typename U, Architecture arch>
HDS_HOST_DEVICE_INLINE void store(T* ptr, U&& x, Arch<arch>) {
  *ptr = std::forward<U>(x);
}

enum class memory_order {
  memory_order_weak,
  memory_order_relaxed,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst
};

template<typename T, Architecture arch>
HDS_HOST_DEVICE_INLINE T atomic_load(T* ptr, memory_order order, Arch<arch>) {
  switch (order) {
    case memory_order::memory_order_weak:
      return load(ptr, Arch<arch>{});
    default:
      assert(false);
  }
}

template<typename T, typename U, Architecture arch>
HDS_HOST_DEVICE_INLINE void atomic_store(T* ptr, U&& x, memory_order order, Arch<arch>) {
  switch (order) {
    case memory_order::memory_order_weak:
      return store(ptr, std::forward<U>(x), Arch<arch>{});
    default:
      assert(false);
  }
}

template<typename T, typename U, typename V, Architecture arch>
HDS_HOST_DEVICE_INLINE T atomic_cas(T* ptr, U&& expected, V&& desired, memory_order order, Arch<arch>);

/// Non templated architecture

template<typename T>
HDS_HOST_DEVICE_INLINE T atomic_load(T* ptr, memory_order order, Architecture arch) {
  switch(arch) {
    case CPU:
      return atomic_load(ptr, order, Arch<Architecture::CPU>{});
    case PTX:
      return atomic_load(ptr, order, Arch<Architecture::PTX>{});
  }
}

template<typename T, typename U>
HDS_HOST_DEVICE_INLINE void atomic_store(T* ptr, U&& x, memory_order order, Architecture arch) {
  switch(arch) {
    case CPU:
      atomic_store(ptr, std::forward<U>(x), order, Arch<Architecture::CPU>{});
      break;
    case PTX:
      atomic_store(ptr, std::forward<U>(x), order, Arch<Architecture::PTX>{});
      break;
  }
}

template<typename T, typename U, typename V>
HDS_HOST_DEVICE_INLINE T atomic_cas(T* ptr, U&& expected, V&& desired, memory_order order, Architecture arch) {
  switch(arch) {
    case CPU:
      atomic_cas(ptr, std::forward<U>(expected), std::forward<V>(desired), order, Arch<Architecture::CPU>{});
      break;
    case PTX:
      atomic_cas(ptr, std::forward<U>(expected), std::forward<V>(desired), order, Arch<Architecture::PTX>{});
      break;
  }
}


}
