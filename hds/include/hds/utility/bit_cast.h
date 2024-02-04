#include <type_traits>

#pragma once

namespace hds {

template<typename To, typename From>
HDS_HOST_DEVICE constexpr
std::enable_if_t<sizeof(To) == sizeof(From), To> bit_cast(const From& src) noexcept {
#if defined(__CUDA_ARCH__)
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
#else
  return std::bit_cast<To>(src);
#endif
}

}


