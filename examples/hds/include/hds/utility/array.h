#include <array>
#if defined(GPU)
#include <cuda/std/array>
#endif

namespace hds {
#if defined(GPU)

template<typename T, std::size_t N>
using array = cuda::std::array<T, N>;

#else

template<typename T, std::size_t N>
using array = std::array<T, N>;

#endif
}

