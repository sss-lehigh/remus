#include <optional>
#if defined(GPU)
#include <cuda/std/optional>
#endif

#pragma once

namespace remus::hds {
#if defined(GPU)

template<typename T>
using optional = ::cuda::std::optional<T>;
using nullopt_t = ::cuda::std::nullopt_t;
constexpr nullopt_t nullopt = ::cuda::std::nullopt;

#else

template<typename T>
using optional = ::std::optional<T>;
using nullopt_t = ::std::nullopt_t;
constexpr nullopt_t nullopt = ::std::nullopt;

#endif
}

