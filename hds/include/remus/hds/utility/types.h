#include <cstdint>
#include <type_traits>

#pragma once

namespace remus::hds {

template <int N> using Int = std::integral_constant<int, N>;

template <unsigned N> using Uint = std::integral_constant<unsigned, N>;

template <uint8_t N> using Uint8 = std::integral_constant<uint8_t, N>;

template <bool B> using Bool = std::integral_constant<bool, B>;

inline constexpr Bool<false> _false = Bool<false>{};
inline constexpr Bool<true> _true = Bool<true>{};

} // namespace remus::hds
