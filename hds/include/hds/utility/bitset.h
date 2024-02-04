#include <bit>
#include <concepts>
#include <cstdint>
#include <cassert>

#include "annotations.h"
#include "types.h"

#pragma once

namespace hds::utility {

template<int N>  
class uint8_bitset {
public:

  static constexpr int bits = N;

  static_assert(N <= 8);

  HDS_HOST_DEVICE constexpr uint8_bitset(std::integral auto x) : data(static_cast<uint8_t>(x)) {}

  constexpr uint8_bitset() = default;
  constexpr uint8_bitset(const uint8_bitset&) = default;
  constexpr uint8_bitset(uint8_bitset&&) = default;
  constexpr uint8_bitset& operator=(const uint8_bitset&) = default;
  constexpr uint8_bitset& operator=(uint8_bitset&&) = default;

  /**
   * set bit to x
   */
  HDS_HOST_DEVICE constexpr void set(uint8_t bit, bool x = true) {
    if (x == true) {
      data |= (0x1 << bit);
    } else {
      data &= (~(0x1 << bit));
    }
  }

  /**
   * set bit to x
   */
  template<bool x>
  HDS_HOST_DEVICE constexpr void set(uint8_t bit, std::integral_constant<bool, x>) {
    if constexpr (x == true) {
      data |= (0x1 << bit);
    } else {
      data &= (~(0x1 << bit));
    }
  }

  /// get bit x
  HDS_HOST_DEVICE constexpr bool get(uint8_t bit) const {
    return ((data >> bit) & 0x1) == 0x1;
  }

  HDS_HOST_DEVICE operator uint8_t() {
    return data;
  }

  HDS_HOST_DEVICE int count() {

    // TODO can be optimized with a popc

    int x = 0;
    for(int i = 0; i < bits; ++i) {
      if(get(i)) {
        x++;
      }
    }
    return x;
  }

  /// Get all bits set
  HDS_HOST_DEVICE static constexpr uint8_bitset<N> get_all_set() {
    uint8_bitset<N> result;
  
    for(int i = 0; i < N; ++i) {
      result.set(i, _true); 
    }
  
    return result;
  }

private:
  uint8_t data = 0x0;
};

template<int N>  
class uint32_bitset {
public:

  static constexpr int bits = N;

  static_assert(N <= 32);

  HDS_HOST_DEVICE constexpr uint32_bitset(std::integral auto x) : data(static_cast<uint32_t>(x)) {}

  constexpr uint32_bitset() = default;
  constexpr uint32_bitset(const uint32_bitset&) = default;
  constexpr uint32_bitset(uint32_bitset&&) = default;
  constexpr uint32_bitset& operator=(const uint32_bitset&) = default;
  constexpr uint32_bitset& operator=(uint32_bitset&&) = default;

  /**
   * set bit to x
   */
  HDS_HOST_DEVICE constexpr void set(uint8_t bit, bool x = true) {
    if (x == true) {
      data |= (0x1 << bit);
    } else {
      data &= (~(0x1 << bit));
    }
  }

  /**
   * set bit to x
   */
  template<bool x>
  HDS_HOST_DEVICE constexpr void set(uint8_t bit, std::integral_constant<bool, x>) {
    if constexpr (x == true) {
      data |= (0x1 << bit);
    } else {
      data &= (~(0x1 << bit));
    }
  }

  /// get bit x
  HDS_HOST_DEVICE constexpr bool get(uint8_t bit) const {
    return ((data >> bit) & 0x1) == 0x1;
  }

  HDS_HOST_DEVICE operator uint32_t() {
    return data;
  }

  HDS_HOST_DEVICE int count() {

    // TODO can be optimized with a popc

    int x = 0;
    for(int i = 0; i < bits; ++i) {
      if(get(i)) {
        x++;
      }
    }
    return x;
  }

  HDS_HOST_DEVICE int unset_bit() {
    for(int i = 0; i < bits; ++i) {
      if(!get(i)) {
        return i;
      }
    }
    return -1;
  }

  /// Get all bits set
  HDS_HOST_DEVICE constexpr uint32_bitset<N> get_all_set() {
    uint32_bitset<N> result;
  
    for(int i = 0; i < N; ++i) {
      result.set(i, _true); 
    }
  
    return result;
  }

private:

  uint32_t data = 0x0;
};


}
