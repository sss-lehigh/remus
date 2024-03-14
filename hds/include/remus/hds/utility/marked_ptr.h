#include "bit_cast.h"
#include "bitset.h"

#pragma once

namespace remus::hds::utility {

template <typename T, uint8_t B> class marked_ptr {
public:
  static constexpr uint8_t bits = B;
  static_assert(bits < 64, "Need at least 1 bit for indexing");

  static constexpr bool is_valid() { return (1 << bits) <= alignof(T); }

  HDS_HOST_DEVICE explicit constexpr marked_ptr(T *pointer) : ptr(pointer) {}

  HDS_HOST_DEVICE explicit constexpr marked_ptr(T *pointer, uint8_bitset<bits> mask) : ptr(pointer) {
    ptr = bit_cast<T *>(bit_cast<size_t>(ptr) | static_cast<size_t>(mask));
  }

  HDS_HOST_DEVICE constexpr marked_ptr(std::nullptr_t pointer) : ptr(pointer) {}

  marked_ptr() = default;
  marked_ptr(const marked_ptr &) = default;
  marked_ptr(marked_ptr &&) = default;

  marked_ptr &operator=(const marked_ptr &) = default;
  marked_ptr &operator=(marked_ptr &&) = default;

  HDS_HOST_DEVICE constexpr T *cast() const {
    auto mask = ~static_cast<size_t>(uint8_bitset<bits>::get_all_set());
    return bit_cast<T *>(bit_cast<size_t>(ptr) & mask);
  }

  HDS_HOST_DEVICE explicit constexpr operator T *() const { return this->cast(); }

  HDS_HOST_DEVICE constexpr uint8_bitset<bits> marks() const {
    auto mask = static_cast<size_t>(uint8_bitset<bits>::get_all_set());
    return uint8_bitset<bits>(static_cast<uint8_t>(bit_cast<size_t>(ptr) & mask));
  }

  HDS_HOST_DEVICE constexpr bool is_marked(int i) const { return marks().get(i); }

  [[nodiscard]] HDS_HOST_DEVICE constexpr marked_ptr<T, bits> set_marked(int i) const {
    auto new_bitset = marks();
    new_bitset.set(i, Bool<true>{});
    marked_ptr<T, bits> new_ptr(cast(), new_bitset);
    return new_ptr;
  }

  [[nodiscard]] HDS_HOST_DEVICE constexpr marked_ptr<T, bits> set_unmarked(int i) const {
    auto new_bitset = marks();
    new_bitset.set(i, Bool<false>{});
    marked_ptr<T, bits> new_ptr(cast(), new_bitset);
    return new_ptr;
  }

  [[nodiscard]] HDS_HOST_DEVICE constexpr marked_ptr<T, bits> set_marks(uint8_bitset<bits> new_bitset) const {
    marked_ptr<T, bits> new_ptr(cast(), new_bitset);
    return new_ptr;
  }

  HDS_HOST_DEVICE T &operator*() { return *this->cast(); }

  HDS_HOST_DEVICE const T &operator*() const { return *this->cast(); }

  HDS_HOST_DEVICE T *operator->() { return this->cast(); }

  HDS_HOST_DEVICE const T *operator->() const { return this->cast(); }

  HDS_HOST_DEVICE constexpr bool operator==(marked_ptr<T, bits> rhs) const { return this->ptr == rhs.ptr; }

  HDS_HOST_DEVICE constexpr bool operator!=(marked_ptr<T, bits> rhs) const { return this->ptr != rhs.ptr; }

  template <typename U, uint8_t N_> friend std::ostream &operator<<(std::ostream &, marked_ptr<U, N_>);

private:
  T *ptr;
};

template <typename T, uint8_t bits> std::ostream &operator<<(std::ostream &os, marked_ptr<T, bits> ptr) {
  os << bit_cast<void *>(ptr);
  return os;
}

} // namespace remus::hds::utility
