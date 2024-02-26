#pragma once

#include <iostream>

namespace rome::rdma {

// Forward declarations
//
// TODO: If we had a base type for rdma_ptr, we could probably not need this?
struct nullptr_type {};

/// A "smart pointer" to memory on another machine
///
/// TODO: This will need more documentation
template <typename T> class rdma_ptr {
public:
  using element_type = T;
  using pointer = T *;
  using reference = T &;
  using id_type = uint16_t;
  using address_type = uint64_t;

  // template <typename U> using rebind = rdma_ptr<U>;

  // Constructors
  constexpr rdma_ptr() : raw_(0) {}
  explicit rdma_ptr(uint64_t raw) : raw_(raw) {}
  rdma_ptr(uint64_t id, T *address)
      : rdma_ptr(id, reinterpret_cast<uint64_t>(address)) {}
  rdma_ptr(id_type id, uint64_t address)
      : raw_((((uint64_t)id) << kAddressBits) | (address & kAddressBitmask)) {}

  // Copy and Move
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  rdma_ptr(const rdma_ptr &p) : raw_(p.raw_) {}
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  rdma_ptr(rdma_ptr &&p) : raw_(p.raw_) {}
  constexpr rdma_ptr(const rdma_ptr<nullptr_type> &) : raw_(0) {}

  // Getters
  uint64_t id() const { return (raw_ & kIdBitmask) >> kAddressBits; }
  uint64_t address() const { return raw_ & kAddressBitmask; }
  uint64_t raw() const { return raw_; }

  // Assignment
  void operator=(const rdma_ptr &p) volatile { raw_ = p.raw_; }
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  void operator=(const rdma_ptr<nullptr_type> &) volatile {
    raw_ = 0;
  }

  /// Increment operator
  rdma_ptr &operator+=(size_t s) {
    const auto address = (raw_ + (sizeof(element_type) * s)) & kAddressBitmask;
    raw_ = (raw_ & kIdBitmask) | address;
    return *this;
  }

  /// Increment operator
  rdma_ptr operator+(size_t s) {
    rdma_ptr new_ptr = *this;
    new_ptr += s;
    return new_ptr;
  }

  /// Pre-increment
  rdma_ptr &operator++() {
    *this += 1;
    return *this;
  }

  /// Post-increment
  rdma_ptr operator++(int) {
    rdma_ptr prev = *this;
    *this += 1;
    return prev;
  }

  /// Decrement operator
  rdma_ptr &operator-=(size_t s) {
    const auto address = (raw_ - (sizeof(element_type) * s)) & kAddressBitmask;
    raw_ = (raw_ & kIdBitmask) | address;
    return *this;
  }
  
  /// Decrement operator
  rdma_ptr operator-(size_t s) {
    rdma_ptr new_ptr = *this;
    new_ptr -= s;
    return new_ptr;
  }

  /// Pre-decrement
  rdma_ptr &operator--() {
    *this -= 1;
    return *this;
  }

  /// Post-decrement
  rdma_ptr operator--(int) {
    rdma_ptr prev = *this;
    *this -= 1;
    return prev;
  }

  // Conversion operators
  explicit operator uint64_t() const { return raw_; }
  template <typename U> explicit operator rdma_ptr<U>() const {
    return rdma_ptr<U>(raw_);
  }

  // Pointer-like functions
  static constexpr element_type *to_address(const rdma_ptr &p) {
    return (element_type *)p.address();
  }
  static constexpr rdma_ptr pointer_to(element_type &p) {
    return rdma_ptr(-1, &p);
  }
  pointer get() const { return (element_type *)address(); }
  pointer operator->() const noexcept { return (element_type *)address(); }
  reference operator*() const noexcept { return *((element_type *)address()); }

  // Stream operator
  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const rdma_ptr<U> &p);

  // Equivalence
  bool operator==(const volatile rdma_ptr<nullptr_type> &) const volatile {
    return raw_ == 0;
  }
  bool operator==(rdma_ptr &n) { return raw_ == n.raw_; }
  template <typename U>
  friend bool operator==(rdma_ptr<U> &p1, rdma_ptr<U> &p2);
  template <typename U>
  friend bool operator==(const volatile rdma_ptr<U> &p1,
                         const volatile rdma_ptr<U> &p2);

  bool operator<(const volatile rdma_ptr<T> &p) { return raw_ < p.raw_; }
  friend bool operator<(const volatile rdma_ptr<T> &p1,
                        const volatile rdma_ptr<T> &p2) {
    return p1.raw() < p2.raw();
  }

private:
  static inline constexpr uint64_t bitsof(const uint32_t &bytes) {
    return bytes * 8;
  }

  static constexpr auto kAddressBits =
      (bitsof(sizeof(uint64_t))) - bitsof(sizeof(id_type));
  static constexpr auto kAddressBitmask = ((1ul << kAddressBits) - 1);
  static constexpr auto kIdBitmask = (uint64_t)(-1) ^ kAddressBitmask;

  uint64_t raw_;
};

/// Operator support for printing a rdma_ptr<U>
template <typename U>
std::ostream &operator<<(std::ostream &os, const rdma_ptr<U> &p) {
  return os << "<id=" << p.id() << ", address=0x" << std::hex << p.address()
            << std::dec << ">";
}

/// Operator support for equality tests of rdma_ptr<U>
template <typename U>
bool operator==(const volatile rdma_ptr<U> &p1,
                const volatile rdma_ptr<U> &p2) {
  return p1.raw_ == p2.raw_;
}

using rdma_nullptr_t = rdma_ptr<nullptr_type>;

/// A single global instance of a null rdma_ptr
constexpr rdma_nullptr_t rdma_nullptr{};

} // namespace rome::rdma
