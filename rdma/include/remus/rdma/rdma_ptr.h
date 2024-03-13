#pragma once

#include <iostream>

namespace remus::rdma {

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
  /// Able to set rdma_ptr to nullptr
  constexpr rdma_ptr(std::nullptr_t) : raw_(0) {}

  // Copy and Move
  template <typename _T = element_type>
  rdma_ptr(const rdma_ptr &p) : raw_(p.raw_) {}
  template <typename _T = element_type>
  rdma_ptr(rdma_ptr &&p) : raw_(p.raw_) {}

  // Getters
  constexpr uint64_t id() const volatile { return (raw_ & kIdBitmask) >> kAddressBits; }
  constexpr uint64_t address() const volatile { return raw_ & kAddressBitmask; }
  constexpr uint64_t raw() const volatile { return raw_; }

  // Assignment
  void operator=(const rdma_ptr &p) volatile { raw_ = p.raw_; }

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

  explicit operator T*() volatile const {
    return reinterpret_cast<T*>(address());
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
  constexpr bool operator==(std::nullptr_t n) const volatile {
    return static_cast<T*>(*this) == n;
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

} // namespace remus::rdma
