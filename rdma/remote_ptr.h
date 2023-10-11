#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>

#include <spdlog/fmt/fmt.h> // [mfs] For now, use spdlog's vendored fmt...

namespace rome::rdma {

template <typename T> class remote_ptr;
struct nullptr_type {};
typedef remote_ptr<nullptr_type> remote_nullptr_t;

template <typename T> class remote_ptr {
public:
  using element_type = T;
  using pointer = T *;
  using reference = T &;
  using id_type = uint16_t;
  using address_type = uint64_t;

  template <typename U> using rebind = remote_ptr<U>;

  // Constructors
  constexpr remote_ptr() : raw_(0) {}
  explicit remote_ptr(uint64_t raw) : raw_(raw) {}
  remote_ptr(uint64_t id, T *address)
      : remote_ptr(id, reinterpret_cast<uint64_t>(address)) {}
  remote_ptr(id_type id, uint64_t address)
      : raw_((((uint64_t)id) << kAddressBits) | (address & kAddressBitmask)) {}

  // Copy and Move
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  remote_ptr(const remote_ptr &p) : raw_(p.raw_) {}
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  remote_ptr(remote_ptr &&p) : raw_(p.raw_) {}
  constexpr remote_ptr(const remote_nullptr_t &n) : raw_(0) {}

  // Getters
  uint64_t id() const { return (raw_ & kIdBitmask) >> kAddressBits; }
  uint64_t address() const { return raw_ & kAddressBitmask; }
  uint64_t raw() const { return raw_; }

  // Assignment
  void operator=(const remote_ptr &p) volatile;
  template <typename _T = element_type,
            std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
  void operator=(const remote_nullptr_t &n) volatile;

  // Increment operator
  remote_ptr &operator+=(size_t s);
  remote_ptr &operator++();
  remote_ptr operator++(int);

  // Conversion operators
  explicit operator uint64_t() const;
  template <typename U> explicit operator remote_ptr<U>() const;

  // Pointer-like functions
  static constexpr element_type *to_address(const remote_ptr &p);
  static constexpr remote_ptr pointer_to(element_type &r);
  pointer get() const { return (element_type *)address(); }
  pointer operator->() const noexcept { return (element_type *)address(); }
  reference operator*() const noexcept { return *((element_type *)address()); }

  // Stream operator
  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const remote_ptr<U> &p);

  // Equivalence
  bool operator==(const volatile remote_nullptr_t &n) const volatile;
  bool operator==(remote_ptr &n) { return raw_ == n.raw_; }
  template <typename U>
  friend bool operator==(remote_ptr<U> &p1, remote_ptr<U> &p2);
  template <typename U>
  friend bool operator==(const volatile remote_ptr<U> &p1,
                         const volatile remote_ptr<U> &p2);

  bool operator<(const volatile remote_ptr<T> &p) { return raw_ < p.raw_; }
  friend bool operator<(const volatile remote_ptr<T> &p1,
                        const volatile remote_ptr<T> &p2) {
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

} // namespace rome::rdma

template <typename T> struct fmt::formatter<::rome::rdma::remote_ptr<T>> {
  typedef ::rome::rdma::remote_ptr<T> remote_ptr;
  constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
    return ctx.end();
  }

  template <typename FormatContext>
  auto format(const remote_ptr &input, FormatContext &ctx)
      -> decltype(ctx.out()) {
    return format_to(ctx.out(), "(id={}, address=0x{:x})", input.id(),
                     input.address());
  }
};

template <typename T> struct std::hash<rome::rdma::remote_ptr<T>> {
  std::size_t operator()(const rome::rdma::remote_ptr<T> &ptr) const {
    return std::hash<uint64_t>()(static_cast<uint64_t>(ptr));
  }
};

namespace rome::rdma {

constexpr remote_nullptr_t remote_nullptr{};

template <typename T>
void remote_ptr<T>::operator=(const remote_ptr &p) volatile {
  raw_ = p.raw_;
}

template <typename T>
template <typename _T, std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
void remote_ptr<T>::operator=(const remote_nullptr_t &n) volatile {
  raw_ = 0;
}

template <typename T> remote_ptr<T> &remote_ptr<T>::operator+=(size_t s) {
  const auto address = (raw_ + (sizeof(element_type) * s)) & kAddressBitmask;
  raw_ = (raw_ & kIdBitmask) | address;
  return *this;
}

template <typename T> remote_ptr<T> &remote_ptr<T>::operator++() {
  *this += 1;
  return *this;
}

template <typename T> remote_ptr<T> remote_ptr<T>::operator++(int) {
  remote_ptr prev = *this;
  *this += 1;
  return prev;
}

template <typename T> remote_ptr<T>::operator uint64_t() const { return raw_; }

template <typename T>
template <typename U>
remote_ptr<T>::operator remote_ptr<U>() const {
  return remote_ptr<U>(raw_);
}

template <typename T>
/* static */ constexpr typename remote_ptr<T>::element_type *
remote_ptr<T>::to_address(const remote_ptr<T> &p) {
  return (element_type *)p.address();
}

template <typename T>
/* static */ constexpr remote_ptr<T> remote_ptr<T>::pointer_to(T &p) {
  return remote_ptr(-1, &p);
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const remote_ptr<U> &p) {
  return os << "<id=" << p.id() << ", address=0x" << std::hex << p.address()
            << std::dec << ">";
}

template <typename T>
bool remote_ptr<T>::operator==(const volatile remote_nullptr_t &n) const
    volatile {
  return raw_ == 0;
}

template <typename U>
bool operator==(const volatile remote_ptr<U> &p1,
                const volatile remote_ptr<U> &p2) {
  return p1.raw_ == p2.raw_;
}

} // namespace rome::rdma