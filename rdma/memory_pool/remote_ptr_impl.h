#include "remote_ptr.h"

namespace rome::rdma {

constexpr remote_nullptr_t remote_nullptr{};

template <typename T>
void remote_ptr<T>::operator=(const remote_ptr& p) volatile {
  raw_ = p.raw_;
}

template <typename T>
template <typename _T, std::enable_if_t<!std::is_same_v<_T, nullptr_type>>>
void remote_ptr<T>::operator=(const remote_nullptr_t& n) volatile {
  raw_ = 0;
}

template <typename T>
remote_ptr<T>& remote_ptr<T>::operator+=(size_t s) {
  const auto address = (raw_ + (sizeof(element_type) * s)) & kAddressBitmask;
  raw_ = (raw_ & kIdBitmask) | address;
  return *this;
}

template <typename T>
remote_ptr<T>& remote_ptr<T>::operator++() {
  *this += 1;
  return *this;
}

template <typename T>
remote_ptr<T> remote_ptr<T>::operator++(int) {
  remote_ptr prev = *this;
  *this += 1;
  return prev;
}

template <typename T>
remote_ptr<T>::operator uint64_t() const {
  return raw_;
}

template <typename T>
template <typename U>
remote_ptr<T>::operator remote_ptr<U>() const {
  return remote_ptr<U>(raw_);
}

template <typename T>
/* static */ constexpr typename remote_ptr<T>::element_type*
remote_ptr<T>::to_address(const remote_ptr<T>& p) {
  return (element_type*)p.address();
}

template <typename T>
/* static */ constexpr remote_ptr<T> remote_ptr<T>::pointer_to(T& p) {
  return remote_ptr(-1, &p);
}

template <typename U>
std::ostream& operator<<(std::ostream& os, const remote_ptr<U>& p) {
  return os << "<id=" << p.id() << ", address=0x" << std::hex << p.address()
            << std::dec << ">";
}

template <typename T>
bool remote_ptr<T>::operator==(const volatile remote_nullptr_t& n) const
    volatile {
  return raw_ == 0;
}

template <typename U>
bool operator==(const volatile remote_ptr<U>& p1,
                const volatile remote_ptr<U>& p2) {
  return p1.raw_ == p2.raw_;
}

}  // namespace rome::rdma