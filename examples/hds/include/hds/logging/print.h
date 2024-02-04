#include <cstdio>
#include <type_traits>

#include "../utility/annotations.h"

#pragma once

namespace hds::logging {

  template<typename T>
  struct hex {
    explicit hex(T x) : val(x) {}
    T val;
  };

  template<typename T>
  HDS_HOST_DEVICE_NO_INLINE void print_val(T);

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(int x) {
    printf("%d", x);
  }

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(hex<int> x) {
    printf("%x", x.val);
  }
  
  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(unsigned x) {
    printf("%u", x);
  }

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(long x) {
    printf("%ld", x);
  }

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(unsigned long x) {
    printf("%lu", x);
  }

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(long long x) {
    printf("%lld", x);
  }

  template<>
  HDS_HOST_DEVICE_NO_INLINE void print_val(unsigned long long x) {
    printf("%llu", x);
  }

  template<typename U>
  HDS_HOST_DEVICE_NO_INLINE void print_val(U* x) {
    if constexpr (std::is_same_v<decltype(x), char*> || std::is_same_v<decltype(x), const char*>) {
      printf("%s", x);
    } else {
      printf("%p", x);
    }
  }

  template<typename T>
  HDS_HOST_DEVICE_NO_INLINE void print(T arg1) {
    print_val(arg1);
  }

  template<typename T, typename ... Arg_t>
  HDS_HOST_DEVICE_NO_INLINE void print(T arg1, Arg_t... args) {
    print_val(arg1);
    print(args...);
  }

  template<typename T, typename ... Arg_t>
  HDS_HOST_DEVICE_NO_INLINE void print_sep(const char* sep, T arg1, Arg_t... args) {
    print_val(arg1);
    print_val(sep);
    print_sep(sep, args...);
  }

}
