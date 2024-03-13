#include <stdexcept>

#if defined(GPU)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

#include "../utility/annotations.h"

#pragma once

namespace remus::hds::threadgroup {

#if defined(GPU)
struct warp_threadgroup {

  static constexpr inline size_t size = 32;

  HDS_DEVICE warp_threadgroup() : warp(cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block())) {}

  HDS_HOST_DEVICE void sync() {
    #if defined (__CUDA_ARCH__)
    warp.sync();
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  HDS_HOST_DEVICE int ballot_index(bool vote) {
    #if defined (__CUDA_ARCH__)
    int b = warp.ballot(vote);
    return __ffs(b) - 1;
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  HDS_HOST_DEVICE bool any(bool vote) {
    #if defined (__CUDA_ARCH__)
    return warp.any(vote);
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  HDS_HOST_DEVICE bool is_leader() {
    #if defined (__CUDA_ARCH__)
    return warp.thread_rank() == 0;
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  HDS_HOST_DEVICE constexpr int leader_rank() {
    return 0;
  }
  
  HDS_HOST_DEVICE int thread_rank() {
    #if defined (__CUDA_ARCH__)
    return warp.thread_rank();
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_or(T val) {
    #if defined (__CUDA_ARCH__)
    namespace cg = cooperative_groups;
    return cg::reduce(warp, val, cg::bit_or<T>());
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_max(T val) {
    #if defined (__CUDA_ARCH__)
    namespace cg = cooperative_groups;
    return cg::reduce(warp, val, cg::greater<T>());
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_min(T val) {
    #if defined (__CUDA_ARCH__)
    namespace cg = cooperative_groups;
    return cg::reduce(warp, val, cg::less<T>());
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  template<typename T>
  HDS_HOST_DEVICE T shfl(T val, int idx) {
    #if defined (__CUDA_ARCH__)
    return warp.shfl(val, idx);
    #else
    throw std::runtime_error("Calling device code from CPU");
    #endif
  }

  cooperative_groups::thread_block_tile<32> warp;
};
#endif

struct single_threadgroup {

  static constexpr inline size_t size = 1;

  template<typename Fn>
  HDS_HOST_DEVICE void parallel_for(int begin, int end, Fn&& fn) {
    std::forward<Fn>(fn)(begin, end);
  }

  HDS_HOST_DEVICE void sync() {}

  HDS_HOST_DEVICE int ballot_index(bool vote) {
    return vote ? 0 : -1;
  }

  HDS_HOST_DEVICE bool any(bool vote) {
    return vote;
  }

  HDS_HOST_DEVICE constexpr bool is_leader() {
    return true;
  }

  HDS_HOST_DEVICE constexpr int leader_rank() {
    return 0;
  }
  
  HDS_HOST_DEVICE constexpr int thread_rank() {
    return 0;
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_or(T val) {
    return val;
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_max(T val) {
    return val;
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T reduce_min(T val) {
    return val;
  }

  template<typename T>
  HDS_HOST_DEVICE constexpr T shfl(T val, int idx) {
    return val;
  }
};


}
