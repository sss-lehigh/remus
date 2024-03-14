#include "../utility/annotations.h"
#include "../utility/atomic.h"
#include <cstdlib>
#include <stdexcept>

#pragma once

namespace remus::hds::allocator {

class device_allocator {
public:
  template <typename T> HDS_HOST_DEVICE T *allocate(size_t amount) {
#if defined(GPU)
    T *ptr;
    auto err = cudaMalloc(&ptr, sizeof(T) * amount);
    if (err != cudaSuccess) {
  #if defined(__CUDA_ARCH__)
      __trap();
  #else
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  #endif
    }
    return ptr;
#else
    return nullptr;
#endif
  }

  template <typename T> HDS_HOST_DEVICE void deallocate(T *ptr, size_t amount) {
#if defined(GPU)
    auto err = cudaFree(ptr);
    if (err != cudaSuccess) {
  #if defined(__CUDA_ARCH__)
      __trap();
  #else
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  #endif
    }
#endif
  }
};

class heap_allocator {
public:
  template <typename T> HDS_HOST_DEVICE T *allocate(size_t amount) {
    return reinterpret_cast<T *>(malloc(sizeof(T) * amount));
  }

  template <typename T> HDS_HOST_DEVICE void deallocate(T *ptr, size_t amount) { free(ptr); }
};

template <typename Allocator> class bump_allocator {
private:
  struct chunk {
    alignas(128) char bytes[128];
  };

public:
  HDS_HOST_DEVICE bump_allocator(size_t size_ = 1ull << 30) : size(size_ / 128) {
    alloc.template allocate<chunk>(size_ / 128);
    alloc.template allocate<atomic<size_t>>(1);
    bump->store(0);
  }

  template <typename T> HDS_HOST_DEVICE T *allocate(size_t amount) {

    size_t chunks = (amount * sizeof(T) + 127) / 128;

    size_t offset = bump->fetch_add(chunks, memory_order_relaxed);

    if (offset + chunks > size) {
      return alloc.template allocate<T>(amount);
    }

    return reinterpret_cast<T *>(memory + offset);
  }

  template <typename T> HDS_HOST_DEVICE void deallocate(T *ptr, size_t amount) {
    if (!(reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(memory) and
          reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(memory + size))) {
      alloc.deallocate(ptr, amount);
    }
  }

  chunk *memory;
  atomic<size_t> *bump;
  size_t size;

  Allocator alloc;
};

template <typename Allocator> class gpu_bump_allocator {
private:
  struct chunk {
    alignas(128) char bytes[128];
  };

public:
  HDS_HOST gpu_bump_allocator(size_t size_ = 1ull << 30) : size(size_ / 128) {
    memory = device_allocator{}.template allocate<chunk>(size);
    bump = device_allocator{}.template allocate<atomic<size_t>>(1);

#if defined(GPU)
    cudaError_t err;
    if ((err = cudaMemset(bump, 0, sizeof(size_t))) != cudaSuccess) {
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
    }
#endif
  }

  template <typename T> HDS_HOST_DEVICE T *allocate(size_t amount) {

    size_t chunks = (amount * sizeof(T) + 127) / 128;

    size_t offset = bump->fetch_add(chunks, memory_order_relaxed);

    if (offset + chunks > size) {
      return alloc.template allocate<T>(amount);
    }

    return reinterpret_cast<T *>(memory + offset);
  }

  template <typename T> HDS_HOST_DEVICE void deallocate(T *ptr, size_t amount) {
    if (!(reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(memory) and
          reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(memory + size))) {
      alloc.deallocate(ptr, amount);
    }
  }

  chunk *memory;
  atomic<size_t> *bump;
  size_t size;

  Allocator alloc;
};

}; // namespace remus::hds::allocator
