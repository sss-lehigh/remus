#include <cstdlib>
#include <stdexcept>
#include "../utility/annotations.h"

#pragma once

namespace hds::allocator {

class device_allocator {
public:

  template<typename T>
  HDS_HOST_DEVICE T* allocate(size_t amount) {
    T* ptr;
    auto err = cudaMalloc(&ptr, sizeof(T) * amount);
    if (err != cudaSuccess) {
      #if defined(__CUDA_ARCH__)
      __trap();
      #else
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
      #endif
    }
    return ptr;
  }

  template<typename T>
  HDS_HOST_DEVICE void deallocate(T* ptr, size_t amount) {
    auto err = cudaFree(ptr);
    if (err != cudaSuccess) {
      #if defined(__CUDA_ARCH__)
      __trap();
      #else
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
      #endif
    }
  }

};

class heap_allocator {
public:

  template<typename T>
  HDS_HOST_DEVICE T* allocate(size_t amount) {
    return reinterpret_cast<T*>(malloc(sizeof(T) * amount));  
  }

  template<typename T>
  HDS_HOST_DEVICE void deallocate(T* ptr, size_t amount) {
    free(ptr);  
  }

};

};
