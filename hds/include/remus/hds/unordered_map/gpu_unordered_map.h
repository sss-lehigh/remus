#include <future>

#include "../allocator/allocator.h"
#include "../threadgroup/threadgroup.h"
#include "kv_linked_list/lazy_linked_list.h"
#include "kv_linked_list/lazy_nodes/reg_cached_nodes.h"
#include "unordered_map.h"

#pragma once

namespace remus::hds {

#if defined(GPU)

template <typename T, typename... Arg_t> __global__ void construct_on_gpu(T *ptr, Arg_t... args) {
  new (ptr) T(args...);
}

template <typename T> __global__ void destruct_on_gpu(T *ptr) { ptr->~T(); }

template <typename T, typename K, typename V>
__global__ void get_um_on_gpu(T *map_, K *keys, optional<V> *results, size_t size) {
  T *map = reinterpret_cast<T *>(__cvta_generic_to_global(map_));

  int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  auto group = threadgroup::warp_threadgroup{};

  for (int i = 0; i < 32; ++i) {
    if (wid * 32 + i >= size)
      return;
    results[wid * 32 + i] = map->get(keys[wid * 32 + i], group);
  }
}

template <typename T, typename K, typename V>
__global__ void insert_um_on_gpu(T *map_, K *keys, V *values, bool *results, size_t size) {
  T *map = reinterpret_cast<T *>(__cvta_generic_to_global(map_));

  int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  auto group = threadgroup::warp_threadgroup{};

  for (int i = 0; i < 32; ++i) {
    if (wid * 32 + i >= size)
      return;
    results[wid * 32 + i] = map->insert(keys[wid * 32 + i], values[wid * 32 + i], group);
  }
}

template <typename T, typename K> __global__ void remove_um_on_gpu(T *map_, K *keys, bool *results, size_t size) {
  T *map = reinterpret_cast<T *>(__cvta_generic_to_global(map_));

  int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  auto group = threadgroup::warp_threadgroup{};

  for (int i = 0; i < 32; ++i) {
    if (wid * 32 + i >= size)
      return;
    results[wid * 32 + i] = map->remove(keys[wid * 32 + i], group);
  }
}

template <typename K, typename V, typename Allocator = allocator::gpu_bump_allocator<allocator::heap_allocator>,
          typename Hash = BasicHash<K>,
          template <typename, typename, int, template <typename, typename, int> typename, typename, typename>
          typename linked_list_ = kv_linked_list::kv_lazy_linked_list,
          template <typename, typename, int> typename node_pointer_ =
            kv_linked_list::lazy_nodes::reg_cached_node_pointer>
class gpu_unordered_map {
private:
  using um_t = unordered_map<K, V, 32, linked_list_, node_pointer_, Allocator,
                             kv_linked_list::kv_inplace_construct<K, V, 32, node_pointer_>, Allocator, Hash>;

public:
  HDS_HOST gpu_unordered_map(uint32_t size) : gpu_unordered_map(size, Allocator{}) {}

  template <typename A> HDS_HOST gpu_unordered_map(uint32_t size, A &&alloc) {
    if (cudaMalloc(&map_, sizeof(um_t)) != cudaSuccess) {
      throw std::runtime_error("Unable to allocate on GPU");
    }
    construct_on_gpu<<<1, 1>>>(map_, fast_div_mod(size), std::forward<A>(alloc));
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("Unable to construct on GPU");
    }
  }

  HDS_HOST ~gpu_unordered_map() noexcept(false) {
    destruct_on_gpu<<<1, 1>>>(map_);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("Unable to destruct on GPU");
    }
    cudaFree(map_);
  }

  HDS_HOST std::future<void> get(K *keys, optional<V> *results, size_t size, cudaStream_t stream = 0x0) {

    get_um_on_gpu<<<(size + 255) / 256, 256, 0, stream>>>(map_, keys, results, size);
    return std::async(std::launch::deferred, [stream]() {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("Unable to synchronize");
      }
    });
  }

  HDS_HOST std::future<void> insert(K *keys, V *values, bool *results, size_t size, cudaStream_t stream = 0x0) {

    insert_um_on_gpu<<<(size + 255) / 256, 256, 0, stream>>>(map_, keys, values, results, size);
    return std::async(std::launch::deferred, [stream]() {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("Unable to synchronize");
      }
    });
  }

  HDS_HOST std::future<void> remove(K *keys, bool *results, size_t size, cudaStream_t stream = 0x0) {

    remove_um_on_gpu<<<(size + 255) / 256, 256, 0, stream>>>(map_, keys, results, size);
    return std::async(std::launch::deferred, [stream]() {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("Unable to synchronize");
      }
    });
  }

  HDS_DEVICE auto map() { return __cvta_generic_to_global(map_); }

private:
  um_t *map_;
};

#endif

}; // namespace remus::hds
