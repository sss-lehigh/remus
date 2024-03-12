#include <cstdio>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <rome/hds/unordered_map/kv_linked_list/lazy_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/lazy_nodes/reg_cached_nodes.h>
#include <rome/hds/threadgroup/threadgroup.h>
#include <set>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
__trap();
#else
exit(1);
#endif
}

#define ASSERT(x, y) if(!(x)) { printf("%s did not evaluate to true for i = %d\n", #x, (y)); error(); }

using namespace rome::hds::kv_linked_list;
using namespace rome::hds;

template<typename T>
__global__ void single_thread_test(T* ll) {

  ll = new (ll) T(); 

  auto group = threadgroup::single_threadgroup{};

  ASSERT(ll->get(1, group) == nullopt, 1);

  for (int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->insert(r, r, group), r);

    if (!ll->validate(group)) {
      error();
    }
  }
  
  for (int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->remove(r, group), r);

    if(!ll->validate(group)) {
      error();
    }
  }

  ll->~T();
}

template<typename T>
__global__ void warp_test(T* ll) {

  auto warp = threadgroup::warp_threadgroup{};
  if (warp.is_leader()) {
    new (ll) T();
  }
  warp.sync();

  static_assert(decltype(warp)::size == 32);

  ASSERT(ll->get(1, warp) == nullopt, 1);

  for(int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->insert(r, r, warp), r);

    if(!ll->validate(warp)) {
      error();
    }
  }

  for(int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->remove(r, warp), r);

    if(!ll->validate(warp)) {
      error();
    }
  }

  warp.sync();
  if (warp.is_leader()) {
    ll->~T();
  }
}

int hoh() {

  allocator::device_allocator dev_mem;
  kv_lock_linked_list<int, int, 2, locked_nodes::reg_cached_node_pointer, allocator::heap_allocator>* st_gpu_ll;
  st_gpu_ll = dev_mem.allocate<kv_lock_linked_list<int, int, 2, locked_nodes::reg_cached_node_pointer, allocator::heap_allocator>>(1);

  single_thread_test<<<1, 1>>>(st_gpu_ll);

  auto err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }

  dev_mem.deallocate(st_gpu_ll, 1);

  kv_lock_linked_list<int, int, 32, locked_nodes::reg_cached_node_pointer, allocator::heap_allocator>* w_gpu_ll;
  w_gpu_ll = dev_mem.allocate<kv_lock_linked_list<int, int, 32, locked_nodes::reg_cached_node_pointer, allocator::heap_allocator>>(1);

  warp_test<<<1, 32>>>(w_gpu_ll);

  err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }
  
  dev_mem.deallocate(w_gpu_ll, 1);

  return 0;
}

int lazy() {

  allocator::device_allocator dev_mem;
  kv_lazy_linked_list<int, int, 2, lazy_nodes::reg_cached_node_pointer, allocator::heap_allocator>* st_gpu_ll;
  st_gpu_ll = dev_mem.allocate<kv_lazy_linked_list<int, int, 2, lazy_nodes::reg_cached_node_pointer, allocator::heap_allocator>>(1);

  single_thread_test<<<1, 1>>>(st_gpu_ll);

  auto err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }

  dev_mem.deallocate(st_gpu_ll, 1);

  kv_lazy_linked_list<int, int, 32, lazy_nodes::reg_cached_node_pointer, allocator::heap_allocator>* w_gpu_ll;
  w_gpu_ll = dev_mem.allocate<kv_lazy_linked_list<int, int, 32, lazy_nodes::reg_cached_node_pointer, allocator::heap_allocator>>(1);

  warp_test<<<1, 32>>>(w_gpu_ll);

  err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }
  
  dev_mem.deallocate(w_gpu_ll, 1);

  return 0;
}

int main() {
  if (hoh() != 0) {
    return 1;
  }
  return lazy();
}

