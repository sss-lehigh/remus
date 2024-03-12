#include <cstdio>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <rome/hds/unordered_map/unordered_map.h>
#include <rome/hds/threadgroup/threadgroup.h>
#include <rome/hds/unordered_map/gpu_unordered_map.h>
#include <set>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

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

using single_thread_um = unordered_map<int, 
                                       int, 
                                       2, 
                                       kv_lock_linked_list,
                                       locked_nodes::reg_cached_node_pointer, 
                                       allocator::heap_allocator>;

using wc_um = unordered_map<int, 
                            int, 
                            32, 
                            kv_lock_linked_list,
                            locked_nodes::reg_cached_node_pointer, 
                            allocator::heap_allocator>;


__global__ void single_thread_test(single_thread_um* map, fast_div_mod d) {

  map = new (map) single_thread_um(d); 

  auto group = threadgroup::single_threadgroup{};

  ASSERT(map->get(1, group) == nullopt, 1);

  for (int i = 0; i < 100; ++i) {
    int r = i;
    ASSERT(map->insert(r, r, group), r);
  }
  
  for (int i = 0; i < 100; ++i) {
    int r = i;
    ASSERT(map->remove(r, group), r);
  }

  map->~single_thread_um(); 
}

__global__ void warp_test(wc_um* map, fast_div_mod d) {

  auto warp = threadgroup::warp_threadgroup{};
  if (warp.is_leader()) {
    new (map) wc_um(d); 
  }
  warp.sync();

  static_assert(decltype(warp)::size == 32);

  ASSERT(map->get(1, warp) == nullopt, 1);

  for(int i = 0; i < 100; ++i) {
    int r = i;
    ASSERT(map->insert(r, r, warp), r);
  }

  for(int i = 0; i < 100; ++i) {
    int r = i;
    ASSERT(map->remove(r, warp), r);
  }

  warp.sync();
  if (warp.is_leader()) {
    map->~wc_um(); 
  }
}

int main() {

  fast_div_mod d(100);

  allocator::device_allocator dev_mem;
  single_thread_um* st_gpu_map;

  st_gpu_map = dev_mem.allocate<single_thread_um>(1);

  single_thread_test<<<1, 1>>>(st_gpu_map, d);

  auto err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }

  dev_mem.deallocate(st_gpu_map, 1);

  wc_um* w_gpu_map;
  w_gpu_map = dev_mem.allocate<wc_um>(1);

  warp_test<<<1, 32>>>(w_gpu_map, d);

  err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }
  
  dev_mem.deallocate(w_gpu_map, 1);

  gpu_unordered_map<int, int> gpu_map(100);

  thrust::device_vector<int> keys(256);
  thrust::sequence(keys.begin(), keys.end(), 1);
  thrust::device_vector<int> values(256, 1);
  thrust::device_vector<bool> results(256);
  thrust::device_vector<optional<int>> results2(256);
  thrust::device_vector<bool> results3(256);

  gpu_map.insert(keys.data().get(), values.data().get(), results.data().get(), keys.size());
  gpu_map.get(keys.data().get(), results2.data().get(), keys.size());
  gpu_map.remove(keys.data().get(), results3.data().get(), keys.size()).wait();

  thrust::host_vector<bool> h_results = results;
  thrust::host_vector<optional<int>> h_results2 = results2;
  thrust::host_vector<bool> h_results3 = results3;

  int count = 0;
  for(auto res : h_results) {
    ASSERT(res, count++);
  }

  count = 0;
  for(auto res : h_results2) {
    ASSERT(res != nullopt && *res == 1, count++);
  }

  count = 0;
  for(auto res : h_results3) {
    ASSERT(res, count++);
  }

  return 0;
}

