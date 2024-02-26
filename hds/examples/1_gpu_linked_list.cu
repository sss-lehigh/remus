#include <cstdio>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/linked_list/lock_linked_list.h>
#include <rome/hds/linked_list/locked_nodes/reg_cached_nodes.h>
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

__global__ void single_thread_test(rome::hds::lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* ll) {

  ll = new (ll) rome::hds::lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>(); 

  auto group = rome::hds::threadgroup::single_threadgroup{};

  ASSERT(!ll->contains(1, group), 1);

  for (int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->insert(r, group), r);

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

  ll->~lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>(); 
}

__global__ void warp_test(rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* ll) {

  auto warp = rome::hds::threadgroup::warp_threadgroup{};
  if (warp.is_leader()) {
    new (ll) rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>(); 
  }
  warp.sync();

  static_assert(decltype(warp)::size == 32);

  ASSERT(!ll->contains(1, warp), 1);

  for(int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->insert(r, warp), r);

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
    ll->~lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>(); 
  }
}

int main() {
  rome::hds::lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator> ll;
  auto group = rome::hds::threadgroup::single_threadgroup{};
  ASSERT(!ll.contains(1, group), 1);

  std::set<int> reference;

  for(int i = 0; i < 100; ++i) {

    if(rand() % 2 == 0) {

      int r = rand();

      bool inserted = reference.insert(r).second;
      ASSERT(ll.insert(r, group) == inserted, r);

      printf("\nInserted %d\n", r);
      ll.print(group);

    } else {

      int r = rand();

      bool removed = (reference.erase(r) == 1);

      ASSERT(ll.remove(r, group) == removed, r);

      printf("\nRemoved %d\n", r);
      ll.print(group);

    }

    if(!ll.validate(group)) {
      return 1;
    }

    for(auto elm : reference) {
      ASSERT(ll.contains(elm, group), elm);
    }
  }

  rome::hds::allocator::device_allocator dev_mem;
  rome::hds::lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* st_gpu_ll;
  st_gpu_ll = dev_mem.allocate<rome::hds::lock_linked_list<int, 2, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>>(1);

  single_thread_test<<<1, 1>>>(st_gpu_ll);

  auto err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }

  dev_mem.deallocate(st_gpu_ll, 1);

  rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* w_gpu_ll;
  w_gpu_ll = dev_mem.allocate<rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>>(1);

  warp_test<<<1, 32>>>(w_gpu_ll);

  err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }
  
  dev_mem.deallocate(w_gpu_ll, 1);

  return 0;
}

__launch_bounds__(1024, 1)
__global__ void warp_insert(rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* ll, int i) {

  auto warp = rome::hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->insert(i, warp);

}

__launch_bounds__(1024, 1)
__global__ void warp_remove(rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* ll, int i) {

  auto warp = rome::hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->remove(i, warp);

}

__global__ void warp_contains(rome::hds::lock_linked_list<int, 32, rome::hds::locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator>* ll, int i) {

  auto warp = rome::hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->contains(i, warp);
 
}
