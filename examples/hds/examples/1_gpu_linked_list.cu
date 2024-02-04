#include <cstdio>

#include <hds/allocator/allocator.h>
#include <hds/linked_list/lock_linked_list.h>
#include <hds/threadgroup/threadgroup.h>
#include <set>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
__trap();
#else
exit(1);
#endif
}

#define ASSERT(x, y) if(!(x)) { printf("%s did not evaluate to true for i = %d\n", #x, (y)); error(); }

__global__ void single_thread_test(hds::linked_list<int, 2, hds::allocator::heap_allocator>* ll) {

  ll = new (ll) hds::linked_list<int, 2, hds::allocator::heap_allocator>(); 

  ASSERT(!ll->contains(1, hds::threadgroup::single_threadgroup{}), 1);

  for (int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->insert(r, hds::threadgroup::single_threadgroup{}), r);

    if (!ll->validate(hds::threadgroup::single_threadgroup{})) {
      error();
    }
  }
  
  for (int i = 0; i < 100; ++i) {
    int r = i;

    ASSERT(ll->remove(r, hds::threadgroup::single_threadgroup{}), r);

    if(!ll->validate(hds::threadgroup::single_threadgroup{})) {
      error();
    }
  }

  ll->~linked_list<int, 2, hds::allocator::heap_allocator>(); 
}

__global__ void warp_test(hds::linked_list<int, 32, hds::allocator::heap_allocator>* ll) {

  auto warp = hds::threadgroup::warp_threadgroup{};
  if (warp.is_leader()) {
    new (ll) hds::linked_list<int, 32, hds::allocator::heap_allocator>(); 
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
    ll->~linked_list<int, 32, hds::allocator::heap_allocator>(); 
  }
}

int main() {
  hds::linked_list<int, 2, hds::allocator::heap_allocator> ll;
  ASSERT(!ll.contains(1, hds::threadgroup::single_threadgroup{}), 1);

  std::set<int> reference;

  for(int i = 0; i < 100; ++i) {

    if(rand() % 2 == 0) {

      int r = rand();

      bool inserted = reference.insert(r).second;
      ASSERT(ll.insert(r, hds::threadgroup::single_threadgroup{}) == inserted, r);

      printf("\nInserted %d\n", r);
      ll.print(hds::threadgroup::single_threadgroup{});

    } else {

      int r = rand();

      bool removed = (reference.erase(r) == 1);

      ASSERT(ll.remove(r, hds::threadgroup::single_threadgroup{}) == removed, r);

      printf("\nRemoved %d\n", r);
      ll.print(hds::threadgroup::single_threadgroup{});

    }

    if(!ll.validate(hds::threadgroup::single_threadgroup{})) {
      return 1;
    }

    for(auto elm : reference) {
      ASSERT(ll.contains(elm, hds::threadgroup::single_threadgroup{}), elm);
    }
  }

  hds::allocator::device_allocator dev_mem;
  hds::linked_list<int, 2, hds::allocator::heap_allocator>* st_gpu_ll;
  st_gpu_ll = dev_mem.allocate<hds::linked_list<int, 2, hds::allocator::heap_allocator>>(1);

  single_thread_test<<<1, 1>>>(st_gpu_ll);

  auto err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }

  dev_mem.deallocate(st_gpu_ll, 1);

  hds::linked_list<int, 32, hds::allocator::heap_allocator>* w_gpu_ll;
  w_gpu_ll = dev_mem.allocate<hds::linked_list<int, 32, hds::allocator::heap_allocator>>(1);

  warp_test<<<1, 32>>>(w_gpu_ll);

  err = cudaDeviceSynchronize();

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorName(err)) + " : " + std::string(cudaGetErrorString(err)));
  }
  
  dev_mem.deallocate(w_gpu_ll, 1);

  return 0;
}

__launch_bounds__(1024, 1)
__global__ void warp_insert(hds::linked_list<int, 32, hds::allocator::heap_allocator>* ll, int i) {

  auto warp = hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->insert(i, warp);

}

__launch_bounds__(1024, 1)
__global__ void warp_remove(hds::linked_list<int, 32, hds::allocator::heap_allocator>* ll, int i) {

  auto warp = hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->remove(i, warp);

}

__global__ void warp_contains(hds::linked_list<int, 32, hds::allocator::heap_allocator>* ll, int i) {

  auto warp = hds::threadgroup::warp_threadgroup{};

  static_assert(decltype(warp)::size == 32);

  ll->contains(i, warp);
 
}
