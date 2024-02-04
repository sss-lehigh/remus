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

  return 0;
}

