#include <cstdio>

#include <remus/hds/allocator/allocator.h>
#include <remus/hds/linked_list/lock_linked_list.h>
#include <remus/hds/linked_list/locked_nodes/reg_cached_nodes.h>
#include <remus/hds/threadgroup/threadgroup.h>
#include <set>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
  __trap();
#else
  exit(1);
#endif
}

#define ASSERT(x, y)                                                                                                   \
  if (!(x)) {                                                                                                          \
    printf("%s did not evaluate to true for i = %d\n", #x, (y));                                                       \
    error();                                                                                                           \
  }

int main() {

  auto group = remus::hds::threadgroup::single_threadgroup{};
  remus::hds::lock_linked_list<int, 2, remus::hds::locked_nodes::reg_cached_node_pointer,
                               remus::hds::allocator::heap_allocator>
    ll;
  ASSERT(!ll.contains(1, group), 1);

  std::set<int> reference;

  for (int i = 0; i < 100; ++i) {

    if (rand() % 2 == 0) {

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

    if (!ll.validate(group)) {
      return 1;
    }

    for (auto elm : reference) {
      ASSERT(ll.contains(elm, group), elm);
    }
  }

  return 0;
}
