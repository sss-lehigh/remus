#include <cstdio>
#include <iostream>
#include <unordered_map>

#include <remus/hds/allocator/allocator.h>
#include <remus/hds/threadgroup/threadgroup.h>
#include <remus/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <remus/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <remus/hds/unordered_map/unordered_map.h>

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

  using namespace remus::hds::kv_linked_list;

  auto group = remus::hds::threadgroup::single_threadgroup{};

  using ll_t = remus::hds::unordered_map<long, int, 2, kv_lock_linked_list, locked_nodes::reg_cached_node_pointer,
                                         remus::hds::allocator::heap_allocator>;

  ll_t ll(100);
  ASSERT(ll.get(1, group) == remus::hds::nullopt, 1);

  std::unordered_map<long, int> reference;

  for (int i = 0; i < 100; ++i) {

    if (rand() % 2 == 0) {

      int r = rand();

      bool inserted = reference.insert({r, r}).second;
      ASSERT(ll.insert(r, r, group) == inserted, r);

      printf("\nInserted %d\n", r);

    } else {

      int r = rand();

      bool removed = (reference.erase(r) == 1);

      ASSERT(ll.remove(r, group) == removed, r);

      printf("\nRemoved %d\n", r);
    }

    for (auto elm : reference) {
      auto res = ll.get(elm.first, group);

      if (res == remus::hds::nullopt) {
        std::cerr << "Got null when querying " << elm.first << std::endl;
      } else if (*res != elm.second) {
        std::cerr << "Got " << *res << " when querying " << elm.first << " but expected " << elm.second << std::endl;
      }

      ASSERT(res != remus::hds::nullopt and *res == elm.second, static_cast<int>(elm.first));
    }
  }

  return 0;
}