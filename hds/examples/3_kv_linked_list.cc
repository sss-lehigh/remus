#include <cstdio>
#include <unordered_map>
#include <iostream>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/lazy_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <rome/hds/unordered_map/kv_linked_list/lazy_nodes/reg_cached_nodes.h>
#include <rome/hds/threadgroup/threadgroup.h>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
__trap();
#else
exit(1);
#endif
}

#define ASSERT(x, y) if(!(x)) { printf("%s did not evaluate to true for i = %d\n", #x, (y)); error(); }

int lazy() {
  using namespace rome::hds::kv_linked_list;

  auto group = rome::hds::threadgroup::single_threadgroup{};
  kv_lazy_linked_list<long, int, 2, lazy_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator> ll;
  ASSERT(ll.get(1, group) == rome::hds::nullopt, 1);

  std::unordered_map<long, int> reference;

  for(int i = 0; i < 100; ++i) {

    if(rand() % 2 == 0) {

      int r = rand();

      bool inserted = reference.insert({r, r}).second;
      ASSERT(ll.insert(r, r, group) == inserted, r);

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
      auto res = ll.get(elm.first, group);

      if(res == rome::hds::nullopt) {
        std::cerr << "Got null when querying " << elm.first << std::endl;
      } else if (*res != elm.second) {
        std::cerr << "Got " << *res << " when querying " << elm.first << " but expected " << elm.second << std::endl;
      }

      ASSERT(res != rome::hds::nullopt and *res == elm.second, static_cast<int>(elm.first));
    }
  }

  return 0;
}

int hoh() {

  using namespace rome::hds::kv_linked_list;

  auto group = rome::hds::threadgroup::single_threadgroup{};
  kv_lock_linked_list<long, int, 2, locked_nodes::reg_cached_node_pointer, rome::hds::allocator::heap_allocator> ll;
  ASSERT(ll.get(1, group) == rome::hds::nullopt, 1);

  std::unordered_map<long, int> reference;

  for(int i = 0; i < 100; ++i) {

    if(rand() % 2 == 0) {

      int r = rand();

      bool inserted = reference.insert({r, r}).second;
      ASSERT(ll.insert(r, r, group) == inserted, r);

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
      auto res = ll.get(elm.first, group);

      if(res == rome::hds::nullopt) {
        std::cerr << "Got null when querying " << elm.first << std::endl;
      } else if (*res != elm.second) {
        std::cerr << "Got " << *res << " when querying " << elm.first << " but expected " << elm.second << std::endl;
      }

      ASSERT(res != rome::hds::nullopt and *res == elm.second, static_cast<int>(elm.first));
    }
  }
  return 0;
}

int main() {
  if(lazy() != 0) return 1;
  return hoh();
}

