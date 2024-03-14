#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>

#include <remus/hds/allocator/rdma_allocator.h>
#include <remus/hds/threadgroup/threadgroup.h>
#include <remus/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <remus/hds/unordered_map/kv_linked_list/locked_nodes/rdma_nodes.h>
#include <unordered_map>

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

std::string hostname() {
  char name[150];
  if (gethostname(name, 150) != 0) {
    throw std::runtime_error("Unable to get host name");
  }
  return std::string(name);
}

int main(int argc, char **argv) {

  REMUS_INIT_LOG();

  std::string name;
  if (argc == 2) {
    name = argv[1];
  } else {
    name = hostname();
  }

  std::cerr << "Using IP/name " << name << std::endl;

  std::vector<remus::rdma::Peer> peers;
  peers.push_back(remus::rdma::Peer(0, name));

  auto host = peers.at(0);

  REMUS_DEBUG("Creating pool");
  auto ctx = new remus::rdma::rdma_capability(host);
  ctx->init_pool(1 << 24, peers);
  ctx->RegisterThread(); // need to register thread?

  remus::hds::allocator::rdma_allocator alloc(ctx);
  remus::hds::kv_linked_list::locked_nodes::rdma_pointer_constructor constructor(ctx);

  using ll_t = remus::hds::kv_linked_list::kv_lock_linked_list<
    int, long, 10, remus::hds::kv_linked_list::locked_nodes::rdma_node_pointer, remus::hds::allocator::rdma_allocator,
    remus::hds::kv_linked_list::locked_nodes::rdma_pointer_constructor>;

  ll_t ll(alloc, constructor);

  auto group = remus::hds::threadgroup::single_threadgroup{};

  try {

    std::unordered_map<int, int> reference;

    for (int i = 0; i < 100; ++i) {

      if (rand() % 2 == 0) {

        int r = rand();

        bool inserted = reference.insert({r, 1}).second;
        ASSERT(ll.insert(r, 1, group) == inserted, r);

        // printf("\nInserted %d\n", r);
        // ll.print(remus::hds::threadgroup::single_threadgroup{});

      } else {

        int r = rand();

        bool removed = (reference.erase(r) == 1);

        ASSERT(ll.remove(r, group) == removed, r);

        // printf("\nRemoved %d\n", r);
        // ll.print(remus::hds::threadgroup::single_threadgroup{});
      }

      if (!ll.validate(group)) {
        return 1;
      }

      for (auto elm : reference) {
        auto res = ll.get(elm.first, group);
        ASSERT(res != remus::hds::nullopt and *res == elm.second, elm.first);
      }
    }

  } catch (const std::exception &e) {
    std::cout << "Failure" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  delete ctx;

  std::cout << "Success" << std::endl;

  return 1;
}
