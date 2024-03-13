#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>

#include <cstdio>
#include <cstring>
#include <iostream>

#include <rome/hds/allocator/rdma_allocator.h>
#include <rome/hds/unordered_map/unordered_map.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/rdma_nodes.h>
#include <rome/hds/threadgroup/threadgroup.h>
#include <unordered_map>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
__trap();
#else
exit(1);
#endif
}

#define ASSERT(x, y) if(!(x)) { printf("%s did not evaluate to true for i = %d\n", #x, (y)); error(); }

std::string hostname() {
  char name[150];
  if(gethostname(name, 150) != 0) {
    throw std::runtime_error("Unable to get host name"); 
  }
  return std::string(name);
}

int main(int argc, char** argv) {

  ROME_INIT_LOG();

  std::string name;
  if (argc == 2) {
    name = argv[1];
  } else {
    name = hostname();
  }

  std::cerr << "Using IP/name " << name << std::endl;

  std::vector<rome::rdma::Peer> peers;
  peers.push_back(rome::rdma::Peer(0, name));
 
  auto host = peers.at(0);

  ROME_DEBUG("Creating pool");
  auto ctx = new rome::rdma::rdma_capability(host);
  ctx->init_pool(1 << 24, peers);
  ctx->RegisterThread(); // need to register thread?

  rome::hds::allocator::rdma_allocator alloc(ctx);
  rome::hds::kv_linked_list::locked_nodes::rdma_pointer_constructor constructor(ctx);

  using map_t = rome::hds::unordered_map<int, 
                                        int, 
                                        10, 
                                        rome::hds::kv_linked_list::kv_lock_linked_list,
                                        rome::hds::kv_linked_list::locked_nodes::rdma_node_pointer, 
                                        rome::hds::allocator::rdma_allocator, 
                                        rome::hds::kv_linked_list::locked_nodes::rdma_pointer_constructor,
                                        rome::hds::allocator::heap_allocator>;

  map_t map(10000, alloc, rome::hds::allocator::heap_allocator{}, constructor);

  auto group = rome::hds::threadgroup::single_threadgroup{};

  try {

    std::unordered_map<int, int> reference;

    for(int i = 0; i < 100; ++i) {

      if(rand() % 2 == 0) {

        int r = rand();

        bool inserted = reference.insert({r, 1}).second;
        ASSERT(map.insert(r, 1, group) == inserted, r);

        //printf("\nInserted %d\n", r);
        //ll.print(rome::hds::threadgroup::single_threadgroup{});

      } else {

        int r = rand();

        bool removed = (reference.erase(r) == 1);

        ASSERT(map.remove(r, group) == removed, r);

        //printf("\nRemoved %d\n", r);
        //ll.print(rome::hds::threadgroup::single_threadgroup{});

      }

      for(auto elm : reference) {
        auto res = map.get(elm.first, group);
        ASSERT(res != rome::hds::nullopt and *res == elm.second, elm.first);
      }
    }


  } catch(const std::exception& e) {
    std::cout << "Failure" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  delete ctx;

  std::cout << "Success" << std::endl;

  return 1;
}

