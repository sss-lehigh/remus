#include <unistd.h>
#include <sys/wait.h>

#include <cstdio>
#include <cstring>
#include <iostream>

#include <rome/hds/allocator/rdma_allocator.h>
#include <rome/hds/linked_list/lock_linked_list.h>
#include <rome/hds/linked_list/locked_nodes/rdma_nodes.h>
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
  rome::hds::locked_nodes::rdma_pointer_constructor constructor(ctx);

  using ll_t = rome::hds::lock_linked_list<int, 
                                           10, 
                                           rome::hds::locked_nodes::rdma_node_pointer, 
                                           rome::hds::allocator::rdma_allocator, 
                                           rome::hds::locked_nodes::rdma_pointer_constructor>;

  ll_t ll(alloc, constructor);

  auto group = rome::hds::threadgroup::single_threadgroup{};

  try {

    std::set<int> reference;

    for(int i = 0; i < 100; ++i) {

      if(rand() % 2 == 0) {

        int r = rand();

        bool inserted = reference.insert(r).second;
        ASSERT(ll.insert(r, group) == inserted, r);

        //printf("\nInserted %d\n", r);
        //ll.print(hds::threadgroup::single_threadgroup{});

      } else {

        int r = rand();

        bool removed = (reference.erase(r) == 1);

        ASSERT(ll.remove(r, group) == removed, r);

        //printf("\nRemoved %d\n", r);
        //ll.print(hds::threadgroup::single_threadgroup{});

      }

      if(!ll.validate(group)) {
        return 1;
      }

      for(auto elm : reference) {
        ASSERT(ll.contains(elm, group), elm);
      }
    }


  } catch(const std::exception& e) {
    std::cout << "Failure" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  delete ctx;

  std::cout << "Success" << std::endl;

  return 0;
}

