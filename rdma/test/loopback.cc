#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>

#include <protos/workloaddriver.pb.h>
#include <remus/logging/logging.h>
#include <remus/rdma/memory_pool.h>
#include <remus/rdma/rdma.h>

int main() {

  bool child = false;

  REMUS_INIT_LOG();

  remus::rdma::Peer node0 = remus::rdma::Peer(0, "node0", (child ? 8081 : 8080));

  remus::rdma::Peer host = node0;

  remus::rdma::rdma_capability capability(host);
  std::vector<remus::rdma::Peer> peers = {node0};
  capability.init_pool(4096, peers);

  return 0;
}
