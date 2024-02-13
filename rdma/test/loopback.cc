#include <unistd.h>
#include <sys/wait.h>

#include <cstdio>
#include <iostream>

#include <protos/workloaddriver.pb.h>
#include <rome/rdma/memory_pool.h>
#include <rome/logging/logging.h>
#include <rome/rdma/rdma.h>

int main() {
  //auto pid = fork();

  //if(pid == -1) {
  //  std::cerr << "Error creating process" << std::endl;
  //  perror("Error: ");
  //  return 1;
  //}

  bool child = false;//pid == 0;

  ROME_INIT_LOG();

  rome::rdma::Peer node0 = rome::rdma::Peer(0, "node0", (child ? 8081 : 8080));
  //rome::rdma::Peer node1 = rome::rdma::Peer(1, "127.0.0.1", (child ? 8080 : 8081));

  rome::rdma::Peer host = node0;

  rome::rdma::rdma_capability capability(host);
  std::vector<rome::rdma::Peer> peers = {node0};//, node1};
  capability.init_pool(4096, peers);

  //if (!child) {
  //  int status;
  //  wait(&status);
  //  if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
  //    std::cerr << "Child did not exit correctly";
  //    return 1; 
  //  } else if(WIFSIGNALED(status)) {
  //    psignal(WTERMSIG(status), "Exit signal");
  //    return 1;
  //  }
  //}

  return 0;

}

