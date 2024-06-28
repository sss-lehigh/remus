#include <vector>
#include <string>
#include <iostream>

#include <unistd.h>

#include <remus/logging/logging.h>
#include <remus/rdma/rdma.h>

void usage(char** argv) {
  std::cout << "Usage: " << argv[0] << " [-h] [-p <port>] -n <comma seperated nodes> [-t <threads>] -i <id>" << std::endl;
}

int main(int argc, char** argv) {

  using namespace remus::rdma;
  using namespace std::string_literals;

  int c;

  int port_num = 8080;
  std::string nodes_str = "";
  int id = 0;
  int threads = 1;

  while ((c = getopt(argc, argv, "p:n:t:i:h")) != -1) {
    switch (c) {
      case 'p':
        port_num = atoi(optarg);
        break;
      case 'n':
        nodes_str = std::string(optarg);
        break;
      case 'i':
        id = atoi(optarg);
        break;
      case 't':
        threads = atoi(optarg);
        break;
      case 'h':
        usage(argv);
        exit(0);
      default:
        usage(argv);
        exit(1);
    }
  }

  REMUS_INIT_LOG();

  // split nodes by ,

  std::vector<std::string> nodes;

  size_t offset = 0;
  while ((offset = nodes_str.find(",")) != std::string::npos) {
    nodes.push_back(nodes_str.substr(0, offset));
    nodes_str.erase(0, offset + 1);
  }
  nodes.push_back(nodes_str);

  for(auto n : nodes) {
    REMUS_DEBUG("Have node: {}", n);
  }

  std::vector<Peer> peers;
 
  int node_id = 0;
  for(auto n : nodes) {
    for(int tid = 0; tid < threads; ++tid) {
      Peer next(node_id, n, port_num + node_id + 1);
      peers.push_back(next);
      REMUS_DEBUG("Peer list {}:{}@{}", node_id, peers.at(node_id).id, peers.at(node_id).address);
      node_id++;
    }
  }

  std::vector<std::thread> mempool_threads;
  std::shared_ptr<rdma_capability>* pools = new std::shared_ptr<rdma_capability>[threads];

  // Create multiple memory pools to be shared (have to use threads since Init
  // is blocking)
  uint32_t block_size = 1 << 10;
  for (int i = 0; i < threads; i++) {
    mempool_threads.emplace_back(std::thread(
      [&](int mp_index, int self_index) {
        Peer self = peers.at(self_index);
        REMUS_DEBUG("Creating pool for {}:{}@{}", self_index, self.id, self.address);
        std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(self);
        pool->init_pool(block_size, peers);
        REMUS_DEBUG("Created pool for {}:{}@{}", self_index, self.id, self.address);
        pools[mp_index] = pool;
      },
      i, i + id * threads));
  }

  // Let the init finish
  for (int i = 0; i < threads; i++) {
    mempool_threads[i].join();
  }

  sleep(10);
  REMUS_DEBUG("Deleting pools now");

  delete[] pools;

  return 0;
}

