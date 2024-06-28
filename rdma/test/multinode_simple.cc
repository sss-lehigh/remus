#include "remus/util/cli.h"
#include <vector>
#include <string>

#include <unistd.h>

#include <remus/logging/logging.h>
#include <remus/rdma/rdma.h>

// todo: replace multinode.cc with the updated API (simplying initialization)

using namespace remus::rdma;
using namespace remus::util;

auto ARGS = {
  I64_ARG_OPT("-t", "Number of threads", 1),
  I64_ARG_OPT("-p", "Starting port num to serve from", 8080),
  I64_ARG("-i", "Node ID"),
  STR_ARG("-n", "CSV of nodes to run with"),
  BOOL_ARG_OPT("--help", "Display help"),
  BOOL_ARG_OPT("-h", "Display help"),
};

struct alignas(64) BigInt {
  int x;
};

int main(int argc, char** argv) {
  using namespace std::string_literals;

  // Import arguments
  ArgMap args;
  if (auto res = args.import_args(ARGS)) {
    REMUS_FATAL(res.value());
  }
  if (auto res = args.parse_args(argc, argv)){
    args.usage();
    REMUS_FATAL(res.value());
  }
  if (args.bget("-h") || args.bget("--help")){
    // print help cleanly
    args.usage();
    return 0;
  }
  std::string nodes_str = args.sget("-n");
  int port_num = args.iget("-p");
  int threads = args.iget("-t");
  int id = args.iget("-i");

  args.report_config();

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
    Peer next = Peer(node_id, n, port_num + node_id + 1);
    peers.push_back(next);
    REMUS_DEBUG("Peer list {}@{}", node_id, peers.at(node_id).address);
    node_id++;
  }

  uint32_t block_size = 1 << 10;
  // Create a rdma capability with 2 sets of connections
  std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(peers.at(id), 2);
  pool->init_pool(block_size, peers); // init is blocking until connections are made. However, we hide the multiple initialization inside it
  rdma_capability_thread* tcapability1 = pool->RegisterThread();
  rdma_capability_thread* tcapability2 = pool->RegisterThread();
  rdma_capability_thread* tcapability3 = pool->RegisterThread();
  REMUS_ASSERT(tcapability1 == tcapability3 && tcapability1 == tcapability2, "Registration returns the same capability");

  rdma_ptr<BigInt> value = tcapability1->Allocate<BigInt>(1);
  value->x = 101;

  REMUS_ASSERT(tcapability1->is_local(value), "Created a local pointer");
  std::thread t1 = std::thread([&](){
    rdma_capability_thread* cap = pool->RegisterThread();
    if (cap != tcapability1){
      // we got a different capability than the master thread
      REMUS_ASSERT(cap->is_local(value), "A different capability still sees local as local pointer");
      auto tmp = cap->Read(value);
      REMUS_ASSERT(tmp->x == 101, "We got the correct value using a different memory pool");
      cap->Deallocate(tmp);
      // TEST we are able to deallocate using a different memory pool (memory resource is in node-scope, not thread scope)
      cap->Deallocate(value);
    } else {
      // we got the same capability as the master thread (only 2 allocated), so do nothing
    }
  });
  std::thread t2 = std::thread([&](){
    rdma_capability_thread* cap = pool->RegisterThread();
    if (cap != tcapability1){
      // we got a different capability than the master thread
      auto tmp = cap->Read(value);
      REMUS_ASSERT(tmp->x == 101, "We got the correct value using a different memory pool");
      cap->Deallocate(tmp);
      // TEST we are able to deallocate using a different memory pool (memory resource is in node-scope, not thread scope)
      cap->Deallocate(value);
    } else {
      // we got the same capability as the master thread (only 2 allocated), so do nothing
    }
  });
  t1.join();
  t2.join();

  REMUS_DEBUG("Deleting pools now");

  return 0;
}

