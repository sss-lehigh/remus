#include <vector>
#include <string>
#include <iostream>
#include <barrier> 

#include <unistd.h>

#include <remus/logging/logging.h>
#include <remus/rdma/rdma.h>
#include <remus/util/cli.h>

#define PORT_NUM 8080

constexpr size_t MAX_MSG_SIZE = 256;
struct FixedMessage {
    char data[MAX_MSG_SIZE];
};

struct Params {
  uint32_t node_id;
  uint32_t node_count;
  uint32_t thread_count;
  uint32_t region_size;

  Params(remus::util::ArgMap args){
    node_id = args.iget("--node_id");
    node_count = args.iget("--node_count");
    thread_count = args.iget("--thread_count");
    region_size = args/iget("--region_size");
  }
}

auto Args = {
  remus::util::I64_ARG("--node_id", "This node's id."),
  remus::util::I64_ARG("--node_count", "How many nodes in the system."),
  remus::util::I64_ARG("--thread_count", "How many threads per node."),
  remus::util::I64_ARG("--region_size", "How big the region should be in 2^x bytes"),
};

int main(int argc, char** argv) {
  using namespace remus::rdma;
  using namespace std::string_literals;

  REMUS_INIT_LOG();

  remus::util::ArgMap args;
  if (auto res = args.import_args(ARGS); res) {
    REMUS_FATAL(res.value());
  }
  if (auto res = args.parse_args(argc, argv); res) {
    args.usage();
    REMUS_FATAL(res.value());
  }

  Params params(args);

  int mp = params.thread_count;
  REMUS_INFO("Distributing {} MemoryPools across {} threads", mp, params.thread_count);

  std::vector<Peer> peers;
  // Also create a set of peer ids that are local to this node
  std::unordered_set<int> locals;
  for (uint16_t i = 0; i < mp * params.node_count; i++) {
    Peer next(i, "node"s + std::to_string((int)i / mp), PORT_NUM + i + 1);
    peers.push_back(next);
    REMUS_DEBUG("Peer list {}:{}@{}", i, peers.at(i).id, peers.at(i).address);
    if((int)i/mp == params.node_id){
      REMUS_DEBUG("Peer {} is local on node {}", i, (int)i/mp);
      locals.insert(i+1);
    }
  }

  /* Initialize the memory pools */
  std::vector<std::thread> mempool_threads;
  std::shared_ptr<rdma_capability>* pools = new std::shared_ptr<rdma_capability>[params.thread_count];
  // Create one memory pool per thread
  uint32_t block_size = params.region_size << 10;
  for (int i = 0; i < params.thread_count; i++) {
    mempool_threads.emplace_back(std::thread(
      [&](int mp_index, int self_index) {
        Peer self = peers.at(self_index);
        REMUS_DEBUG("Creating pool for {}:{}@{}", self_index, self.id, self.address);
        std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(self);
        pool->init_pool(block_size, peers);
        REMUS_DEBUG("Created pool for {}:{}@{}", self_index, self.id, self.address);
        pools[mp_index] = pool;
      },
      i, i + id * params.thread_count));
  }

  // Let the init finish
  for (int i = 0; i < params.thread_count; i++) {
    mempool_threads[i].join();
  }

  /* Create and launch the clients */
  std::vector<std::thread> client_threads;
  client_threads.reserve(params.thread_count);
  std::barrier client_barrier(params.thread_count);
  std::vector<rdma_ptr<FixedMessage>> root_ptrs;
  root_ptrs.reserve(peers.size());

  for (int i = 0; i < params.thread_count; i++){
    client_threads.emplace_back(std::thread([&mp, &peers, &pools, &params, &locals, &client_barrier](int tidx){
      auto self_idx = (id * params.thread_count) + tidx;
      Peer self = peers.at(self_idx);
      auto pool = pools[tidx];
      std::vector<Peer> others;

      /* Include our own endpoint in the peers */
      REMUS_DEBUG("Including self in others for loopback connection");
      std::copy(peers.begin(), peers.end(), std::back_inserter(others));

      FixedMessage msg;
      std::string msg_str = "From node " + std::to_string(self_idx) + " in thread " + std::to_string(i) + ": Hello world!";
      /* Zero out the buffer */
      std::memset(msg.data, 0, MAX_MSG_SIZE);
      std::strncpy(msg.data, msg_str.c_str(), MAX_MSG_SIZE - 1);

      /* Allocate region in mp for this thread */
      rdma_ptr<FixedMessage> my_msg_ptr = pool->Allocate<FixedMessage>(peers.size());

      /* Establish connections using two-sided verbs */
      RemoteObjectProto proto;
      proto.set_raddr(my_msg_ptr);
      /* Send out the addres information using two-sided */
      for(auto p : peers){
        remus::util::Status status = pool->Send<RemoteObjectProto>(p, proto);
        OK_OR_FAIL(status);
        REMUS_DEBUG("Node {} sent msg pointer to node {}", self_.id, p.id);
      }
      client_barrier.arrive_and_wait();

      /* Wait for the root pointers to be shared with the other peers */
      for(auto p : peers){
        remus::util::StatusVal got = pool->Recv<RemoteObjectProto>(p);
        OK_OR_FAIL(got.status);
        rdma_ptr<FixedMessage> root = rdma_ptr<FixedMessage>(p.id, got.val->raddr());
        root_ptrs.at(p.id) = root;
      }
      client_barrier.arrive_and_wait();

      /* ----Begin usage of one-sided verbs---- */
      pool->RegisterThread();
      /* Write the messages */
      for(int j = 0; j < peers.size(); ++j){
        pool->Write(root_ptrs[j], msg);
      }
      client_barrier.arrive_and_wait();
      for(int j = 0; j < peers.size(); ++j){
        FixedMessage msg_read = pool->Read(my_msg_ptr + j);
        REMUS_INFO("Node {} thread {} has message[{}]", params.node_id, i, msg_read.data);
      }
      client_barrier.arrive_and_wait();
      /* ----End usage of one-sided verbs---- */

      REMUS_INFO("[CLIENT THREAD {}] -- End of Execution", self.id);
    }, i));
  }
  sleep(10); // TODO: Sleep here? 

  /* Clean up */
  REMUS_DEBUG("Deleting pools now");
  delete[] pools;

  return 0;
}

