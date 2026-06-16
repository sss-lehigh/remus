#include <atomic>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cloudlab.h"
#include "exp_cfg.h"
#include <remus/cfg.h>
#include <remus/cli.h>
#include <remus/compute_node.h>
#include <remus/compute_thread.h>
#include <remus/logging.h>
#include <remus/mem_node.h>
#include <remus/util.h>

struct alignas(64) KVPair {
  uint32_t key_;
  uint32_t value_;

  bool operator==(const KVPair &other) const {
    return key_ == other.key_ && value_ == other.value_;
  }
};

int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->import(EXP_ARGS);
  args->parse(argc, argv);

  // Extract the args we need in EVERY node
  uint64_t id = args->uget(remus::NODE_ID);
  uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  uint64_t mn = args->uget(remus::LAST_MN_ID);
  uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  uint64_t cn = args->uget(remus::LAST_CN_ID);
  uint64_t nelems = args->uget(NELEMS);

  // prepare network information about this machine and about memnodes
  remus::MachineInfo self(id, id_to_dns_name(id));
  std::vector<remus::MachineInfo> memnodes;
  for (uint64_t i = m0; i <= mn; ++i) {
    memnodes.emplace_back(i, id_to_dns_name(i));
  }

  // Information needed if this machine will operate as a memory node
  std::unique_ptr<remus::MemoryNode> memory_node;

  // Information needed if this machine will operate as a compute node
  std::shared_ptr<remus::ComputeNode> compute_node;

  // Memory Node configuration must come first!
  if (id >= m0 && id <= mn) {
    // Make the pools, await connections
    memory_node.reset(new remus::MemoryNode(self, args));
  }

  // Configure this to be a Compute Node?
  if (id >= c0 && id <= cn) {
    compute_node.reset(new remus::ComputeNode(self, args));
    // NB:  If this ComputeNode is also a MemoryNode, then we need to pass the
    //      rkeys to the local MemoryNode.  There's no harm in doing them first.
    if (memory_node.get() != nullptr) {
      auto rkeys = memory_node->get_local_rkeys();
      compute_node->connect_local(memnodes, rkeys);
    }
    compute_node->connect_remote(memnodes);
  }

  // If this is a memory node, pause until it has received all the connections
  // it's expecting, then spin until the control channel in each segment
  // becomes 1. Then, shutdown the memory node.
  if (memory_node) {
    memory_node->init_done();
  }

  std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
  uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
  if (id >= c0 && id <= cn) {
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }
    REMUS_INFO("Starting one-sided test");

    // Allocate nelems kvpairs
    if (id == c0) {
      remus::rdma_ptr<KVPair> ptr =
          compute_threads[0]->allocate<KVPair>(nelems);
      compute_threads[0]->set_root(ptr);
    }

    // Create threads to allocate and write concurrently
    std::vector<std::thread> worker_threads;

    // Launch worker threads
    for (auto &t : compute_threads) {
      worker_threads.push_back(std::thread([t, total_threads, nelems]() {
        // Wait for all threads to be ready
        t->arrive_control_barrier(total_threads);
        remus::rdma_ptr<KVPair> ptr = t->get_root<KVPair>();
        // Write the index into the buffer as the key and index + 1 as the value
        // for each KVPair

        for (uint64_t j = 0; j < nelems; j++) {
          KVPair kvp;
          kvp.key_ = j;
          kvp.value_ = j + 1;
          t->Write<KVPair>(ptr + j, kvp);
        }
        t->arrive_control_barrier(total_threads);

        t->arrive_control_barrier(total_threads);
      }));
    }

    // Wait for all threads to complete
    for (auto &t : worker_threads) {
      t.join();
    }
  }
  if (id == c0) {
    REMUS_INFO("Experiment Complete!");
    compute_node->send_shutdown();
    REMUS_INFO("Environment Cleaned Up!");
  }
  if (memory_node) {
    memory_node->await_shutdown();
    REMUS_INFO("Memory Node Shutdown!");
  }

  return 0;
}