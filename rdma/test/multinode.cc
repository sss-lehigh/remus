#include <memory>
#include <unistd.h>
#include <vector>

// [mfs] Should remus expose a single include for all public functionality?

#include <remus/cfg.h>
#include <remus/cli.h>
#include <remus/compute_node.h>
#include <remus/compute_thread.h>
#include <remus/logging.h>
#include <remus/mem_node.h>
#include <remus/util.h>

#include "cloudlab.h"

int main(int argc, char **argv) {
  // Configure logging
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->parse(argc, argv);
  args->report_config();

  // Extract the args we need in EVERY node
  uint64_t id = args->uget(remus::NODE_ID);
  uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  uint64_t mn = args->uget(remus::LAST_MN_ID);
  uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  uint64_t cn = args->uget(remus::LAST_CN_ID);

  // prepare network information about this machine and about the memory nodes
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
    // Make the ComputeNode, then reach out to all memory nodes to get their
    // rkeys
    compute_node.reset(new remus::ComputeNode(self, args));
    // NB:  If this ComputeNode is also a MemoryNode, then we need to give it
    //      the rkeys of the local MemoryNode.  Order doesn't matter, so we do
    //      them first.
    if (memory_node.get() != nullptr) {
      auto rkeys = memory_node->get_local_rkeys();
      compute_node->connect_local(memnodes, rkeys);
    }
    compute_node->connect_remote(memnodes);
  }

  // If this is a memory node, pause until it has received all the connections
  // it's expecting
  if (memory_node) {
    memory_node->init_done();
  }

  // At this point, everything is configured!  If this is a compute node, make
  // some threads and have them use RDMA
  if (id >= c0 && id <= cn) {
    // Threads at this node
    const uint64_t cn_threads = args->uget(remus::CN_THREADS);
    // Total threads in the experiment (all must reach the barriers)
    const uint64_t barrier_thread_count = (cn - c0 + 1) * cn_threads;

    std::vector<std::thread> worker_threads;
    for (uint64_t i = 0; i < cn_threads; ++i) {
      worker_threads.push_back(std::thread([&]() {
        // Create thread context
        auto t = std::make_unique<remus::ComputeThread>(id, compute_node, args);
        // Wait for all threads on all nodes to have thread contexts.
        // Implicitly, passing the barrier means all Compute and Memory nodes
        // are done with configuration
        t->arrive_control_barrier(barrier_thread_count);

        // You could do work here :)
        sleep(2);

        // Don't gather stats until everyone finishes the experiment
        t->arrive_control_barrier(barrier_thread_count);

        // You could gather stats here.  Include MemoryNode control blocks and
        // also ComputeNode stats.

        // Don't shut down any thread until everyone finishes gathering stats
        t->arrive_control_barrier(barrier_thread_count);
      }));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }

  // All threads are now shut down, so the main ComputeNode can tell the
  // MemoryNodes to shut down.
  if (id == c0) {
    REMUS_INFO("Experiment Complete!");
    compute_node->send_shutdown();
    REMUS_INFO("Environment Cleaned Up!");
  }

  // Memory nodes won't exit the experiment until they receive the shutdown
  // signal from the main thread of the first ComputeNode
  if (memory_node) {
    memory_node->await_shutdown();
    REMUS_INFO("Memory Node Shutdown!");
  }

  // Since ComputeNode is a smart pointer, it will destruct as main exits.

  return 0;
}
