// [mfs] This is on track.  A few issues:
// 1. Make sure destructors are happening for everything that matters, including
//    RDMA resources
// 2. We need a few barriers and some simple one-sided tests.  Actually, a
//    barrier would be enough of a test.
// 3. How are we going to do graceful cleanup among memory nodes and compute
//    nodes?  We probably need memory nodes to park and wait somewhere, but
//    that's going to be tricky if they're also compute nodes.
// 4. Why do we have so many includes from remus?

#include <memory>
#include <unistd.h>
#include <vector>

#include <remus/cfg.h>
#include <remus/cli.h>
#include <remus/compute_node.h>
#include <remus/compute_thread.h>
#include <remus/logging.h>
#include <remus/mem_node.h>
#include <remus/util.h>

#include "cloudlab.h"

struct Counter {
  bool locked;
  uint64_t value;
};

int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->parse(argc, argv);

  // Extract the args we need in EVERY node
  uint64_t id = args->uget(remus::NODE_ID);
  uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  uint64_t mn = args->uget(remus::LAST_MN_ID);
  uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  uint64_t cn = args->uget(remus::LAST_CN_ID);

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

  // [mfs]  This is confusing to me.  I feel like we should be constructing
  //        std::threads here, and they should then:
  //          1. Create their own ComputeThread contexts
  //          2. Wait at a barrier
  //          3. Print a message
  //          4. Sleep for 2 seconds
  //          5. Wait at a barrier
  //          6. Exit
  //        Then the main thread should just join on all those threads

  // [mfs] Should this be a vector of shared ptrs or unique ptrs, instead of raw
  // pointers to ComputeThread?

  std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;

  if (id >= c0 && id <= cn) {
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }
    if (id == c0) {
      compute_threads[0]->set_root(remus::rdma_ptr<uint64_t>((uint64_t)0));
      REMUS_INFO("pass basic write test");
      compute_threads[0]->get_root<uint64_t>();
      REMUS_INFO("pass get read test");
      compute_threads[0]->cas_root(remus::rdma_ptr<uint64_t>((uint64_t)0), remus::rdma_ptr<uint64_t>((uint64_t)1));
      REMUS_INFO("pass cas test");
      compute_threads[0]->faa_root<uint64_t>(1);
      REMUS_INFO("pass faa test");
      auto ptr = compute_threads[0]->allocate<Counter>();
      REMUS_INFO("pass allocate test");
      compute_threads[0]->Write<uint64_t>(
          remus::rdma_ptr<uint64_t>(ptr.raw() + offsetof(Counter, value)), (uint64_t)0);
      compute_threads[0]->Write<bool>(
          remus::rdma_ptr<bool>(ptr.raw() + offsetof(Counter, locked)),
          false);
      compute_threads[0]->set_root(ptr);
      REMUS_INFO("pass init root test");
    }
    std::vector<std::thread> worker_threads;
    uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
    for (auto &t : compute_threads) {
      worker_threads.push_back(std::thread([t, total_threads]() {
        t->arrive_control_barrier(total_threads);
        REMUS_INFO("pass arrive_control_barrier test");
        auto root = t->get_root<Counter>();
        while (t->CompareAndSwap(remus::rdma_ptr<bool>(
                                     root.raw() + offsetof(Counter, locked)),
                                 false, true) != false)
          ;
        REMUS_INFO("pass cas bool test");
        t->Write<uint64_t>(
            remus::rdma_ptr<uint64_t>(root.raw() + offsetof(Counter, value)),
            t->Read<uint64_t>(remus::rdma_ptr<uint64_t>(
                root.raw() + offsetof(Counter, value))) +
                1);
        t->Write<bool>(
            remus::rdma_ptr<bool>(root.raw() + offsetof(Counter, locked)),
            false);
        t->arrive_control_barrier(total_threads);
        REMUS_INFO("thread {} arrive_control_barrierd at barrier again",
                   (uint64_t)t.get());
        
    }));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
    REMUS_INFO("pass root test");
  }
}
