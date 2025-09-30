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

#include <remus/remus.h>

#include "cloudlab.h"
#include "ds_workload.h"
#include "lf_list.h"

using Key_t = uint64_t;
using Val_t = uint64_t;
using ds_t = LockFreeList<Key_t, Val_t>;
using ds_workload_t = ds_workload<ds_t, Key_t, Val_t>;

int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->import(EXP_ARGS);
  args->import(DS_EXP_ARGS);
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

  std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
  uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
  if (id >= c0 && id <= cn) {
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }
    if (id == c0) {
      auto ds_ptr = ds_t::New(compute_threads[0]);
      compute_threads[0]->set_root(ds_ptr);
    }
    std::vector<std::thread> worker_threads;
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); i++) {
      worker_threads.push_back(std::thread(
          [&](uint64_t i) {
            compute_threads[i]->arrive_control_barrier(total_threads);
            auto ds_ptr = compute_threads[i]->get_root<ds_t>();
            // [mfs] We should get rid of ds_t for the tutorial
            ds_t ds_handle(ds_ptr); // calls the constructor for LockFreeList
            auto workload = std::make_shared<ds_workload_t>(
                ds_handle, i, id, compute_threads[i], args);
            compute_threads[i]->arrive_control_barrier(total_threads);
            workload->prefill();
            compute_threads[i]->arrive_control_barrier(total_threads);
            std::chrono::high_resolution_clock::time_point start_time;
            if (id == c0 && i == 0) {
              start_time = std::chrono::high_resolution_clock::now();
            }
            // [mfs] Need a barrier after reading the clock, because
            // conservative timing is necessary
            compute_threads[i]->arrive_control_barrier(total_threads);
            workload->run();
            compute_threads[i]->arrive_control_barrier(total_threads);
            if (id == c0 && i == 0) {
              auto end_time = std::chrono::high_resolution_clock::now();
              auto duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();
              // REMUS_INFO("thread {} run time: {} us",
              //            (uint64_t)compute_threads[i].get(), duration);
              auto metrics =
                  compute_threads[i]->allocate<ds_workload_t::Metrics>();
              compute_threads[i]->Write(metrics, ds_workload_t::Metrics());
              // [mfs] I don't really love it that we don't destruct the data
              //       structure before overwriting root...
              //
              // [mfs] There is a "destruct" method, so all we'd need to do is
              //       call that, and also reclaim the old root.  Or let
              //       "destruct" reclaim "This".
              compute_threads[i]->set_root(metrics);
              compute_threads[i]->arrive_control_barrier(total_threads);
              workload->collect(metrics);
              compute_threads[i]->arrive_control_barrier(total_threads);
              compute_threads[0]->Read(metrics).to_file(duration,
                                                        compute_threads[0]);
            } else {
              compute_threads[i]->arrive_control_barrier(total_threads);
              auto metrics = remus::rdma_ptr<ds_workload_t::Metrics>(
                  compute_threads[i]->get_root<ds_workload_t::Metrics>());
              workload->collect(metrics);
              compute_threads[i]->arrive_control_barrier(total_threads);
            }
          },
          i));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }
};
