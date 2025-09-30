// [mfs] The only remaining issue in this code is that we need to do the whole
// "reclaim after barrier" thing...

#include <memory>
#include <unistd.h>
#include <vector>

#include <remus/remus.h>

#include "cloudlab.h"
#include "lf_list.h"
#include "manager.h"

using Key_t = uint64_t;
using Val_t = uint64_t;
using ds_t = LockFreeList<Key_t, Val_t>;
using ds_workload_t = ds_workload<ds_t, Key_t, Val_t>;

int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
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

  // If this is a compute node, create threads and run the experiment
  if (id >= c0 && id <= cn) {
    // Create ComputeThread contexts
    std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
    uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }

    // Compute Node 0 will construct the data structure and save it in root
    if (id == c0) {
      auto ds_ptr = ds_t::New(compute_threads[0]);
      compute_threads[0]->set_root(ds_ptr);
    }

    // Make threads and start them
    std::vector<std::thread> worker_threads;
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); i++) {
      worker_threads.push_back(std::thread(
          [&](uint64_t i) {
            auto &ct = compute_threads[i];
            // Wait, because this node might be ahead of the node initializing
            // the root:
            ct->arrive_control_barrier(total_threads);

            // Get the root, make a local reference to it
            auto ds_ptr = ct->get_root<ds_t>();
            ds_t ds_handle(ds_ptr); // calls the constructor for LockFreeList

            // Make a workload manager for this thread
            ds_workload_t workload(ds_handle, i, id, ct, args);
            ct->arrive_control_barrier(total_threads);

            // Prefill the data structure
            workload.prefill();
            ct->arrive_control_barrier(total_threads);

            // Get the starting time before any thread does any work
            std::chrono::high_resolution_clock::time_point start_time =
                std::chrono::high_resolution_clock::now();
            ct->arrive_control_barrier(total_threads);

            // Run the workload
            workload.run();
            ct->arrive_control_barrier(total_threads);

            // Compute the end time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

            // "lead thread" will reclaim the data structure and then put a
            // global metrics object into the root
            if (id == c0 && i == 0) {
              ds_handle.destroy(ct);
              auto metrics = ct->allocate<Metrics>();
              ct->Write(metrics, Metrics());
              ct->set_root(metrics);
            }
            ct->arrive_control_barrier(total_threads);

            // All threads aggregate metrics
            auto metrics = remus::rdma_ptr<Metrics>(ct->get_root<Metrics>());
            workload.collect(metrics);
            ct->arrive_control_barrier(total_threads);

            // First thread writes aggregated metrics to file
            if (id == c0 && i == 0) {
              compute_threads[0]->Read(metrics).to_file(duration,
                                                        compute_threads[0]);
            }
          },
          i));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }
};
