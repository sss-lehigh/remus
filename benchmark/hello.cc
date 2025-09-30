#include <memory>
#include <remus/remus.h>
#include <vector>

#include "cloudlab.h"

/// An object that we'll save in remote memory.  It's just an array :)
struct SharedObject {
  uint64_t values[1024];
};

int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->parse(argc, argv);
  if (args->bget(remus::HELP)) {
    args->usage();
    return 0;
  }
  args->report_config();

  // Extract the args for determining names and roles
  uint64_t id = args->uget(remus::NODE_ID);
  uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  uint64_t mn = args->uget(remus::LAST_MN_ID);
  uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  uint64_t cn = args->uget(remus::LAST_CN_ID);
  uint64_t threads = args->uget(remus::CN_THREADS);

  // prepare network information about this machine and about memnodes
  std::vector<remus::MachineInfo> memnodes;
  for (uint64_t i = m0; i <= mn; ++i) {
    memnodes.emplace_back(i, id_to_dns_name(i));
  }

  // Compute the name of this machine
  remus::MachineInfo self(id, id_to_dns_name(id));

  // Configure as a MemoryNode?
  std::unique_ptr<remus::MemoryNode> memory_node;
  if (id >= m0 && id <= mn) {
    memory_node.reset(new remus::MemoryNode(self, args));
  }

  // Configure as a ComputeNode?
  std::shared_ptr<remus::ComputeNode> compute_node;
  if (id >= c0 && id <= cn) {
    compute_node.reset(new remus::ComputeNode(self, args));
    // NB:  If this ComputeNode is also a MemoryNode, then we need to pass the
    //      rkeys to the local MemoryNode.  There's no harm in doing them first.
    if (memory_node.get() != nullptr) {
      compute_node->connect_local(memnodes, memory_node->get_local_rkeys());
    }
    compute_node->connect_remote(memnodes);
  }

  // Reclaim threads when all connections have been made
  if (memory_node) {
    memory_node->init_done();
  }

  if (id >= c0 && id <= cn) {
    // Create a context for each Compute Thread
    std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
    for (uint64_t i = 0; i < threads; ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }

    // The process is still sequential.  The main thread on Compute Node 0 will
    // create the shared object, using one of the available ComputeThread
    // objects:
    if (id == c0) {
      auto ptr = compute_threads[0]->allocate<SharedObject>();
      for (unsigned i = 0; i < 1024; ++i) {
        // Since the memory isn't local, we need to initialize it explicitly
        compute_threads[0]->Write<uint64_t>(
            remus::rdma_ptr<uint64_t>(ptr.raw() +
                                      offsetof(SharedObject, values[i])),
            (uint64_t)0);
      }
      // Make the SharedObject visible through the global root pointer, which is
      // at Memory Node 0, Segment 0.
      compute_threads[0]->set_root(ptr);
    }

    // Barriers will need to know the total number of threads across all
    // machines
    uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);

    // Create threads on this ComputeNode.  The main thread will stop doing any
    // work now.
    std::vector<std::thread> local_threads;
    for (uint64_t i = 0; i < threads; ++i) {
      uint64_t local_id = i;
      uint64_t global_thread_id = id * threads + i;
      auto t = compute_threads[i];
      local_threads.push_back(std::thread([t, global_thread_id, total_threads,
                                           local_id, id]() {
        // All threads, on all machine, synchronize here.  This ensures the
        // initialization is done and the root is updated before any thread
        // on any machine passes the next line.
        t->arrive_control_barrier(total_threads);
        auto root = t->get_root<SharedObject>();

        // Read from a unique location in SharedObject, ensure it's 0
        auto my_loc = remus::rdma_ptr<uint64_t>(
            root.raw() + offsetof(SharedObject, values[global_thread_id + 1]));
        uint64_t my_val = t->Read<uint64_t>(my_loc);
        if (my_val != 0) {
          REMUS_FATAL("Thread {}({}:{}) observed {}", global_thread_id, id,
                      local_id, my_val);
        }

        // Write a unique value to that unique location
        t->Write<uint64_t>(my_loc, global_thread_id);

        // Use a CompareAndSwap to increment it by 1
        bool success =
            global_thread_id ==
            t->CompareAndSwap(my_loc, global_thread_id, global_thread_id + 1);
        if (!success) {
          uint64_t v = t->Read<uint64_t>(my_loc);
          REMUS_FATAL("Thread {}({}:{}) CAS failed (observed {} at 0x{:x})",
                      global_thread_id, id, local_id, v, my_loc.raw());
        }

        // Wait for all threads to finish their work
        t->arrive_control_barrier(total_threads);

        // Now thread 0 can check everything:
        if (global_thread_id == 0) {
          for (uint64_t i = 1; i < total_threads + 1; ++i) {
            auto found = t->Read<uint64_t>(remus::rdma_ptr<uint64_t>(
                root.raw() + offsetof(SharedObject, values[i])));
            if (found != i) {
              REMUS_FATAL("In position {}, expected {}, found {}", i, i, found);
            }
            REMUS_INFO("All checks succeeded!");
          }
        }

        // Reclaim the object before terminating
        if (global_thread_id == 0) {
          t->deallocate(root);
          t->set_root(remus::rdma_ptr<SharedObject>(nullptr));
        }
      }));
    }
    for (auto &t : local_threads) {
      t.join();
    }
  }
}
