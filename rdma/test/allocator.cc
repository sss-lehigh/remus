#include <atomic>
#include <memory>
#include <thread>
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

  std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
  uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
  if (id >= c0 && id <= cn) {
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }
    REMUS_INFO("Starting concurrent allocation test");

    // Size of large memory to allocate per thread
    const size_t large_size = 10000; // 10KB per thread
    const size_t small_size = 100;   // 100B

    // Create threads to allocate and write concurrently
    std::vector<std::thread> worker_threads;

    // Launch worker threads
    for (auto &t : compute_threads) {
      worker_threads.push_back(std::thread([t, large_size, total_threads]() {
        // Wait for all threads to be ready
        t->arrive_control_barrier(total_threads);

        REMUS_INFO("Thread {} starting allocation of {}B", (uint64_t)t.get(),
                   large_size);

        // Allocate large memory
        auto ptr = t->allocate<char>(large_size);

        if (ptr == nullptr) {
          REMUS_FATAL("Thread {} failed to allocate memory", (uint64_t)t.get());
          return;
        }

        REMUS_INFO("Thread {} allocated at {}", (uint64_t)t.get(),
                   (uint64_t)ptr);

        // Write pattern to memory (offset)
        for (size_t j = 0; j < large_size; j++) {
          char value = static_cast<char>(j % 256);
          t->Write<char>(remus::rdma_ptr<char>(ptr.raw() + j), value);

          // Immediately read back to verify write consistency
          char readback = t->Read<char>(remus::rdma_ptr<char>(ptr.raw() + j));
          if (readback != value) {
            REMUS_FATAL("Thread {} immediate verification failed at offset {}: "
                        "wrote {}, read back {}",
                        (uint64_t)t.get(), j, (int)value, (int)readback);
            break;
          }
        }

        REMUS_INFO("Thread {} finished writing with immediate verification",
                   (uint64_t)t.get());

        // Verify entire pattern after all writes
        bool verify_error = false;
        for (size_t j = 0; j < large_size; j++) {
          char expected = static_cast<char>(j % 256);
          char actual = t->Read<char>(remus::rdma_ptr<char>(ptr.raw() + j));

          if (actual != expected) {
            REMUS_FATAL("Thread {} full verification failed at offset {}: "
                        "expected {}, got {}",
                        (uint64_t)t.get(), j, (int)expected, (int)actual);
            verify_error = true;
            break;
          }
        }

        if (verify_error) {
          REMUS_FATAL("Thread {} full verification failed", (uint64_t)t.get());
        } else {
          REMUS_INFO("Thread {} full verification successful",
                     (uint64_t)t.get());
        }

        // Deallocate memory
        t->deallocate<char>(ptr);
        REMUS_INFO("Thread {} deallocated memory", (uint64_t)t.get());

        // Reallocate the same size and verify we get the same pointer
        auto new_ptr = t->allocate<char>(large_size);

        if (new_ptr == nullptr) {
          REMUS_FATAL("Thread {} failed to reallocate memory",
                      (uint64_t)t.get());
          return;
        }

        // Check if the new pointer is the same as the old one
        if (new_ptr.raw() == ptr.raw()) {
          REMUS_INFO("Thread {} successfully reallocated at same address: {}",
                     (uint64_t)t.get(), (uint64_t)new_ptr);
        } else {
          REMUS_FATAL(
              "Thread {} reallocated at different address: old={}, new={}",
              (uint64_t)t.get(), (uint64_t)ptr, (uint64_t)new_ptr);
        }

        // Deallocate the new pointer
        t->deallocate<char>(new_ptr);
        REMUS_INFO("Thread {} deallocated reallocated memory",
                   (uint64_t)t.get());

        // Allocate small memory
        auto small_ptr = t->allocate<char>(small_size);
        if (small_ptr == nullptr) {
          REMUS_FATAL("Thread {} failed to allocate small memory",
                      (uint64_t)t.get());
          return;
        }

        // Write pattern to small memory
        for (size_t j = 0; j < small_size; j++) {
          char value = static_cast<char>(j % 256);
          t->Write<char>(remus::rdma_ptr<char>(small_ptr.raw() + j), value);

          // Immediate verification
          char actual =
              t->Read<char>(remus::rdma_ptr<char>(small_ptr.raw() + j));
          if (actual != value) {
            REMUS_FATAL("Thread {} small memory verification failed at offset "
                        "{}: expected {}, got {}",
                        (uint64_t)t.get(), j, (int)value, (int)actual);
            return;
          }
        }

        REMUS_INFO("Thread {} finished writing small memory with immediate "
                   "verification",
                   (uint64_t)t.get());

        // Verify entire pattern after all writes
        bool small_verify_error = false;
        for (size_t j = 0; j < small_size; j++) {
          char expected = static_cast<char>(j % 256);
          char actual =
              t->Read<char>(remus::rdma_ptr<char>(small_ptr.raw() + j));

          if (actual != expected) {
            REMUS_FATAL("Thread {} small memory full verification failed at "
                        "offset {}: expected {}, got {}",
                        (uint64_t)t.get(), j, (int)expected, (int)actual);
            small_verify_error = true;
            break;
          }
        }

        if (small_verify_error) {
          REMUS_FATAL("Thread {} small memory full verification failed",
                      (uint64_t)t.get());
        } else {
          REMUS_INFO("Thread {} small memory full verification successful",
                     (uint64_t)t.get());
        }

        // Deallocate small memory
        t->deallocate<char>(small_ptr);
        REMUS_INFO("Thread {} deallocated small memory", (uint64_t)t.get());

        // Reallocate the same size and verify we get the same pointer
        auto new_small_ptr = t->allocate<char>(small_size);

        if (new_small_ptr == nullptr) {
          REMUS_FATAL("Thread {} failed to reallocate small memory",
                      (uint64_t)t.get());
          return;
        }

        // Check if the new pointer is the same as the old one
        if (new_small_ptr.raw() == small_ptr.raw()) {
          REMUS_INFO("Thread {} successfully reallocated small memory at same "
                     "address: {}",
                     (uint64_t)t.get(), (uint64_t)new_small_ptr);
        } else {
          REMUS_FATAL("Thread {} reallocated small memory at different "
                      "address: old={}, new={}",
                      (uint64_t)t.get(), (uint64_t)small_ptr,
                      (uint64_t)new_small_ptr);
        }

        // Deallocate the new small pointer
        t->deallocate<char>(new_small_ptr);
        REMUS_INFO("Thread {} deallocated reallocated small memory",
                   (uint64_t)t.get());

        // Allocate small memory again for the final test
        small_ptr = t->allocate<char>(small_size);
        if (small_ptr == nullptr) {
          REMUS_FATAL("Thread {} failed to allocate small memory",
                      (uint64_t)t.get());
          return;
        }

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