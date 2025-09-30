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
#include <remus/logging.h>
#include <remus/mem_node.h>
#include <remus/simple_async_compute_thread.h>
#include <remus/util.h>

#include "cloudlab.h"
void init_write(std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
                const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
                size_t total_threads) {
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    compute_thread->Write<uint64_t>(ptr + i, (uint64_t)0);
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == (uint64_t)0,
                 "Write value mismatch");
  }
  compute_thread->arrive_control_barrier(total_threads);
}

void sync_write(std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
                const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
                size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    compute_thread->Write<uint64_t>(ptr + i, i);
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->arrive_control_barrier(total_threads);
}

void sync_write_zero_copy(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
    size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  compute_thread->arrive_control_barrier(total_threads);
  auto local_alloc = compute_thread->local_allocate<uint64_t>(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    *(local_alloc + i) = i;
    compute_thread->Write<uint64_t>(ptr + i, local_alloc + i);
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->local_deallocate(local_alloc);
  compute_thread->arrive_control_barrier(total_threads);
}
void async_write(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
    size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  std::vector<remus::AsyncResultVoid> res;
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    res.emplace_back(compute_thread->WriteAsync<uint64_t>(ptr + i, i));
  }
  for (size_t i = 0; i < num_ops; i++) {
    while (!res[i].get_ready()) {
      res[i].resume();
    }
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->arrive_control_barrier(total_threads);
}
void async_write_zero_copy(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
    size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  std::vector<remus::AsyncResultVoid> res;
  compute_thread->arrive_control_barrier(total_threads);
  auto local_alloc = compute_thread->local_allocate<uint64_t>(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    *(local_alloc + i) = i;
    res.emplace_back(
        compute_thread->WriteAsync<uint64_t>(ptr + i, local_alloc + i));
  }
  for (size_t i = 0; i < num_ops; i++) {
    while (!res[i].get_ready()) {
      res[i].resume();
    }
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->local_deallocate(local_alloc);
  compute_thread->arrive_control_barrier(total_threads);
}
void write_seq(std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
               const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
               size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops - 1; i++) {
    compute_thread->WriteSeq<uint64_t>(ptr + i, i);
  }
  compute_thread->WriteSeq<uint64_t>(ptr + num_ops - 1, num_ops - 1, true,
                                     true);
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->arrive_control_barrier(total_threads);
}
void write_seq_zero_copy(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, remus::rdma_ptr<uint64_t> ptr,
    size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  auto local_alloc = compute_thread->local_allocate<uint64_t>(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    *(local_alloc + i) = i;
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops - 1; i++) {
    compute_thread->WriteSeq<uint64_t>(ptr + i, local_alloc + i);
  }
  compute_thread->WriteSeq<uint64_t>(ptr + num_ops - 1, num_ops - 1, true,
                                     true);
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->local_deallocate(local_alloc);
  compute_thread->arrive_control_barrier(total_threads);
}
void async_write_seq(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, const uint64_t num_groups,
    remus::rdma_ptr<uint64_t> ptr, size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  std::vector<remus::AsyncResult<std::optional<std::vector<uint64_t>>>> res_seq;
  res_seq.reserve(num_groups + 1);
  size_t ops_per_group = num_ops / num_groups;
  size_t remaining_ops = num_ops % num_groups;
  compute_thread->arrive_control_barrier(total_threads);
  // handle remaining operations
  for (size_t group = 0; group < num_groups; group++) {
    // Do num_ops/10 - 1 non-signaled reads
    if (ops_per_group == 0) {
      break;
    }
    for (size_t j = 0; j < ops_per_group - 1; j++) {
      compute_thread->WriteSeqAsync<uint64_t>(ptr + j + group * ops_per_group,
                                              j + group * ops_per_group);
    }
    // Do 1 signaled read to complete the group
    res_seq.emplace_back(compute_thread->WriteSeqAsync<uint64_t>(
        ptr + ops_per_group - 1 + group * ops_per_group,
        ops_per_group - 1 + group * ops_per_group, true, true));
  }
  // Handle remaining operations
  if (remaining_ops > 0) {
    for (size_t j = 0; j < remaining_ops - 1; j++) {
      compute_thread->WriteSeqAsync<uint64_t>(
          ptr + j + num_groups * ops_per_group, j + num_groups * ops_per_group);
    }
    res_seq.emplace_back(compute_thread->WriteSeqAsync<uint64_t>(
        ptr + remaining_ops - 1 + num_groups * ops_per_group,
        remaining_ops - 1 + num_groups * ops_per_group, true, true));
  }
  // Wait for all groups to complete
  for (uint64_t i = 0; i < res_seq.size(); i++) {
    while (!res_seq[i].get_ready()) {
      res_seq[i].resume();
    }
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->arrive_control_barrier(total_threads);
}

void async_write_seq_zero_copy(
    std::shared_ptr<remus::SimpleAsyncComputeThread> compute_thread,
    const uint64_t num_ops, const uint64_t num_groups,
    remus::rdma_ptr<uint64_t> ptr, size_t total_threads) {
  init_write(compute_thread, num_ops, ptr, total_threads);
  std::vector<remus::AsyncResult<std::optional<std::vector<uint64_t>>>> res_seq;
  res_seq.reserve(num_groups + 1);
  size_t ops_per_group = num_ops / num_groups;
  size_t remaining_ops = num_ops % num_groups;
  auto local_alloc = compute_thread->local_allocate<uint64_t>(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    *(local_alloc + i) = i;
  }
  compute_thread->arrive_control_barrier(total_threads);
  // handle remaining operations
  for (size_t group = 0; group < num_groups; group++) {
    // Do num_ops/10 - 1 non-signaled reads
    if (ops_per_group == 0) {
      break;
    }
    for (size_t j = 0; j < ops_per_group - 1; j++) {
      compute_thread->WriteSeqAsync<uint64_t>(ptr + j + group * ops_per_group,
                                              local_alloc + j +
                                                  group * ops_per_group);
    }
    // Do 1 signaled read to complete the group
    res_seq.emplace_back(compute_thread->WriteSeqAsync<uint64_t>(
        ptr + ops_per_group - 1 + group * ops_per_group,
        local_alloc + ops_per_group - 1 + group * ops_per_group, true, true));
  }
  // Handle remaining operations
  if (remaining_ops > 0) {
    for (size_t j = 0; j < remaining_ops - 1; j++) {
      compute_thread->WriteSeqAsync<uint64_t>(
          ptr + j + num_groups * ops_per_group,
          local_alloc + j + num_groups * ops_per_group);
    }
    res_seq.emplace_back(compute_thread->WriteSeqAsync<uint64_t>(
        ptr + remaining_ops - 1 + num_groups * ops_per_group,
        local_alloc + remaining_ops - 1 + num_groups * ops_per_group, true,
        true));
  }
  // Wait for all groups to complete
  for (uint64_t i = 0; i < res_seq.size(); i++) {
    while (!res_seq[i].get_ready()) {
      res_seq[i].resume();
    }
  }
  compute_thread->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(compute_thread->Read<uint64_t>(ptr + i) == i,
                 "Write value mismatch");
  }
  compute_thread->local_deallocate(local_alloc);
  compute_thread->arrive_control_barrier(total_threads);
}
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

  std::vector<std::shared_ptr<remus::SimpleAsyncComputeThread>> compute_threads;
  uint64_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
  if (id >= c0 && id <= cn) {
    const int num_ops = 256;
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::SimpleAsyncComputeThread>(id, compute_node,
                                                            args));
    }

    if (id == c0) {
      auto ptr = compute_threads[0]->allocate<uint64_t>(num_ops);
      for (size_t i = 0; i < num_ops; i++) {
        compute_threads[0]->Write<uint64_t>(ptr + i, (uint64_t)0);
        REMUS_ASSERT(compute_threads[0]->Read<uint64_t>(ptr + i) == (uint64_t)0,
                     "Write value mismatch");
      }
      compute_threads[0]->set_root(ptr);
      REMUS_ASSERT(compute_threads[0]->no_leak_detected(), "Leak detected");
    }
    std::vector<std::thread> worker_threads;
    for (auto &t : compute_threads) {
      worker_threads.push_back(std::thread([&]() {
        t->arrive_control_barrier(total_threads);
        auto root = t->get_root<uint64_t>();
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        sync_write(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        sync_write_zero_copy(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        async_write(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        async_write_zero_copy(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        write_seq(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        write_seq_zero_copy(t, num_ops, root, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        for (size_t num_groups = 1; num_groups <= num_ops; num_groups *= 4) {
          async_write_seq(t, num_ops, num_groups, root, total_threads);
          REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
          async_write_seq_zero_copy(t, num_ops, num_groups, root,
                                    total_threads);
          REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        }
      }));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }
  REMUS_INFO("Write test passed");
}
