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

void sync_read(remus::rdma_ptr<size_t> ptr,
               std::shared_ptr<remus::SimpleAsyncComputeThread> t,
               size_t num_ops, std::vector<size_t> &result,
               size_t total_threads) {
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    result[i] = t->Read<size_t>(ptr);
  }
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(result[i] == 42, "value reat not match");
  }
  t->arrive_control_barrier(total_threads);
}

void sync_read_zero_copy(remus::rdma_ptr<size_t> ptr,
                         std::shared_ptr<remus::SimpleAsyncComputeThread> t,
                         size_t num_ops, std::vector<size_t> &result,
                         size_t total_threads) {
  t->arrive_control_barrier(total_threads);
  auto local_alloc = t->local_allocate<size_t>(num_ops);
  for (size_t i = 0; i < num_ops; i++) {
    t->Read<size_t>(ptr, local_alloc + i);
    result[i] = *(local_alloc + i);
  }
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(result[i] == 42, "value reat not match");
  }
  t->local_deallocate(local_alloc);
  t->arrive_control_barrier(total_threads);
}

void async_read(remus::rdma_ptr<size_t> ptr,
                std::shared_ptr<remus::SimpleAsyncComputeThread> t,
                size_t num_ops, std::vector<size_t> &result,
                size_t total_threads) {
  std::vector<remus::AsyncResult<size_t>> res;
  res.reserve(num_ops);
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    res.emplace_back(t->ReadAsync<size_t>(ptr));
  }
  for (size_t i = 0; i < num_ops; i++) {
    while (!res[i].get_ready()) {
      res[i].resume();
    }
    result[i] = res[i].get_value();
  }
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(result[i] == 42, "value reat not match");
  }
  t->arrive_control_barrier(total_threads);
}

void async_read_zero_copy(remus::rdma_ptr<size_t> ptr,
                          std::shared_ptr<remus::SimpleAsyncComputeThread> t,
                          size_t num_ops, std::vector<size_t> &result,
                          size_t total_threads) {
  std::vector<remus::AsyncResult<size_t>> res;
  res.reserve(num_ops);
  auto local_alloc = t->local_allocate<size_t>(num_ops);
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    t->ReadAsync<size_t>(ptr, local_alloc + i);
    result[i] = *(local_alloc + i);
  }
  for (size_t i = 0; i < num_ops; i++) {
    while (!res[i].get_ready()) {
      res[i].resume();
    }
    result[i] = res[i].get_value();
  }
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(result[i] == 42, "value reat not match");
  }
  t->local_deallocate(local_alloc);
  t->arrive_control_barrier(total_threads);
}

void read_seq(remus::rdma_ptr<size_t> ptr,
              std::shared_ptr<remus::SimpleAsyncComputeThread> t,
              size_t num_ops, size_t total_threads) {
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops - 1; i++) {
    t->ReadSeq<size_t>(ptr);
  }
  auto result = t->ReadSeq<size_t>(ptr, true, true).value();
  t->arrive_control_barrier(total_threads); 
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(result[i] == 42, "value reat not match");
  }
  t->arrive_control_barrier(total_threads);
}

void read_seq_zero_copy(remus::rdma_ptr<size_t> ptr,
                        std::shared_ptr<remus::SimpleAsyncComputeThread> t,
                        size_t num_ops, size_t total_threads) {
  auto local_alloc = t->local_allocate<size_t>(num_ops);
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops - 1; i++) {
    t->ReadSeq<size_t>(ptr, local_alloc + i);
  }
  t->ReadSeq<size_t>(ptr, local_alloc + num_ops - 1, true, true).value();
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < num_ops; i++) {
    REMUS_ASSERT(*(local_alloc + i) == 42, "value reat not match");
  }
  t->local_deallocate(local_alloc);
  t->arrive_control_barrier(total_threads);
}

void async_read_seq(remus::rdma_ptr<size_t> ptr,
                    std::shared_ptr<remus::SimpleAsyncComputeThread> t,
                    size_t num_ops, size_t num_groups, size_t total_threads) {
  std::vector<remus::AsyncResult<std::optional<std::vector<size_t>>>> res_seq;
  res_seq.reserve(num_groups + 1);
  size_t ops_per_group = num_ops / num_groups;
  size_t remaining_ops = num_ops % num_groups;
  t->arrive_control_barrier(total_threads);
  // handle remaining operations
  for (size_t group = 0; group < num_groups; group++) {
    // Do num_ops/10 - 1 non-signaled reads
    if (ops_per_group == 0) {
      break;
    }
    for (size_t j = 0; j < ops_per_group - 1; j++) {
      t->ReadSeqAsync<size_t>(ptr);
    }
    // Do 1 signaled read to complete the group
    res_seq.emplace_back(t->ReadSeqAsync<size_t>(ptr, true, true));
  }
  // Handle remaining operations
  if (remaining_ops > 0) {
    for (size_t j = 0; j < remaining_ops - 1; j++) {
      t->ReadSeqAsync<size_t>(ptr);
    }
    res_seq.emplace_back(t->ReadSeqAsync<size_t>(ptr, true, true));
  }

  // Wait for all groups to complete
  for (size_t i = 0; i < res_seq.size(); i++) {
    while (!res_seq[i].get_ready()) {
      res_seq[i].resume();
    }
  }

  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < res_seq.size(); i++) {
    auto result = res_seq[i].get_value().value();
    for (size_t j = 0; j < result.size(); j++) {
      REMUS_ASSERT(result[j] == 42, "value read not match");
    }
  }
  t->arrive_control_barrier(total_threads);
}

void async_read_seq_zero_copy(
    remus::rdma_ptr<size_t> ptr,
    std::shared_ptr<remus::SimpleAsyncComputeThread> t, size_t num_ops,
    size_t num_groups, size_t total_threads) {
  auto local_alloc = t->local_allocate<size_t>(num_ops);
  std::vector<remus::AsyncResult<std::optional<std::vector<size_t>>>> res_seq;
  res_seq.reserve(num_groups + 1);
  size_t ops_per_group = num_ops / num_groups;
  size_t remaining_ops = num_ops % num_groups;
  t->arrive_control_barrier(total_threads);
  // handle remaining operations
  for (size_t group = 0; group < num_groups; group++) {
    // Do num_ops/10 - 1 non-signaled reads
    if (ops_per_group == 0) {
      break;
    }
    for (size_t j = 0; j < ops_per_group - 1; j++) {
      t->ReadSeqAsync<size_t>(ptr, local_alloc + group * ops_per_group + j);
    }
    // Do 1 signaled read to complete the group
    res_seq.emplace_back(t->ReadSeqAsync<size_t>(
        ptr, local_alloc + group * ops_per_group + ops_per_group - 1, true,
        true));
  }
  // Handle remaining operations
  if (remaining_ops > 0) {
    for (size_t j = 0; j < remaining_ops - 1; j++) {
      t->ReadSeqAsync<size_t>(ptr,
                              local_alloc + num_groups * ops_per_group + j);
    }
    res_seq.emplace_back(t->ReadSeqAsync<size_t>(
        ptr, local_alloc + num_groups * ops_per_group + remaining_ops - 1, true,
        true));
  }
  // Wait for all groups to complete
  for (size_t i = 0; i < res_seq.size(); i++) {
    while (!res_seq[i].get_ready()) {
      res_seq[i].resume();
    }
  }
  t->arrive_control_barrier(total_threads);
  for (size_t i = 0; i < res_seq.size(); i++) {
    auto result = res_seq[i].get_value().value();
    for (size_t j = 0; j < result.size(); j++) {
      REMUS_ASSERT(result[j] == 42, "value read not match");
    }
  }
  t->local_deallocate(local_alloc);
  t->arrive_control_barrier(total_threads);
}


int main(int argc, char **argv) {
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->parse(argc, argv);

  // Extract the args we need in EVERY node
  size_t id = args->uget(remus::NODE_ID);
  size_t m0 = args->uget(remus::FIRST_MN_ID);
  size_t mn = args->uget(remus::LAST_MN_ID);
  size_t c0 = args->uget(remus::FIRST_CN_ID);
  size_t cn = args->uget(remus::LAST_CN_ID);

  // prepare network information about this machine and about memnodes
  remus::MachineInfo self(id, id_to_dns_name(id));
  std::vector<remus::MachineInfo> memnodes;
  for (size_t i = m0; i <= mn; ++i) {
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

  std::vector<std::shared_ptr<remus::SimpleAsyncComputeThread>> compute_threads;
  size_t total_threads = (cn - c0 + 1) * args->uget(remus::CN_THREADS);
  if (id >= c0 && id <= cn) {
    for (size_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::SimpleAsyncComputeThread>(id, compute_node,
                                                            args));
    }
    if (id == c0) {
      auto ptr = compute_threads[0]->allocate<size_t>();
      compute_threads[0]->Write<size_t>(ptr, 42);
      compute_threads[0]->set_root(ptr);
      REMUS_ASSERT(compute_threads[0]->no_leak_detected(), "Leak detected");
    }
    const size_t num_ops = 256;
    std::vector<std::thread> worker_threads;
    for (auto &t : compute_threads) {
      worker_threads.push_back(std::thread([&]() {
        t->arrive_control_barrier(total_threads);
        auto root = t->get_root<size_t>();
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        std::vector<size_t> result_sync(num_ops);
        sync_read(root, t, num_ops, result_sync, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        std::vector<size_t> result_async(num_ops);
        async_read(root, t, num_ops, result_async, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        read_seq(root, t, num_ops, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        read_seq_zero_copy(root, t, num_ops, total_threads);
        REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        for (size_t num_groups = 1; num_groups <= num_ops; num_groups *= 4) {
            async_read_seq(root, t, num_ops, num_groups, total_threads);
            REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
            async_read_seq_zero_copy(root, t, num_ops, num_groups, total_threads);
            REMUS_ASSERT(t->no_leak_detected(), "Leak detected");
        }
      }));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }
  REMUS_INFO("Read test passed");
}
