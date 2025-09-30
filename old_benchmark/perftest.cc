#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <unordered_map>
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
#include "exp_cfg.h"

enum Operation { Write, Read, CAS, FAA };

std::unordered_map<Operation, std::string> op_name = {
    {Operation::Write, "Write"},
    {Operation::Read, "Read"},
    {Operation::CAS, "CAS"},
    {Operation::FAA, "FAA"}};

void metrics(std::string exp_name, uint64_t nnodes, uint64_t nthreads,
             uint64_t ops_per_thread, std::chrono::microseconds duration,
             std::string exp_op, uint64_t zero_copy) {
  // Use std::ios::out instead of std::ios::app to overwrite
  std::ofstream file("metrics.txt", std::ios::out);
  file << "Experiment: " << exp_name << std::endl;
  // TODO: `TYPE_SIZE` is a rather magical thing.  Document it better?
  file << "TypeSize: " << TYPE_SIZE << std::endl;
  file << "OpType: " << exp_op << std::endl;
  file << "ZeroCopy: " << zero_copy << std::endl;
  file << "Nodes: " << nnodes << std::endl;
  file << "Threads: " << nthreads << std::endl;

  // Calculate total operations
  uint64_t total_ops = ops_per_thread * nthreads * nnodes;

  // Throughput: ops/sec = (total_ops * 1,000,000) / duration_in_microseconds
  file << "Throughput(ops/sec): " << (total_ops * 1000000) / duration.count()
       << std::endl;

  // Bandwidth: MB/sec = (total_ops * bytes_per_op) / (duration_in_microseconds
  // * 1,000,000) * 1,000,000 / (1024*1024) Simplified: (total_ops * TYPE_SIZE)
  // / duration.count() / (1024*1024) But since we want MB/sec directly:
  file << "Bandwidth(MB/sec): "
       << (static_cast<double>(total_ops) * TYPE_SIZE) /
              (duration.count() * 1.048576)
       << std::endl;

  // Latency: microseconds per operation = duration_in_microseconds / ops
  if (exp_op == op_name[Operation::Write]) {
    file << "Latency(us): "
         << static_cast<double>(duration.count()) / ops_per_thread / 2
         << std::endl;
  } else {
    file << "Latency(us): "
         << static_cast<double>(duration.count()) / ops_per_thread << std::endl;
  }
};

union alignas(64) Type {
  uint8_t padding[TYPE_SIZE];
  uint64_t value;
};

int main(int argc, char **argv) {
  // Configure logging
  remus::INIT();

  // Configure and parse the arguments
  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->import(EXP_ARGS);
  args->parse(argc, argv);
  args->report_config();

  // Extract the args we need in EVERY node
  uint64_t id = args->uget(remus::NODE_ID);
  uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  uint64_t mn = args->uget(remus::LAST_MN_ID);
  uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  uint64_t cn = args->uget(remus::LAST_CN_ID);
  uint64_t ops = args->uget(OPS);
  auto exp_op_str = args->sget(EXP_OP);
  Operation exp_op;
  auto zero_copy = args->uget(ZERO_COPY);
  if (exp_op_str == "Read") {
    exp_op = Operation::Read;
  } else if (exp_op_str == "Write") {
    exp_op = Operation::Write;
  } else if (exp_op_str == "CAS") {
    exp_op = Operation::CAS;
  } else if (exp_op_str == "FAA") {
    exp_op = Operation::FAA;
  } else {
    REMUS_FATAL("Invalid operation: {}", exp_op_str);
    exit(1);
  }
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

  Type typeval;
  // At this point, everything is configured!  If this is a compute node, make
  // some threads and have them use RDMA
  if (id >= c0 && id <= cn) {
    // Threads at this node
    const uint64_t cn_threads = args->uget(remus::CN_THREADS);
    // Total threads in the experiment (all must reach the barriers)
    const uint64_t barrier_thread_count = (cn - c0 + 1) * cn_threads;

    std::vector<std::thread> worker_threads;
    for (uint64_t i = 0; i < cn_threads; ++i) {

      worker_threads.push_back(std::thread(

          // [mfs] The two code paths look very similar.  Can we merge more, so
          // the example is shorter?

          [&](int i) {
            // Create thread context
            auto t =
                std::make_unique<remus::ComputeThread>(id, compute_node, args);
            Type *type = t->local_allocate<Type>();
            if (id == c0 && i == 0) {
              auto ptr = t->allocate<Type>();
              t->set_root(ptr);
              // Wait for all threads on all nodes to have thread contexts.
              // Implicitly, passing the barrier means all Compute and Memory
              // nodes are done with configuration
              if (args->sget(EXP_NAME) != "perftest") {
                t->arrive_control_barrier(barrier_thread_count);
              }
              auto start = std::chrono::high_resolution_clock::now();
              // Perform the operations
              for (uint64_t j = 0; j < ops; j++) {
                if (exp_op == Operation::Write) {
                  if (zero_copy) {
                    t->Write(ptr, type);
                  } else {
                    t->Write(ptr, typeval);
                  }
                } else if (exp_op == Operation::Read) {
                  if (zero_copy) {
                    t->Read(ptr, type);
                  } else {
                    t->Read(ptr);
                  }
                } else if (exp_op == Operation::CAS) {
                  t->CompareAndSwap(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                    type->value, type->value + 1);
                } else if (exp_op == Operation::FAA) {
                  t->FetchAndAdd(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                 type->value);
                } else {
                  REMUS_FATAL("Invalid operation: {}", exp_op_str);
                }
              }
              // Don't exit until everyone finishes the experiment
              if (args->sget(EXP_NAME) != "perftest") {
                t->arrive_control_barrier(barrier_thread_count);
              }
              auto end = std::chrono::high_resolution_clock::now();

              auto duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        start);
              metrics(args->sget(EXP_NAME), cn - c0 + 1, cn_threads, ops,
                      duration, op_name[exp_op], zero_copy);
            } else {
              // wait until the root is set
              t->arrive_control_barrier(barrier_thread_count);
              auto ptr = t->get_root<Type>();
              for (uint64_t j = 0; j < ops; j++) {
                if (exp_op == Operation::Write) {
                  if (zero_copy) {
                    t->Write(ptr, type);
                  } else {
                    t->Write(ptr, typeval);
                  }
                } else if (exp_op == Operation::Read) {
                  if (zero_copy) {
                    t->Read(ptr, type);
                  } else {
                    t->Read(ptr);
                  }
                } else if (exp_op == Operation::CAS) {
                  t->CompareAndSwap(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                    typeval.value, typeval.value + 1);
                } else if (exp_op == Operation::FAA) {
                  t->FetchAndAdd(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                 typeval.value);
                } else {
                  REMUS_FATAL("Invalid operation: {}", exp_op_str);
                }
              }
              // dont't exit until everyone finishes the experiment
              if (args->sget(EXP_NAME) != "perftest") {
                t->arrive_control_barrier(barrier_thread_count);
              }
            }
            REMUS_INFO("All threads finished!");
            t->arrive_control_barrier(barrier_thread_count);
          },
          i));
    }
    for (auto &t : worker_threads) {
      t.join();
    }

    // TODO: Should we have some kind of memory_node->shutdown() here?
  }
  return 0;
}
