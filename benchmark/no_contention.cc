#include <unistd.h>

#include <memory>
#include <vector>

// [mfs] Should remus expose a single include for all public functionality?

#include <remus/cfg.h>
#include <remus/cli.h>
#include <remus/compute_node.h>
#include <remus/ext_compute_thread.h>
#include <remus/ext_mem_node.h>
#include <remus/logging.h>
#include <remus/util.h>

#include <chrono>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cloudlab.h"
#include "exp_cfg.h"

//TODO: THIS NEEDS TO BE FIXED!
void metrics(std::string exp_name, uint64_t nnodes, uint64_t nthreads,
             uint64_t ops_per_thread, std::chrono::microseconds duration,
             std::string exp_op, uint64_t zero_copy) {
  std::ofstream file("metrics.txt",
                     std::ios::out);  // Use std::ios::out instead of
                                      // std::ios::app to overwrite
  file << "Experiment: " << exp_name << std::endl;
  file << "TypeSize: " << TYPE_SIZE << std::endl;
  file << "OpType: " << exp_op << std::endl;
  file << "ZeroCopy: " << zero_copy << std::endl;
  file << "Nodes: " << nnodes << std::endl;
  file << "Threads: " << nthreads << std::endl;

  // Calculate metrics
  uint64_t total_ops = ops_per_thread * nthreads * nnodes;
  uint64_t total_bytes = total_ops * TYPE_SIZE;
  double usec = static_cast<double>(duration.count()); 
  double seconds = usec / 1e6; // convert to seconds
  // Calculate throughput 
  double ops_per_s = static_cast<double>(total_ops) / seconds; 
  file << "Throughput(ops/sec): " << ops_per_s << std::endl;
  // Calculate bandwidth
  double mb_per_s = static_cast<double>(total_bytes) / 1e6 / seconds;
  file << "Bandwidth(MB/sec): " << mb_per_s << std::endl;
  // Calculate latency
  double s_per_op = 1.0 / ops_per_s;
  double latency_us = s_per_op * 1e6; // convert to microseconds
  file << "Latency(us): " << latency_us << std::endl;

}

union alignas(64) Type {
  uint8_t padding[TYPE_SIZE];
  uint64_t value;
};

enum Operation { Write, Read, CAS, FAA };
int main(int argc, char **argv) {
  std::unordered_map<Operation, std::string> op_name = {
      {Operation::Write, "Write"},
      {Operation::Read, "Read"},
      {Operation::CAS, "CAS"},
      {Operation::FAA, "FAA"}};
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
  std::unique_ptr<remus::ExtMemoryNode> memory_node;

  // Information needed if this machine will operate as a compute node
  std::shared_ptr<remus::ComputeNode> compute_node;

  // Memory Node configuration must come first!
  if (id >= m0 && id <= mn) {
    // Make the pools, await connections
    memory_node.reset(new remus::ExtMemoryNode(self, args));
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
  typeval.value = 0; // Initialize the type value to zero
  
  
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
          [&](int i) {
            // Create thread context
            auto t = std::make_unique<remus::EXTComputeThread>(id, compute_node,
                                                               args);
            //Every thread allocates its own landing object and target  --> no contention
            Type *type = t->local_allocate<Type>();
            auto ptr = t->allocate<Type>();
            t->set_root(ptr);

            t->arrive_control_barrier(barrier_thread_count);
            
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
            // End timing immediately after the operations
            auto end = std::chrono::high_resolution_clock::now();
            // Don't exit until everyone finishes the experiment
            
            t->arrive_control_barrier(barrier_thread_count);

            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
            // Log the metrics
            if (i == 0) {
              metrics(args->sget(EXP_NAME), cn - c0 + 1, cn_threads, ops,
                      duration, op_name[exp_op], zero_copy);
            }
            REMUS_INFO("All threads finished!");
          },
          i));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
  }
  return 0;
}