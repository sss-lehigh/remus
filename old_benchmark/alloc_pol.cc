#include <memory>
#include <unistd.h>
#include <vector>

// [mfs] Should remus expose a single include for all public functionality?

#include "cloudlab.h"
#include "exp_cfg.h"
#include <chrono>
#include <fstream>
#include <random>
#include <remus/cfg.h>
#include <remus/cli.h>
#include <remus/compute_node.h>
#include <remus/compute_thread.h>
#include <remus/logging.h>
#include <remus/mem_node.h>
#include <remus/util.h>
#include <string>
#include <unordered_map>

void metrics(std::string exp_name, uint64_t nnodes, uint64_t nthreads,
             uint64_t ops_per_thread, std::chrono::microseconds duration,
             std::string exp_op, uint64_t zero_copy, uint64_t elements,
             uint64_t overlap) {
  std::ofstream file(
      "metrics.txt",
      std::ios::out); // Use std::ios::out instead of std::ios::app to overwrite
  file << "Experiment: " << exp_name << std::endl;
  file << "TypeSize: " << TYPE_SIZE << std::endl;
  file << "OpType: " << exp_op << std::endl;
  file << "ZeroCopy: " << zero_copy << std::endl;
  file << "Elements: " << elements << std::endl;
  file << "Overlap: " << overlap << std::endl;
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

  // Latency: microseconds per operation = duration_in_microseconds / total_ops
  file << "Latency(us): " << static_cast<double>(duration.count()) / total_ops
       << std::endl;
}
union Type {
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
  uint64_t sn = args->uget(remus::SEGS_PER_MN);
  uint64_t ops = args->uget(OPS);
  auto exp_op_str = args->sget(EXP_OP);
  Operation exp_op;
  auto zero_copy = args->uget(ZERO_COPY);
  auto overlap = args->uget(OVERLAP);
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

  std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
  remus::rdma_ptr<Type> *elements =
      new remus::rdma_ptr<Type>[args->uget(ELEMENTS)];
  auto elements_per_slab = args->uget(ELEMENTS) / sn / (mn - m0 + 1);

  if (id >= c0 && id <= cn) {
    for (uint64_t i = 0; i < args->uget(remus::CN_THREADS); ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }
    // allocate memory elements
    if (id == c0) {
      // Allocate memory for the elements start array - one entry per slab
      // across all memory nodes
      auto elements_start = compute_threads[0]->allocate<remus::rdma_ptr<Type>>(
          sn * (mn - m0 + 1));

      // Allocate elements for all slabs except the last one
      for (uint64_t i = 0; i < sn * (mn - m0 + 1) - 1; i++) {
        // Allocate a chunk of elements for this slab
        auto elements_start_i =
            compute_threads[0]->allocate<Type>(elements_per_slab);

        // Set up pointers to each individual orec in this chunk
        for (uint64_t j = 0; j < elements_per_slab; ++j) {
          elements[i * elements_per_slab + j] = elements_start_i + j;
          // REMUS_INFO("Orecs[{}]: {}", i * orec_per_slab + j,
          // (uintptr_t)orecs[i * orec_per_slab + j]);
        }

        // Store the base pointer of this chunk in the orecs_start array
        compute_threads[0]->Write(elements_start + i, elements_start_i);
      }
      // Handle the last slab which may have a different number of orecs
      uint64_t allocated_elements =
          elements_per_slab * (sn * (mn - m0 + 1) - 1);
      uint64_t remaining_elements = args->uget(ELEMENTS) - allocated_elements;
      // Allocate the remaining orecs
      auto elements_start_last =
          compute_threads[0]->allocate<Type>(remaining_elements);

      // Set up pointers to each individual orec in the last chunk
      for (uint64_t j = 0; j < remaining_elements; ++j) {
        elements[allocated_elements + j] = elements_start_last + j;
        // REMUS_INFO("Orecs[{}]: {}", allocated_orecs + j,
        // (uintptr_t)orecs[allocated_orecs + j]);
      }

      // Store the base pointer of the last chunk in the orecs_start array
      compute_threads[0]->Write(elements_start + (sn * (mn - m0 + 1) - 1),
                                elements_start_last);
      // Set the root of the compute thread to the first element
      compute_threads[0]->set_root(elements_start);
      compute_threads[0]->arrive_control_barrier(cn - c0 + 1);
      compute_threads[0]->arrive_control_barrier(cn - c0 + 1);
    } else {
      compute_threads[0]->arrive_control_barrier(cn - c0 + 1);
      auto elements_start =
          compute_threads[0]->get_root<remus::rdma_ptr<Type>>();
      for (uint64_t i = 0; i < sn * (mn - m0 + 1) - 1; i++) {
        auto elements_start_i = compute_threads[0]->Read(elements_start + i);
        for (uint64_t j = 0; j < elements_per_slab; ++j) {
          elements[i * elements_per_slab + j] = elements_start_i + j;
        }
      }
      uint64_t allocated_elements =
          elements_per_slab * (sn * (mn - m0 + 1) - 1);
      uint64_t remaining_elements = args->uget(ELEMENTS) - allocated_elements;
      auto elements_start_last =
          compute_threads[0]->Read(elements_start + (sn * (mn - m0 + 1) - 1));
      for (uint64_t j = 0; j < remaining_elements; ++j) {
        elements[allocated_elements + j] = elements_start_last + j;
      }
      compute_threads[0]->arrive_control_barrier(cn - c0 + 1);
    }
    // Threads at this node
    const uint64_t cn_threads = args->uget(remus::CN_THREADS);
    // Total threads in the experiment (all must reach the barriers)
    const uint64_t barrier_thread_count = (cn - c0 + 1) * cn_threads;

    //
    // start worker threads
    //
    std::vector<std::thread> worker_threads;
    Type typeval;
    for (uint64_t i = 0; i < cn_threads; ++i) {

      worker_threads.push_back(std::thread(

          [&](int i) {
            using std::uniform_int_distribution;
            uniform_int_distribution<size_t> index_dist(
                0, args->uget(ELEMENTS) - 1);
            std::mt19937 gen(std::random_device{}());
            // Create thread context

            Type *type = compute_threads[i]->local_allocate<Type>();
            remus::rdma_ptr<Type> *ptrs = new remus::rdma_ptr<Type>[ops];
            for (uint64_t j = 0; j < ops; j++) {
              ptrs[j] = elements[index_dist(gen)];
            }
            // wait until the prepare phase is done
            compute_threads[i]->arrive_control_barrier(barrier_thread_count);
            auto start = std::chrono::high_resolution_clock::now();
            if (id == c0 && i == 0) {
              for (uint64_t j = 0; j < ops; j++) {
                auto ptr = ptrs[j];
                if (exp_op == Operation::Write) {
                  if(zero_copy) {
                    compute_threads[i]->Write(ptr, type);
                  } else {
                    compute_threads[i]->Write(ptr, typeval);
                  }
                } else if (exp_op == Operation::Read) {
                  if(zero_copy) {
                    compute_threads[i]->Read(ptr, type);
                  } else {
                    compute_threads[i]->Read(ptr);
                  }
                } else if (exp_op == Operation::CAS) {
                  compute_threads[i]->CompareAndSwap(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                    typeval.value, typeval.value + 1);
                } else if (exp_op == Operation::FAA) {
                  compute_threads[i]->FetchAndAdd(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                 typeval.value);
                } else {
                  REMUS_FATAL("Invalid operation: {}", exp_op_str);
                }
              }
              // dont't exit until everyone finishes the experiment
              compute_threads[i]->arrive_control_barrier(barrier_thread_count);
              auto end = std::chrono::high_resolution_clock::now();
              auto duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        start);
              metrics(args->sget(EXP_NAME), cn + 1, cn_threads, ops, duration,
                      op_name[exp_op], zero_copy, args->uget(ELEMENTS),
                      overlap);
            } else {
              for (uint64_t j = 0; j < ops; j++) {
                auto ptr = ptrs[j];
                if (exp_op == Operation::Write) {
                  if(zero_copy) {
                    compute_threads[i]->Write(ptr, type);
                  } else {
                    compute_threads[i]->Write(ptr, typeval);
                  }
                } else if (exp_op == Operation::Read) {
                  if(zero_copy) {
                    compute_threads[i]->Read(ptr, type);
                  } else {
                    compute_threads[i]->Read(ptr);
                  }
                } else if (exp_op == Operation::CAS) {
                  compute_threads[i]->CompareAndSwap(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                    typeval.value, typeval.value + 1);
                } else if (exp_op == Operation::FAA) {
                  compute_threads[i]->FetchAndAdd(remus::rdma_ptr<uint64_t>((uintptr_t)ptr),
                                 typeval.value);
                } else {
                  REMUS_FATAL("Invalid operation: {}", exp_op_str);
                }
              }
              // dont't exit until everyone finishes the experiment
              compute_threads[i]->arrive_control_barrier(barrier_thread_count);
            }
          },
          i));
    }
    for (auto &t : worker_threads) {
      t.join();
    }
    REMUS_INFO("All threads finished!");
  }

  // Since ComputeNode is a smart pointer, it will destruct as main exits.

  return 0;
}
