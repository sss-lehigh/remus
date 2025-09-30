#pragma once

#include <atomic>
#include <csignal>
#include <random>

#include <remus/remus.h>

using std::uniform_int_distribution;
using namespace remus;

// Command-line options for integer set microbenchmarks
constexpr const char *NUM_OPS = "--num-ops";
constexpr const char *PREFILL = "--prefill";
constexpr const char *INSERT = "--insert";
constexpr const char *REMOVE = "--remove";
constexpr const char *KEY_LB = "--key-lb";
constexpr const char *KEY_UB = "--key-ub";

/// An ARGS object for integer set microbenchmarks
auto DS_EXP_ARGS = {
    U64_ARG_OPT(NUM_OPS, "Number of operations to run per thread", 8192),
    U64_ARG_OPT(PREFILL, "Percent of elements to prefill the data structure",
                50),
    U64_ARG_OPT(INSERT, "Percent of operations that should be inserts", 50),
    U64_ARG_OPT(REMOVE, "Percent of operations that should be removes", 50),
    U64_ARG_OPT(KEY_LB, "Lower bound of the key range", 0),
    U64_ARG_OPT(KEY_UB, "Upper bound of the key range", 1024),
};

/// Metrics is used to track events during the execution of an experiment
struct Metrics {
  size_t get_t = 0;       // calls to `get()` that returned true
  size_t get_f = 0;       // calls to `get()` that returned false
  size_t ins_t = 0;       // calls to `insert()` that returned true
  size_t ins_f = 0;       // calls to `insert()` that returned false
  size_t rmv_t = 0;       // calls to `remove()` that returned true
  size_t rmv_f = 0;       // calls to `remove()` that returned false
  size_t op_count = 0;    // expected total number of operations
  size_t write_ops = 0;   // number of RDMA writes
  size_t write_bytes = 0; // bytes written over RDMA
  size_t read_ops = 0;    // number of RDMA reads
  size_t read_bytes = 0;  // bytes read over RDMA
  size_t faa_ops = 0;     // number of RDMA FAA operations
  size_t cas_ops = 0;     // number of RDMA CAS operations

  /// Write this Metrics object to a file
  ///
  /// @param filename  The name of the file to write to
  /// @param duration  The duration of the experiment, in microseconds
  /// @param compute_thread A compute thread, for additional metrics
  void to_file(double duration, std::shared_ptr<ComputeThread> compute_thread) {
    std::ofstream file("metrics.txt", std::ios::out);
    file << "duration: " << duration << std::endl;
    file << "get_t: " << get_t << std::endl;
    file << "get_f: " << get_f << std::endl;
    file << "ins_t: " << ins_t << std::endl;
    file << "ins_f: " << ins_f << std::endl;
    file << "rmv_t: " << rmv_t << std::endl;
    file << "rmv_f: " << rmv_f << std::endl;
    file << "op_count: " << op_count << std::endl;
    file << "write: " << compute_thread->metrics_.write.ops << std::endl;
    file << "bytes_write: " << compute_thread->metrics_.write.bytes
         << std::endl;
    file << "read: " << compute_thread->metrics_.read.ops << std::endl;
    file << "bytes_read: " << compute_thread->metrics_.read.bytes << std::endl;
    file << "faa: " << compute_thread->metrics_.faa << std::endl;
    file << "cas: " << compute_thread->metrics_.cas << std::endl;
  }
};

/// A per-thread object for running an integer set data structure experiment
///
/// @tparam S  The type of the data structure to test
/// @tparam K  The type of keys in that data structure
template <typename S, typename K> struct IntSetTest {
  using CT = std::shared_ptr<ComputeThread>;
  using AM = std::shared_ptr<ArgMap>;

  Metrics metrics_; // This thread's metrics
  S &set_;          // A reference to the set

  uint64_t thread_id_; // This thread's thread Id
  uint64_t node_id_;   // The Id of the compute node where the thread is running

  /// Construct an IntSetTest object
  ///
  /// @param set            A reference to the data structure
  /// @param thread_id      The thread's Id
  /// @param node_id        The Compute Node id
  IntSetTest(S &set, uint64_t thread_id, uint64_t node_id)
      : set_(set), thread_id_(thread_id), node_id_(node_id) {}

  /// Perform a distributed prefill of the data structure
  ///
  /// This uses the number of threads and the key range to compute a contiguous
  /// subrange of keys to insert, and has each thread insert their range
  ///
  /// @param ct     The calling thread's Remus context
  /// @param params The arguments to the program
  void prefill(CT &ct, AM &params) {
    // How big is the per-thread range of keys?
    auto total_threads =
        params->uget(CN_THREADS) *
        (params->uget(LAST_CN_ID) - params->uget(FIRST_CN_ID) + 1);
    auto key_lb = params->uget(KEY_LB);
    auto key_ub = params->uget(KEY_UB);
    auto range_length = (key_ub - key_lb + 1) / total_threads;
    // How many keys to insert within that range?
    auto num_keys =
        (key_ub - key_lb + 1) * params->uget(PREFILL) / 100 / total_threads;
    // Ok, do the insert
    auto start_key = key_lb + thread_id_ * range_length;
    auto end_key = start_key + range_length;
    auto step = (end_key - start_key) / num_keys;
    for (auto key = start_key; key < end_key; key += step) {
      K key_tmp = static_cast<K>(key);
      set_.insert(key_tmp, ct);
    }
  }

  /// Aggregate this thread's metrics into a global (remote memory) metrics
  /// object
  ///
  /// @param ct        The calling thread's Remus context
  /// @param g_metrics The global metrics object
  void collect(CT &ct, rdma_ptr<Metrics> g_metrics) {
    auto base = g_metrics.raw();
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, get_t)),
                    metrics_.get_t);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, get_f)),
                    metrics_.get_f);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, ins_t)),
                    metrics_.ins_t);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, ins_f)),
                    metrics_.ins_f);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, rmv_t)),
                    metrics_.rmv_t);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, rmv_f)),
                    metrics_.rmv_f);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, op_count)),
                    metrics_.op_count);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, write_ops)),
                    ct->metrics_.write.ops);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, write_bytes)),
                    ct->metrics_.write.bytes);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, read_ops)),
                    ct->metrics_.read.ops);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, read_bytes)),
                    ct->metrics_.read.bytes);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, faa_ops)),
                    ct->metrics_.faa);
    ct->FetchAndAdd(rdma_ptr<uint64_t>(base + offsetof(Metrics, cas_ops)),
                    ct->metrics_.cas);
  }

  /// Run the experiment in this thread by executing a tight loop of operations,
  /// selected using the ratios given through command-line arguments.
  ///
  /// @param ct     The calling thread's Remus context
  /// @param params The arguments to the program
  void run(CT &ct, AM &params) {
    // set up a PRNG for the thread, and a few distributions
    using std::uniform_int_distribution;
    uniform_int_distribution<size_t> key_dist(params->uget(KEY_LB),
                                              params->uget(KEY_UB));
    uniform_int_distribution<size_t> action_dist(0, 100);
    std::mt19937 gen(std::random_device{}());
    // Get the target operation ratios from ARGS
    auto insert_ratio = params->uget(INSERT);
    auto remove_ratio = params->uget(REMOVE);
    auto lookup_ratio = 100 - insert_ratio - remove_ratio;
    uint64_t num_ops = params->uget(NUM_OPS);
    // Do a fixed number of operations per thread
    for (uint64_t i = 0; i < num_ops; ++i) {
      size_t key = key_dist(gen);
      size_t action = action_dist(gen);
      if (action <= lookup_ratio) {
        K key_tmp = static_cast<K>(key);
        if (set_.get(key_tmp, ct)) {
          ++metrics_.get_t;
        } else {
          ++metrics_.get_f;
        }
      } else if (action < lookup_ratio + insert_ratio) {
        K key_tmp = static_cast<K>(key);
        if (set_.insert(key_tmp, ct)) {
          ++metrics_.ins_t;
        } else {
          ++metrics_.ins_f;
        }
      } else {
        K key_tmp = static_cast<K>(key);
        if (set_.remove(key_tmp, ct)) {
          ++metrics_.rmv_t;
        } else {
          ++metrics_.rmv_f;
        }
      }
      ++metrics_.op_count;
    }
  }
};