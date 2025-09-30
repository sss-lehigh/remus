// [mfs] The only remaining issue in this code is that we only report one
// thread's RDMA metrics, instead of all threads.

#pragma once

#include <atomic>
#include <csignal>
#include <random>

#include <remus/remus.h>

using std::uniform_int_distribution;
using namespace remus;

// Command-line operations for data structure microbenchmarks
constexpr const char *NUM_OPS = "--num-ops";
constexpr const char *PREFILL = "--prefill";
constexpr const char *INSERT = "--insert";
constexpr const char *REMOVE = "--remove";
constexpr const char *KEY_LB = "--key-lb";
constexpr const char *KEY_UB = "--key-ub";

/// ARGS for data structure experiments
auto DS_EXP_ARGS = {
    U64_ARG_OPT(NUM_OPS, "Number of operations to run per thread", 65536),
    U64_ARG_OPT(PREFILL, "Percent of elements to prefill the data structure",
                50),
    U64_ARG_OPT(INSERT, "Percent of operations that should be inserts", 50),
    U64_ARG_OPT(REMOVE, "Percent of operations that should be removes", 50),
    U64_ARG_OPT(KEY_LB, "Lower bound of the key range", 0),
    U64_ARG_OPT(KEY_UB, "Upper bound of the key range", 4096),
};

/// Metrics is used to track events during the execution of an experiment
struct Metrics {
  size_t get_t = 0;    // calls to `get()` that returned true
  size_t get_f = 0;    // calls to `get()` that returned false
  size_t ins_t = 0;    // calls to `insert()` that returned true
  size_t ins_f = 0;    // calls to `insert()` that returned false
  size_t rmv_t = 0;    // calls to `remove()` that returned true
  size_t rmv_f = 0;    // calls to `remove()` that returned false
  size_t op_count = 0; // expected total number of operations

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
    // [mfs] In the rest of this code, I'm concerned that we're only reporting
    //       one thread's info, instead of all threads' info
    file << "write: " << compute_thread->metrics_.write.ops << std::endl;
    file << "bytes_write: " << compute_thread->metrics_.write.bytes
         << std::endl;
    file << "read: " << compute_thread->metrics_.read.ops << std::endl;
    file << "bytes_read: " << compute_thread->metrics_.read.bytes << std::endl;
    file << "faa: " << compute_thread->metrics_.faa << std::endl;
    file << "cas: " << compute_thread->metrics_.cas << std::endl;
  }
};

/// A per-thread object for running a data structure experiment
///
/// @tparam ds_t The type of the data structure to test
/// @tparam K    The type of keys in that data structure
/// @tparam V    The type of values in that data structure
template <typename ds_t, typename K, typename V> struct ds_workload {
  Metrics metrics_; // This thread's metrics
  ds_t &ds_;        // A reference to the data structure
  std::shared_ptr<ComputeThread> compute_thread_; // This thread's ComputeThread
  std::shared_ptr<ArgMap> params_; // The program-wide command-line arguments

  uint64_t thread_id_; // This thread's thread Id
  uint64_t node_id_;   // The Id of the compute node where the thread is running

  /// Construct a ds_workload object
  ///
  /// @param ds             A reference to the data structure
  /// @param thread_id      The thread's Id
  /// @param node_id        The Compute Node id
  /// @param compute_thread The thread's ComputeThread context
  /// @param params         The program-wide command-line arguments
  ds_workload(ds_t &ds, uint64_t thread_id, uint64_t node_id,
              std::shared_ptr<ComputeThread> compute_thread,
              std::shared_ptr<ArgMap> params)
      : ds_(ds), compute_thread_(compute_thread), params_(params),
        thread_id_(thread_id), node_id_(node_id) {}

  /// Perform a distributed prefill of the data structure
  ///
  /// This uses the number of threads and the key range to compute a contiguous
  /// subrange of keys to insert, and has each thread insert their range
  void prefill() {
    // How big is the per-thread range of keys?
    auto total_threads =
        params_->uget(CN_THREADS) *
        (params_->uget(LAST_CN_ID) - params_->uget(FIRST_CN_ID) + 1);
    auto key_lb = params_->uget(KEY_LB);
    auto key_ub = params_->uget(KEY_UB);
    auto range_length = (key_ub - key_lb + 1) / total_threads;
    // How many keys to insert within that range?
    auto num_keys =
        (key_ub - key_lb + 1) * params_->uget(PREFILL) / 100 / total_threads;
    // Ok, do the insert
    auto start_key = key_lb + thread_id_ * range_length;
    auto end_key = start_key + range_length;
    auto step = (end_key - start_key) / num_keys;
    for (auto key = start_key; key < end_key; key += step) {
      K key_tmp = static_cast<K>(key);
      V val_tmp = static_cast<V>(key);
      ds_.insert(key_tmp, val_tmp, compute_thread_);
    }
  }

  /// Aggregate this thread's metrics into a global (remote memory) metrics
  /// object
  ///
  /// @param global_metrics The global metrics object
  void collect(rdma_ptr<Metrics> global_metrics) {
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, get_t)),
        metrics_.get_t);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, get_f)),
        metrics_.get_f);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, ins_t)),
        metrics_.ins_t);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, ins_f)),
        metrics_.ins_f);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, rmv_t)),
        metrics_.rmv_t);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, rmv_f)),
        metrics_.rmv_f);
    compute_thread_->FetchAndAdd(
        rdma_ptr<uint64_t>(global_metrics.raw() + offsetof(Metrics, op_count)),
        metrics_.op_count);
  }

  /// Run the experiment in this thread by executing a tight loop of operations,
  /// selected using the ratios given through command-line arguments.
  void run() {
    // set up a PRNG for the thread, and a few distributions
    using std::uniform_int_distribution;
    uniform_int_distribution<size_t> key_dist(params_->uget(KEY_LB),
                                              params_->uget(KEY_UB));
    uniform_int_distribution<size_t> action_dist(0, 100);
    std::mt19937 gen(std::random_device{}());
    // Get the target operation ratios from ARGS
    auto insert_ratio = params_->uget(INSERT);
    auto remove_ratio = params_->uget(REMOVE);
    auto lookup_ratio = 100 - insert_ratio - remove_ratio;
    uint64_t num_ops = params_->uget(NUM_OPS);
    // Do a fixed number of operations per thread
    for (uint64_t i = 0; i < num_ops; ++i) {
      size_t key = key_dist(gen);
      size_t action = action_dist(gen);
      if (action <= lookup_ratio) {
        K key_tmp = static_cast<K>(key);
        if (ds_.get(key_tmp, compute_thread_)) {
          ++metrics_.get_t;
        } else {
          ++metrics_.get_f;
        }
      } else if (action < lookup_ratio + insert_ratio) {
        K key_tmp = static_cast<K>(key);
        V val_tmp = static_cast<V>(key);
        if (ds_.insert(key_tmp, val_tmp, compute_thread_)) {
          ++metrics_.ins_t;
        } else {
          ++metrics_.ins_f;
        }
      } else {
        K key_tmp = static_cast<K>(key);
        if (ds_.remove(key_tmp, compute_thread_)) {
          ++metrics_.rmv_t;
        } else {
          ++metrics_.rmv_f;
        }
      }
      ++metrics_.op_count;
    }
  }
};