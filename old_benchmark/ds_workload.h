#include <atomic>
#include <csignal>
#include <random>

#include <remus/remus.h>

#include "exp_cfg.h"

using std::uniform_int_distribution;
using namespace remus;

static std::atomic<bool> running_;

static std::atomic<int> num_ops_;

template <typename ds_t, typename K, typename V> class ds_workload {
public:
  struct Metrics {
    size_t get_t;
    size_t get_f;
    size_t ins_t;
    size_t ins_f;
    size_t rmv_t;
    size_t rmv_f;
    size_t op_count;

    void to_file(double duration, // microseconds
                 std::shared_ptr<ComputeThread> compute_thread) {
      std::ofstream file("metrics.txt", std::ios::out);
      // [mfs] Why not just report the duration and not compute the throughput
      // for each of these?
      file << "get_t: " << get_t * 1000000 / duration << std::endl;
      file << "get_f: " << get_f * 1000000 / duration << std::endl;
      file << "ins_t: " << ins_t * 1000000 / duration << std::endl;
      file << "ins_f: " << ins_f * 1000000 / duration << std::endl;
      file << "rmv_t: " << rmv_t * 1000000 / duration << std::endl;
      file << "rmv_f: " << rmv_f * 1000000 / duration << std::endl;
      file << "op_count: " << op_count * 1000000 / duration << std::endl;
      file << "write: "
           << compute_thread->metrics_.write.ops * 1000000 / duration
           << std::endl;
      file << "bytes_write: "
           << compute_thread->metrics_.write.bytes * 1000000 / duration
           << std::endl;
      file << "read: " << compute_thread->metrics_.read.ops * 1000000 / duration
           << std::endl;
      file << "bytes_read: "
           << compute_thread->metrics_.read.bytes * 1000000 / duration
           << std::endl;
      file << "faa: " << compute_thread->metrics_.faa * 1000000 / duration
           << std::endl;
      file << "cas: " << compute_thread->metrics_.cas * 1000000 / duration
           << std::endl;
    }
    Metrics()
        : get_t(0), get_f(0), ins_t(0), ins_f(0), rmv_t(0), rmv_f(0),
          op_count(0) {}
  };

protected:
  Metrics metrics_;
  ds_t &ds_;
  std::shared_ptr<ComputeThread> compute_thread_;
  std::shared_ptr<ArgMap> params_;

  uint64_t thread_id_;
  uint64_t node_id_;

public:
  ds_workload(ds_t &ds, uint64_t thread_id, uint64_t node_id,
              std::shared_ptr<ComputeThread> compute_thread,
              std::shared_ptr<ArgMap> params)
      : ds_(ds), compute_thread_(compute_thread), params_(params),
        thread_id_(thread_id), node_id_(node_id) {}
  ~ds_workload() { REMUS_INFO("ds_workload destructing"); }
  void prefill() {
    auto total_threads =
        params_->uget(CN_THREADS) *
        (params_->uget(LAST_CN_ID) - params_->uget(FIRST_CN_ID) + 1);
    auto key_lb = params_->uget(KEY_LB);
    auto key_ub = params_->uget(KEY_UB);
    auto total_range_size = (key_ub - key_lb + 1) / total_threads;
    auto total_fill_size =
        (key_ub - key_lb + 1) * params_->uget(PREFILL) / 100 / total_threads;
    auto start_key = key_lb + thread_id_ * total_range_size;
    auto end_key = start_key + total_range_size;
    auto step = (end_key - start_key) / total_fill_size;
    for (auto key = start_key; key < end_key; key += step) {
      K key_tmp = static_cast<K>(key);
      V val_tmp = static_cast<V>(key);
      ds_.insert(key_tmp, val_tmp, compute_thread_);
    }
  }
  static void signal_handler(int) { running_ = false; }

  void run() {
    running_ = true;
    std::signal(SIGALRM, signal_handler);
    if (params_->uget(TIME_MODE) == 0) {
      num_ops_ = params_->uget(NUM_OPS);
    } else {
      alarm(params_->uget(RUN_TIME));
    }

    work();
  }
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

protected:
  void work() {
    // set up a PRNG for the thread
    using std::uniform_int_distribution;
    uniform_int_distribution<size_t> key_dist(params_->uget(KEY_LB),
                                              params_->uget(KEY_UB));
    uniform_int_distribution<size_t> action_dist(0, 100);
    std::mt19937 gen(std::random_device{}());
    auto insert_ratio = params_->uget(INSERT);
    auto remove_ratio = params_->uget(REMOVE);
    auto lookup_ratio = 100 - insert_ratio - remove_ratio;
    while (running_) {
      if (params_->uget(TIME_MODE) == 0 && num_ops_-- <= 0) {
        break;
      }
      size_t key = key_dist(gen);
      size_t action = action_dist(gen);
      if (action <= lookup_ratio) {
        K key_tmp = static_cast<K>(key);
        if (ds_.get(key_tmp, compute_thread_))
          ++metrics_.get_t;
        else
          ++metrics_.get_f;
      } else if (action < lookup_ratio + insert_ratio) {
        K key_tmp = static_cast<K>(key);
        V val_tmp = static_cast<V>(key);
        if (ds_.insert(key_tmp, val_tmp, compute_thread_))
          ++metrics_.ins_t;
        else
          ++metrics_.ins_f;
      } else {
        K key_tmp = static_cast<K>(key);
        if (ds_.remove(key_tmp, compute_thread_))
          ++metrics_.rmv_t;
        else
          ++metrics_.rmv_f;
      }
      ++metrics_.op_count;
    }
  }
};