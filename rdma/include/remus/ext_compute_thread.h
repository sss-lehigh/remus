#pragma once
#include "cfg.h"
#include "compute_thread.h"
#include "ext_rdma_ops.h"
namespace remus {
class EXTComputeThread : public ComputeThread {
  const uint64_t cached_slice_start;
  uint64_t cached_slice_curr;
  uint64_t seg_size_;
  uint64_t mn;

public:
  EXTComputeThread(uint64_t id, std::shared_ptr<ComputeNode> cn,
                     std::shared_ptr<ArgMap> args)
      : ComputeThread(id, cn, args),
        cached_slice_start(1ULL << (args->uget(remus::CN_THREAD_BUFSZ) - 1)),
        cached_slice_curr(cached_slice_start),
        seg_size_(1ULL << (args->uget(remus::CN_THREAD_BUFSZ))),
        mn(args->uget(remus::LAST_MN_ID) - args->uget(remus::FIRST_MN_ID) + 1) {}
  /// Read a fixed-sized object from the RDMA heap (overwrite the basic version,
  /// add check for overflow)
  template <typename T> T Read(rdma_ptr<T> ptr) {
    REMUS_ASSERT(sizeof(T) <= cached_slice_start,
                 "Cached slice overflow in Read");
    /// Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    internal::ReadInternal(ptr, 0, sizeof(T), sizeof(T), seg_slice_, rkey,
                           ci.lkey, ci.conn_.get(), &counters_[0]);
    return *(T *)seg_slice_;
  }
  // this is a special version that allows read to seg
  template <typename T>
  void Read(remus::rdma_ptr<T> ptr, T *seg) {
    REMUS_ASSERT((uint64_t)seg + sizeof(T) <= (uint64_t)seg_slice_ + seg_size_, "Cached slice overflow in Read to seg");
    /// Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    internal::ReadInternal(ptr, 0, sizeof(T), sizeof(T), (uint8_t *)seg, rkey, ci.lkey,
                           ci.conn_.get(), &counters_[0]);
  }

  /// Write to an RDMA heap (overwrite the basic version, add check for
  /// overflow)
  template <typename T> void Write(rdma_ptr<T> ptr, const T &val) {
    REMUS_ASSERT(sizeof(T) <= cached_slice_start,
                 "Cached slice overflow in Write");
    // Use the scheduling policy to select the next connection
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    internal::Write(ptr, val, seg_slice_, rkey, ci.lkey, ci.conn_.get(),
                    &counters_[0]);
  }

  // this is a special version that allows specify the size of the write
  template <typename T> void Write(rdma_ptr<T> ptr, const T &val, size_t size) {
    REMUS_ASSERT(size * sizeof(T) <= cached_slice_start, "Cached slice overflow in Write with size");
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = this->compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = this->compute_node_->get_rkey(ptr.raw());
    internal::Write(ptr, val, seg_slice_, rkey, ci.lkey, ci.conn_.get(),
                    &counters_[0], size * sizeof(T));
  }
  // this is a special version that allows specify the local address
  template <typename T>
  void Write(rdma_ptr<T> ptr, T *seg) {
    REMUS_ASSERT(uint64_t(seg) + sizeof(T) <= (uint64_t)seg_slice_ + seg_size_, "Cached slice overflow in Write with local address");
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(ptr.id());
    auto &ci = this->compute_node_->get_conn(ptr.raw(), conn_idx);
    uint32_t rkey = this->compute_node_->get_rkey(ptr.raw());
    internal::Write(ptr, (uint8_t *)seg, rkey, ci.lkey, ci.conn_.get(), &counters_[0],
                    sizeof(T));
  }
  // only allocate memory in local memory seg_slice_, n means n bytes
  template <typename T>
  T *local_allocate(std::size_t n = 1) {
    REMUS_ASSERT(cached_slice_curr + n*sizeof(T) <= seg_size_, "Cached slice overflow in local_allocate");
    auto ret = (T *)(cached_slice_curr + (uint64_t)seg_slice_);
    cached_slice_curr += n*sizeof(T);
    return ret;
  }

  void reset_cached_slice() { cached_slice_curr = cached_slice_start; }

  template <typename T>
  void BatchWrite(std::vector<rdma_ptr<T>> &addresses,
                  const std::vector<T> &values, bool use_one) {
    // [mfs]  This is not yet using the qp-sched-pol to pick which connection to
    //        use, instead it uses 0.
    REMUS_ASSERT(sizeof(T) * internal::kMaxWr <=
                     cached_slice_start,
                 "Cached slice overflow in BatchWrite");
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(addresses[0].id());
    auto &ci = this->compute_node_->get_conn(addresses[0].raw(), conn_idx);
    uint32_t rkey = this->compute_node_->get_rkey(addresses[0].raw());
    internal::BatchWrite(addresses, values, use_one, seg_slice_, rkey, ci.lkey,
                         ci.conn_.get(), &counters_[0]);
  }

  template <typename T>
  std::vector<T> BatchRead(std::vector<rdma_ptr<T>> &addresses) {
    // [mfs]  This is not yet using the qp-sched-pol to pick which connection to
    //        use, instead it uses 0.
    REMUS_ASSERT(sizeof(T) * internal::kMaxWr <=
                     cached_slice_start,
                 "Cached slice overflow in BatchRead");
    uint32_t conn_idx = qp_sched_pol_.get_lane_idx(addresses[0].id());
    auto &ci = this->compute_node_->get_conn(addresses[0].raw(), conn_idx);
    uint32_t rkey = this->compute_node_->get_rkey(addresses[0].raw());
    return internal::BatchRead(addresses, seg_slice_, rkey, ci.lkey,
                               ci.conn_.get(), &counters_[0]);
  }
  ~EXTComputeThread() {
    // send shutdown to all memory nodes's first segment's control block's control_flag_
    for (uint64_t i = 0; i < mn; i++) {
      auto control_flag = rdma_ptr<uint64_t>(this->compute_node_->get_seg_start(i, 0) +
                            offsetof(internal::ControlBlock, control_flag_));
      FetchAndAdd(control_flag, 1);
    }
    REMUS_INFO("ComputeThread destructed");
  }
};
} // namespace remus