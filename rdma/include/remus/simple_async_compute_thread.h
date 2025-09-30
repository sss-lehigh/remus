#pragma once

#include "remus/compute_thread.h"
#include "remus/simple_async_result.h"

namespace remus {

/// @brief A simple ComputeThread that uses coroutines to perform asynchronous
/// operations
class SimpleAsyncComputeThread : public ComputeThread {
 public:
  SimpleAsyncComputeThread(uint64_t id, std::shared_ptr<ComputeNode> cn,
                           std::shared_ptr<ArgMap> args)
      : ComputeThread(id, cn, args) {}

  /// @brief A simple asynchronous read operation
  /// @tparam T The type of the object read
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param fence If true, a fence is issued after the read operation
  /// @return An AsyncResult that will yield the object read from the RDMA heap
  template <typename T>
  AsyncResult<T> ReadAsync(rdma_ptr<T> ptr, bool fence = false) {
    /// Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto counter = op_counter_t(this).val();
    REMUS_ASSERT(counter != nullptr,
                 "Counter is not enough, increase the number of "
                 "counters or reduce the number of requests");

    auto staging_buf = staging_buf_t(this, sizeof(T), alignof(T)).val();
    REMUS_ASSERT(staging_buf != nullptr,
                 "Staging buffer is not enough, increase the staging buffers "
                 "or reduce the number of requests");
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                         counter, sizeof(T), true, fence);
    internal::Post(send_wr, ci.conn_.get(), counter);
    while (!internal::PollAsync(ci.conn_.get(), counter, ptr)) {
      co_yield std::suspend_always();
    }
    co_return *(T *)staging_buf;
  }
  /// @brief A sequential version of read that uses coroutine index coro_idx and
  /// sequence index seq_idx
  /// @tparam T The type of the object read
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param signal If true, the operation is signaled
  /// @param fence If true, a fence is issued after the read operation
  /// @return An AsyncResult that will yield a vector of objects read from the
  /// RDMA heap
  template <typename T>
  AsyncResult<std::optional<std::vector<T>>> ReadSeqAsync(rdma_ptr<T> ptr,
                                                          bool signal = false,
                                                          bool fence = false) {
    /// Use the scheduling policy to select the next connection
    auto coro_idx =
        0;  // because we don't support more than one top level coroutine
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto staging_buf_ptr =
        std::make_unique<seq_staging_buf_t>(this, sizeof(T), alignof(T));
    auto staging_buf = staging_buf_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].staging_bufs.push_back(
        std::move(staging_buf_ptr));
    REMUS_DEBUG("Debug: start create staging_buf_ptr");
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    REMUS_DEBUG("Debug: finish create op_counter_ptr");
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    REMUS_DEBUG("Debug: finish push op_counter");
    REMUS_ASSERT(staging_buf != nullptr,
                 "Staging buffer is not enough, increase the staging buffers "
                 "or reduce the number of requests");
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    REMUS_DEBUG("push {} to seq_send_wrs[{}][{}]",
                (uint64_t)(uint8_t *)staging_buf, coro_idx, seq_idx);
    if (!signal) {
      internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                           nullptr, sizeof(T), signal, fence);
      co_return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    REMUS_DEBUG("Debug: link seq_idx = {}", seq_idx);

    internal::ReadConfig(send_wr, sge, ptr, staging_buf, rkey, ci.lkey_,
                         op_counter, sizeof(T), signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    REMUS_DEBUG("Debug: erase seq_idx = {}", seq_idx);
    seq_send_wrs[coro_idx].erase(seq_idx);
    co_return result;
  }
  /// @brief Zero-copy version of the AsyncRead that allows reading directly
  /// into a segment
  /// @details
  /// Although this can be used interleaved with other read/write_seq
  /// operations, results of read by calling this function won't be in the
  /// result vector.
  /// @tparam T The type of the object read
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param seg A pointer to the segment where the data will be read into
  /// @param signal If true, the operation is signaled
  /// @param fence If true, a fence is issued after the read operation
  /// @param size The size of the object to read, defaults to sizeof(T)
  /// @return An AsyncResult that will yield a vector of objects read from the
  /// RDMA heap
  template <typename T>
  AsyncResult<std::optional<std::vector<T>>> ReadSeqAsync(
      rdma_ptr<T> ptr, T *seg, bool signal = false, bool fence = false,
      size_t size = sizeof(T)) {
    /// Use the scheduling policy to select the next connection
    auto coro_idx = 0;
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    uint32_t rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::ReadConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                           nullptr, size, signal, fence);
      co_return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::ReadConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                         op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    co_return result;
  }
  /// @brief A simple asynchronous write operation
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param val The value to write to the RDMA heap
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true, the write is performed locally if the pointer
  /// is local
  /// @return An AsyncResultVoid that completes when the write operation is done
  template <typename T>
  AsyncResultVoid WriteAsync(rdma_ptr<T> ptr, const T &val, bool fence = true,
                             size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), &val, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      co_return;
    }
    // Use the scheduling policy to select the next connection
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto staging_buf = staging_buf_t(this, size, alignof(T)).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                          op_counter, size, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    co_return;
  }

  /// @brief Zero-copy version of the AsyncWrite that allows writing directly
  /// into a segment
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param seg A pointer to the segment where the data will be written into
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true, the write is performed locally if the pointer
  /// is local
  /// @return An AsyncResultVoid that completes when the write operation is done
  template <typename T>
  AsyncResultVoid WriteAsync(rdma_ptr<T> ptr, T *seg, bool fence = true,
                             size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), seg, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      co_return;
    }
    auto lane = Lane{qp_sched_pol_.get_lane_idx(ptr.id()),
                     compute_node_->lane_op_counters_};
    auto &ci = this->compute_node_->get_conn(ptr.raw(), lane.lane_idx);
    auto rkey = this->compute_node_->get_rkey(ptr.raw());
    auto op_counter = op_counter_t(this).val();
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                          op_counter, size, true, fence);
    internal::Post(send_wr, ci.conn_.get(), op_counter);
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    co_return;
  }
  /// @brief A sequential version of write that uses coroutine index coro_idx
  /// and sequence index seq_idx
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param val The value to write to the RDMA heap
  /// @param signal If true, the operation is signaled
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true, the write is performed locally if the pointer
  /// is local
  /// @return An AsyncResult that will yield a vector of objects written to the
  /// RDMA heap
  ///
  /// NB: ensure ptr in seq belong to the same memory segment
  template <typename T>
  AsyncResult<std::optional<std::vector<T>>> WriteSeqAsync(
      rdma_ptr<T> ptr, const T &val, bool signal = false, bool fence = false,
      size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), &val, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      co_return std::nullopt;
    }
    auto coro_idx = 0;
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = compute_node_->get_rkey(ptr.raw());
    auto staging_buf_ptr =
        std::make_unique<seq_staging_buf_t>(this, size, alignof(T));
    auto staging_buf = staging_buf_ptr->val();
    REMUS_ASSERT(staging_buf != nullptr,
                 "Staging buffer is not enough, increase the staging buffers "
                 "or reduce the number of requests");
    seq_send_wrs[coro_idx][seq_idx].staging_bufs.push_back(
        std::move(staging_buf_ptr));
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                            nullptr, size, signal, fence);
      co_return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::WriteConfig(send_wr, sge, ptr, val, staging_buf, rkey, ci.lkey_,
                          op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    co_return result;
  }

  // this is a special version that allows write from seg

  /// @brief Zero-copy version of the WriteSeqAsync that allows writing directly
  /// into a segment
  /// @tparam T The type of the object to write
  /// @param ptr The rdma_ptr pointing to the object in the RDMA heap
  /// @param seg A pointer to the segment where the data will be written into
  /// @param signal If true, the operation is signaled
  /// @param fence If true, a fence is issued after the write operation
  /// @param size The size of the object to write, defaults to sizeof(T)
  /// @param local_copy If true, the write is performed locally if the pointer
  /// @return An AsyncResult that will yield a vector of objects written to the
  template <typename T>
  AsyncResult<std::optional<std::vector<T>>> WriteSeqAsync(
      rdma_ptr<T> ptr, T *seg, bool signal = false, bool fence = false,
      size_t size = sizeof(T), bool local_copy = true) {
    if (local_copy && is_local(ptr)) {
      memcpy((void *)ptr.address(), seg, size);
      _mm_clflush((void *)ptr.address());
      if (fence) {
        _mm_sfence();
      }
      co_return std::nullopt;
    }
    auto coro_idx =
        0;  // because we don't support more than one top level coroutine
    auto seq_idx = find_seq_idx(ptr, coro_idx);
    auto &ci = compute_node_->get_conn(
        ptr.raw(), seq_send_wrs[coro_idx][seq_idx].lane->lane_idx);
    auto rkey = this->compute_node_->get_rkey(ptr.raw());
    auto op_counter_ptr = std::make_unique<op_counter_t>(this);
    auto op_counter = op_counter_ptr->val();
    seq_send_wrs[coro_idx][seq_idx].op_counters.push_back(
        std::move(op_counter_ptr));
    auto send_wr = std::make_shared<ibv_send_wr>(ibv_send_wr{});
    auto sge = std::make_shared<ibv_sge>(ibv_sge{});
    seq_send_wrs[coro_idx][seq_idx].send_wrs.push_back({send_wr, sge});
    if (!signal) {
      internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                            nullptr, size, signal, fence);
      co_return std::nullopt;
    }
    link_seq_send_wrs(seq_idx, coro_idx);
    internal::WriteConfig(send_wr, sge, ptr, (uint8_t *)seg, rkey, ci.lkey_,
                          op_counter, size, signal, fence);
    internal::Post(seq_send_wrs[coro_idx][seq_idx].send_wrs.front().wr,
                   ci.conn_.get(), op_counter);
    seq_send_wrs[coro_idx][seq_idx].posted = true;
    std::vector<T> result;
    while (!internal::PollAsync(ci.conn_.get(), op_counter, ptr)) {
      co_yield std::suspend_always();
    }
    get_seq_op_result<T>(seq_idx, coro_idx, result);
    seq_send_wrs[coro_idx].erase(seq_idx);
    co_return result;
  }
};
}  // namespace remus