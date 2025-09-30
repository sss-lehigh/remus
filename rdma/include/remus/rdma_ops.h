#pragma once

#include <atomic>
#include <cstdint>

#include "connection.h"
#include "rdma_ptr.h"

// TODO:  Could we have async versions of the one-sided operations?
//
// TODO:  We might need a better way of polling the completion queue
namespace remus::internal {

/// utility function for configuring a one-sided read over RDMA
///
/// @tparam T TODO
/// @param send_wr
/// @param sge
/// @param ptr
/// @param seg
/// @param rkey
/// @param lkey
/// @param ack
/// @param size
/// @param signal
/// @param fence
template <typename T>
inline void
ReadConfig(std::shared_ptr<ibv_send_wr> send_wr, std::shared_ptr<ibv_sge> sge,
           rdma_ptr<T> ptr, uint8_t *seg, int32_t rkey, int32_t lkey,
           std::atomic<int> *ack, size_t size, bool signal, bool fence) {
  T *local = (T *)seg;

  sge->addr = reinterpret_cast<uint64_t>(local);
  sge->length = size;
  sge->lkey = lkey;

  send_wr->wr_id = (uint64_t)ack;
  send_wr->num_sge = 1;
  send_wr->sg_list = sge.get();
  send_wr->opcode = IBV_WR_RDMA_READ;
  send_wr->send_flags =
      (fence ? IBV_SEND_FENCE : 0) | (signal ? IBV_SEND_SIGNALED : 0);
  send_wr->wr.rdma.remote_addr = ptr.address();
  send_wr->wr.rdma.rkey = rkey;
}

/// utility function for configuring a one-sided write over RDMA
///
/// @tparam T
/// @param send_wr
/// @param sge
/// @param ptr
/// @param val
/// @param seg
/// @param rkey
/// @param lkey
/// @param ack
/// @param size
/// @param signal
/// @param fence
template <typename T>
inline void WriteConfig(std::shared_ptr<ibv_send_wr> send_wr,
                        std::shared_ptr<ibv_sge> sge, rdma_ptr<T> ptr,
                        const T &val, uint8_t *seg, int32_t rkey, int32_t lkey,
                        std::atomic<int> *ack, size_t size, bool signal,
                        bool fence) {
  T *local = (T *)seg;

  REMUS_ASSERT((uint64_t)local != ptr.address(), "WTF");
  // TODO: Is this memset really necessary?  Maybe for gaps in structs?
  std::memset(local, 0, size);
  *local = val;
  sge->addr = reinterpret_cast<uint64_t>(local);
  sge->length = size;
  sge->lkey = lkey;

  send_wr->wr_id = (uint64_t)ack;
  send_wr->num_sge = 1;
  send_wr->sg_list = sge.get();
  send_wr->opcode = IBV_WR_RDMA_WRITE;
  send_wr->send_flags =
      (signal ? IBV_SEND_SIGNALED : 0) | (fence ? IBV_SEND_FENCE : 0);
  send_wr->wr.rdma.remote_addr = ptr.address();
  send_wr->wr.rdma.rkey = rkey;
}

/// utility function for configuring a one-sided write over RDMA, but seg is
/// already set up
///
/// @tparam T
/// @param send_wr
/// @param sge
/// @param ptr
/// @param seg
/// @param rkey
/// @param lkey
/// @param ack
/// @param size
/// @param signal
/// @param fence
template <typename T>
void WriteConfig(std::shared_ptr<ibv_send_wr> send_wr,
                 std::shared_ptr<ibv_sge> sge, rdma_ptr<T> ptr, uint8_t *seg,
                 int32_t rkey, int32_t lkey, std::atomic<int> *ack, size_t size,
                 bool signal, bool fence) {
  T *local = (T *)seg;
  REMUS_ASSERT((uint64_t)local != ptr.address(), "WTF");
  sge->addr = reinterpret_cast<uint64_t>(local);
  sge->length = size;
  sge->lkey = lkey;

  send_wr->wr_id = (uint64_t)ack;
  send_wr->num_sge = 1;
  send_wr->sg_list = sge.get();
  send_wr->opcode = IBV_WR_RDMA_WRITE;
  send_wr->send_flags =
      (signal ? IBV_SEND_SIGNALED : 0) | (fence ? IBV_SEND_FENCE : 0);
  send_wr->wr.rdma.remote_addr = ptr.address();
  send_wr->wr.rdma.rkey = rkey;
}

/// utility function for configuring a one-sided compare and swap over RDMA
///
/// @tparam T
/// @param send_wr
/// @param sge
/// @param ptr
/// @param expected
/// @param swap
/// @param prev_
/// @param rkey
/// @param lkey
/// @param ack
/// @param signal
/// @param fence
template <typename T>
  requires(sizeof(T) <= 8)
inline void CompareAndSwapConfig(std::shared_ptr<ibv_send_wr> send_wr,
                                 std::shared_ptr<ibv_sge> sge, rdma_ptr<T> ptr,
                                 uint64_t expected, uint64_t swap,
                                 uint64_t *prev_, int32_t rkey, int32_t lkey,
                                 std::atomic<int> *ack, bool signal,
                                 bool fence) {

  sge->addr = reinterpret_cast<uint64_t>(prev_);
  sge->length = sizeof(uint64_t);
  sge->lkey = lkey;

  send_wr->wr_id = (uint64_t)ack;
  send_wr->num_sge = 1;
  send_wr->sg_list = sge.get();
  send_wr->opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  send_wr->send_flags =
      (signal ? IBV_SEND_SIGNALED : 0) | (fence ? IBV_SEND_FENCE : 0);
  send_wr->wr.atomic.remote_addr = ptr.address();
  send_wr->wr.atomic.rkey = rkey;
  send_wr->wr.atomic.compare_add = expected;
  send_wr->wr.atomic.swap = swap;
}

/// utility function for configuring a one-sided fetch and add over RDMA
///
/// @tparam T
/// @param send_wr
/// @param sge
/// @param ptr
/// @param add
/// @param prev
/// @param rkey
/// @param lkey
/// @param ack
/// @param signal
/// @param fence
template <typename T>
  requires(sizeof(T) <= 8)
inline void FetchAndAddConfig(std::shared_ptr<ibv_send_wr> send_wr,
                              std::shared_ptr<ibv_sge> sge, rdma_ptr<T> ptr,
                              uint64_t add, uint64_t *prev, int32_t rkey,
                              int32_t lkey, std::atomic<int> *ack, bool signal,
                              bool fence) {
  sge->addr = reinterpret_cast<uint64_t>(prev);
  sge->length = sizeof(uint64_t);
  sge->lkey = lkey;

  send_wr->wr_id = (uint64_t)ack;
  send_wr->num_sge = 1;
  send_wr->sg_list = sge.get();
  send_wr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  send_wr->send_flags =
      (signal ? IBV_SEND_SIGNALED : 0) | (fence ? IBV_SEND_FENCE : 0);
  send_wr->wr.atomic.remote_addr = ptr.address();
  send_wr->wr.atomic.rkey = rkey;
  send_wr->wr.atomic.compare_add = add;
}

/// utility function for performing a one-sided read over RDMA
///
/// @param send_wr
/// @param conn
/// @param ack
inline void Post(std::shared_ptr<ibv_send_wr> send_wr, Connection *conn,
          std::atomic<int> *ack) {
  *ack = 1;
  conn->send_onesided(send_wr.get());
}

/// utility function for polling the completion queue
///
/// @tparam T
/// @param conn
/// @param ack
/// @param ptr
template <typename T>
inline void Poll(Connection *conn, std::atomic<int> *ack, rdma_ptr<T> ptr) {
  // Poll until we match on the condition
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    // TODO: Why doesn't write look for poll != 1, but this does?
    if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                   "ibv_poll_cq(): {} @ {}",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   format_rdma_ptr(ptr));
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }
}

/// utility function for polling the completion queue asynchronously
///
/// @tparam T
/// @param conn
/// @param ack
/// @param ptr
/// @return
template <typename T>
bool PollAsync(Connection *conn, std::atomic<int> *ack, rdma_ptr<T> ptr) {
  // Poll until we match on the condition
  ibv_wc wc;
  if (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      return false;
    // TODO: Why doesn't write look for poll != 1, but this does?
    if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(poll == 1 && wc.status == IBV_WC_SUCCESS,
                   "ibv_poll_cq(): {} @ {}",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   format_rdma_ptr(ptr));
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
    return false;
  }
  return true;
}

} // namespace remus::internal
