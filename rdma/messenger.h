#pragma once

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include "../../logging/logging.h"
#include "../../util/status_util.h"
#include "../logging/logging.h"
#include "../util/status_util.h"
#include "../vendor/sss/status.h"
#include "memory.h"
#include "messenger.h"
#include "util.h"

namespace rome::rdma {

// Contains a copy of a received message by a `RdmaChannel` for the caller of
// `RdmaChannel::TryDeliver`.
struct Message {
  std::unique_ptr<uint8_t[]> buffer;
  size_t length;
};

class RdmaMessenger {
public:
  virtual ~RdmaMessenger() = default;
  virtual sss::Status SendMessage(const Message &msg) = 0;
  virtual sss::StatusVal<Message> TryDeliverMessage() = 0;
};

// [mfs] Do we ever use this?
class EmptyRdmaMessenger : public RdmaMessenger {
public:
  ~EmptyRdmaMessenger() = default;
  explicit EmptyRdmaMessenger(rdma_cm_id *id) {}
  sss::Status SendMessage(const Message &msg) override {
    return sss::Status::Ok();
  }
  sss::StatusVal<Message> TryDeliverMessage() override {
    return {{sss::Ok, {}}, Message{nullptr, 0}};
  }
};

} // namespace rome::rdma

namespace rome::rdma {

template <uint32_t kCapacity = 1ul << 12, uint32_t kRecvMaxBytes = 1ul << 8>
class TwoSidedRdmaMessenger : public RdmaMessenger {
  static_assert(kCapacity % 2 == 0, "Capacity must be divisible by two.");

public:
  explicit TwoSidedRdmaMessenger(rdma_cm_id *id);

  sss::Status SendMessage(const Message &msg) override;

  // Attempts to deliver a sent message by checking for completed receives and
  // then returning a `Message` containing a copy of the received buffer.
  sss::StatusVal<Message> TryDeliverMessage() override;

private:
  // Memorry region IDs.
  static constexpr char kSendId[] = "send";
  static constexpr char kRecvId[] = "recv";

  // Reset the receive buffer and post `ibv_recv_wr` on the RQ. This should only
  // be called when all posted receives have corresponding completions,
  // otherwise there may be a race on memory by posted recvs.
  void PrepareRecvBuffer();

  // The remotely accessible memory used for the send and recv buffers.
  RdmaMemory rm_;

  // A pointer to the QP used to post sends and receives.
  rdma_cm_id *id_; //! NOT OWNED

  // Pointer to memory region identified by `kSendId`.
  ibv_mr *send_mr_;

  // Size of send buffer.
  const int send_capacity_;

  // Pointer to the base address and next unposted address of send buffer.
  uint8_t *send_base_, *send_next_;

  uint32_t send_total_;

  // Pointer to memory region identified by `kRecvId`.
  ibv_mr *recv_mr_;

  // Size of the recv buffer.
  const int recv_capacity_;

  // Pointer to base address and next unposted address.
  uint8_t *recv_base_, *recv_next_;

  // Tracks the number of completed receives. This is used by
  // `PrepareRecvBuffer()` to ensure that it is only called when all posted
  // receives have been completed.
  uint32_t recv_total_;
};

} // namespace rome::rdma

namespace rome::rdma {

template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::TwoSidedRdmaMessenger(
    rdma_cm_id *id)
    : rm_(kCapacity, id->pd), id_(id), send_capacity_(kCapacity / 2),
      send_total_(0), recv_capacity_(kCapacity / 2), recv_total_(0) {
  OK_OR_FAIL(rm_.RegisterMemoryRegion(kSendId, 0, send_capacity_));
  OK_OR_FAIL(rm_.RegisterMemoryRegion(kRecvId, send_capacity_, recv_capacity_));
  auto t1 = rm_.GetMemoryRegion(kSendId);
  STATUSVAL_OR_DIE(t1);
  send_mr_ = t1.val.value();
  auto t2 = rm_.GetMemoryRegion(kRecvId);
  STATUSVAL_OR_DIE(t2);
  recv_mr_ = t2.val.value();
  send_base_ = reinterpret_cast<uint8_t *>(send_mr_->addr);
  send_next_ = send_base_;
  recv_base_ = reinterpret_cast<uint8_t *>(recv_mr_->addr);
  recv_next_ = recv_base_;
  PrepareRecvBuffer();
}

template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
sss::Status TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::SendMessage(
    const Message &msg) {
  // The proto we send may not be larger than the maximum size received at the
  // peer. We assume that everyone uses the same value, so we check what we
  // know locally instead of doing something fancy to ask the remote node.
  if (msg.length >= kRecvMaxBytes) {
    sss::Status err = {sss::ResourceExhausted, ""};
    err << "Message too large: expected<=" << kRecvMaxBytes
        << ", actual=" << msg.length;
    return err;
  }

  // If the new message will not fit in remaining memory, then we reset the
  // head pointer to the beginning.
  auto tail = send_next_ + msg.length;
  auto end = send_base_ + send_capacity_;
  if (tail > end) {
    send_next_ = send_base_;
  }
  std::memcpy(send_next_, msg.buffer.get(), msg.length);

  // Copy the proto into the send buffer.
  ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uint64_t>(send_next_);
  sge.length = msg.length;
  sge.lkey = send_mr_->lkey;

  // Note that we use a custom `ibv_send_wr` here since we want to add an
  // immediate. Otherwise we could have just used `rdma_post_send()`.
  ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.num_sge = 1;
  wr.sg_list = &sge;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.wr_id = send_total_++;

  ibv_send_wr *bad_wr;
  {
    int ret = ibv_post_send(id_->qp, &wr, &bad_wr);
    if (ret != 0) {
      sss::Status err = {sss::InternalError, ""};
      err << "ibv_post_send(): " << strerror(errno);
      return err;
    }
  }

  // Assumes that the CQ associated with the SQ is synchronous.
  ibv_wc wc;
  int comps = rdma_get_send_comp(id_, &wc);
  while (comps < 0 && errno == EAGAIN) {
    comps = rdma_get_send_comp(id_, &wc);
  }

  if (comps < 0) {
    sss::Status e = {sss::InternalError, {}};
    e << "rdma_get_send_comp: {}" << strerror(errno);
    return e;
  } else if (wc.status != IBV_WC_SUCCESS) {
    sss::Status e = {sss::InternalError, {}};
    e << "rdma_get_send_comp(): " << ibv_wc_status_str(wc.status);
    return e;
  }

  send_next_ += msg.length;
  return sss::Status::Ok();
}

// Attempts to deliver a sent message by checking for completed receives and
// then returning a `Message` containing a copy of the received buffer.
template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
sss::StatusVal<Message>
TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::TryDeliverMessage() {
  ibv_wc wc;
  auto ret = rdma_get_recv_comp(id_, &wc);
  if (ret < 0 && errno != EAGAIN) {
    sss::Status e = {sss::InternalError, {}};
    e << "rdma_get_recv_comp: " << strerror(errno);
    return {e, {}};
  } else if (ret < 0 && errno == EAGAIN) {
    return {{sss::Unavailable, "Retry"}, {}};
  } else {
    switch (wc.status) {
    case IBV_WC_WR_FLUSH_ERR:
      return {{sss::Aborted, "QP in error state"}, {}};
    case IBV_WC_SUCCESS: {
      // Prepare the response.
      sss::StatusVal<Message> res = {sss::Status::Ok(), {}};
      res.val->buffer = std::make_unique<uint8_t[]>(wc.byte_len);
      std::memcpy(res.val->buffer.get(), recv_next_, wc.byte_len);
      res.val->length = wc.byte_len;

      // If the tail reached the end of the receive buffer then all posted
      // wrs have been consumed and we can post new ones.
      // `PrepareRecvBuffer` also handles resetting `recv_next_` to point to
      // the base address of the receive buffer.
      recv_total_++;
      recv_next_ += kRecvMaxBytes;
      if (recv_next_ > recv_base_ + (recv_capacity_ - kRecvMaxBytes)) {
        PrepareRecvBuffer();
      }
      return res;
    }
    default: {
      sss::Status err = {sss::InternalError, {}};
      err << "rdma_get_recv_comp(): " << ibv_wc_status_str(wc.status);
      return {err, {}};
    }
    }
  }
}

template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
void TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::PrepareRecvBuffer() {
  ROME_ASSERT(recv_total_ % (recv_capacity_ / kRecvMaxBytes) == 0,
              "Unexpected number of completions from RQ");
  // Prepare the recv buffer for incoming messages with the assumption that
  // the maximum received message will be `max_recv_` bytes long.
  for (auto curr = recv_base_;
       curr <= recv_base_ + (recv_capacity_ - kRecvMaxBytes);
       curr += kRecvMaxBytes) {
    RDMA_CM_ASSERT(rdma_post_recv, id_, nullptr, curr, kRecvMaxBytes, recv_mr_);
  }
  recv_next_ = recv_base_;
}

} // namespace rome::rdma