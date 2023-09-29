#pragma once

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include "../../logging/logging.h"
#include "../../util/status_util.h"
#include "../channel/twosided_messenger.h"
#include "../rdma_memory.h"
#include "../rdma_util.h"
#include "rdma_messenger.h"

namespace rome::rdma {

using ::util::InternalErrorBuilder;
using ::util::ResourceExhaustedErrorBuilder;

template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::TwoSidedRdmaMessenger(
    rdma_cm_id *id)
    : rm_(kCapacity, id->pd), id_(id), send_capacity_(kCapacity / 2),
      send_total_(0), recv_capacity_(kCapacity / 2), recv_total_(0) {
  ROME_ASSERT_OK(rm_.RegisterMemoryRegion(kSendId, 0, send_capacity_));
  ROME_ASSERT_OK(
      rm_.RegisterMemoryRegion(kRecvId, send_capacity_, recv_capacity_));
  send_mr_ = VALUE_OR_DIE(rm_.GetMemoryRegion(kSendId));
  recv_mr_ = VALUE_OR_DIE(rm_.GetMemoryRegion(kRecvId));
  send_base_ = reinterpret_cast<uint8_t *>(send_mr_->addr);
  send_next_ = send_base_;
  recv_base_ = reinterpret_cast<uint8_t *>(recv_mr_->addr);
  recv_next_ = recv_base_;
  PrepareRecvBuffer();
}

template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
absl::Status TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::SendMessage(
    const Message &msg) {
  // The proto we send may not be larger than the maximum size received at the
  // peer. We assume that everyone uses the same value, so we check what we
  // know locally instead of doing something fancy to ask the remote node.
  ROME_CHECK_QUIET(ROME_RETURN(ResourceExhaustedErrorBuilder()
                               << "Message too large: expected<="
                               << kRecvMaxBytes << ", actual=" << msg.length),
                   msg.length < kRecvMaxBytes);

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
  RDMA_CM_CHECK(ibv_post_send, id_->qp, &wr, &bad_wr);

  // Assumes that the CQ associated with the SQ is synchronous.
  ibv_wc wc;
  int comps = rdma_get_send_comp(id_, &wc);
  while (comps < 0 && errno == EAGAIN) {
    comps = rdma_get_send_comp(id_, &wc);
  }

  if (comps < 0) {
    return util::InternalErrorBuilder()
           << "rdma_get_send_comp: {}" << strerror(errno);
  } else if (wc.status != IBV_WC_SUCCESS) {
    return util::InternalErrorBuilder()
           << "rdma_get_send_comp(): " << ibv_wc_status_str(wc.status);
  }

  send_next_ += msg.length;
  return absl::OkStatus();
}

// Attempts to deliver a sent message by checking for completed receives and
// then returning a `Message` containing a copy of the received buffer.
template <uint32_t kCapacity, uint32_t kRecvMaxBytes>
absl::StatusOr<Message>
TwoSidedRdmaMessenger<kCapacity, kRecvMaxBytes>::TryDeliverMessage() {
  ibv_wc wc;
  auto ret = rdma_get_recv_comp(id_, &wc);
  if (ret < 0 && errno != EAGAIN) {
    return util::InternalErrorBuilder()
           << "rdma_get_recv_comp: " << strerror(errno);
  } else if (ret < 0 && errno == EAGAIN) {
    return absl::UnavailableError("Retry");
  } else {
    switch (wc.status) {
    case IBV_WC_WR_FLUSH_ERR:
      return absl::AbortedError("QP in error state");
    case IBV_WC_SUCCESS: {
      // Prepare the response.
      Message msg;
      msg.buffer = std::make_unique<uint8_t[]>(wc.byte_len);
      std::memcpy(msg.buffer.get(), recv_next_, wc.byte_len);
      msg.length = wc.byte_len;

      // If the tail reached the end of the receive buffer then all posted
      // wrs have been consumed and we can post new ones.
      // `PrepareRecvBuffer` also handles resetting `recv_next_` to point to
      // the base address of the receive buffer.
      recv_total_++;
      recv_next_ += kRecvMaxBytes;
      if (recv_next_ > recv_base_ + (recv_capacity_ - kRecvMaxBytes)) {
        PrepareRecvBuffer();
      }
      return msg;
    }
    default:
      return InternalErrorBuilder()
             << "rdma_get_recv_comp(): " << ibv_wc_status_str(wc.status);
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