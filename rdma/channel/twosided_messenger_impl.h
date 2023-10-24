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
      // [esl] I was getting an error with the code that was previously here. The optional was throwing a bad_optional_access exception. I think the optional had to be constructed with a value. It might be better to refactor Message to make it moveable and construct in a series of steps instead of one long statement...
      sss::StatusVal<Message> res = {sss::Status::Ok(), std::make_optional(Message{std::make_unique<uint8_t[]>(wc.byte_len), wc.byte_len})};
      std::memcpy(res.val->buffer.get(), recv_next_, wc.byte_len);
      ROME_TRACE("{} {}", res.val->buffer.get(), res.val->length);

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