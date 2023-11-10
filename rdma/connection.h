#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <limits>
#include <memory>
#include <netdb.h>
#include <optional>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <string>
#include <vector>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"
#include "memory_pool.h"

namespace rome::rdma::internal {

/// Connection encapsulates an RDMA communication identifier between two
/// endpoints.  It is created when one node connects to a listening endpoint on
/// another node, but once created, it is oblivious to whether it was created by
/// the listening side or not.
///
/// Connections are a fundamental aspect of RDMA interaction.  It is necessary
/// to have a connection between two machines before one-sided operations can be
/// issued between those machines. A side effect is that the connection allows
/// two-sided operations between those machines.  This is realized in our design
/// by a constructed connection only having public methods to send, receive,
/// post one-sided write requests, and poll for completions.
///
/// TODO: This API is in flux.
class Connection {
  /// A byte stream, representing data sent or received via two-sided
  /// communication.
  ///
  /// NB: The use of a unique_ptr ensures that the memory gets cleaned up when
  ///     the Message goes out of scope.
  ///
  /// TODO: We need to think more carefully about the design.  Right now, the
  ///       public interface of Send/Recv only allow for protos.  It also seems
  ///       that those protos have a fixed maximum size, which is determined by
  ///       the sizes of pinned memory regions.  Thus it might be possible for
  ///       threads to pre-allocate a small number of Messages, and not need the
  ///       allocator overhead that this Message type introduces.
  struct Message {
    std::unique_ptr<uint8_t[]> buffer;
    size_t length;
  };

  // [mfs] TODO: Make these arguments to the ctor?
  static constexpr const uint32_t kCapacity = 1ul << 12;
  static constexpr const uint32_t kRecvMaxBytes = 1ul << 8;
  static_assert(kCapacity % 2 == 0, "Capacity must be divisible by two.");

  RdmaMemory rm_;           // Remotely accessible memory for send/recv buffers.
  rdma_cm_id *id_;          // Pointer to the QP for sends/receives
  ibv_mr *send_mr_;         // Memory region identified by `kSendId`
  const int send_cap_;      // Capacity (in bytes) of the send buffer
  uint8_t *send_base_;      // Base address of send buffer
  uint8_t *send_next_;      // Next un-posted address within send buffer
  uint32_t send_total_ = 0; // Number of sends; also generates send ids
  ibv_mr *recv_mr_;         // Memory region identified by `kRecvId`
  const int recv_cap_;      // Capacity (in bytes) of recv buffer
  uint8_t *recv_base_;      // Base address of recv buffer
  uint8_t *recv_next_;      // Next unposted address within recv buffer
  uint32_t recv_total_ = 0; // Completed receives; helps track completion

  // Memory region IDs.
  //
  // TODO: Refactor memory pool so that we don't need these
  static constexpr char kSendId[] = "send";
  static constexpr char kRecvId[] = "recv";

  // [mfs] These should probably be template parameters
  static constexpr size_t kMemoryPoolMessengerCapacity = 1 << 12;
  static constexpr size_t kMemoryPoolMessageSize = 1 << 8;

  uint32_t src_id_;
  uint32_t dst_id_;

  sss::Status SendMessage(const Message &msg) {
    // The proto we send may not be larger than the maximum size received at
    // the peer. We assume that everyone uses the same value, so we check
    // what we know locally instead of doing something fancy to ask the
    // remote node.
    if (msg.length >= kRecvMaxBytes) {
      sss::Status err = {sss::ResourceExhausted, ""};
      err << "Message too large: expected<=" << kRecvMaxBytes
          << ", actual=" << msg.length;
      return err;
    }

    // If the new message will not fit in remaining memory, then we reset
    // the head pointer to the beginning.
    auto tail = send_next_ + msg.length;
    auto end = send_base_ + send_cap_;
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
      return e << "rdma_get_send_comp: {}" << strerror(errno);
    } else if (wc.status != IBV_WC_SUCCESS) {
      sss::Status e = {sss::InternalError, {}};
      return e << "rdma_get_send_comp(): " << ibv_wc_status_str(wc.status);
    }

    send_next_ += msg.length;
    return sss::Status::Ok();
  }

  // Attempts to deliver a sent message by checking for completed receives
  // and then returning a `Message` containing a copy of the received
  // buffer.
  sss::StatusVal<Message> TryDeliverMessage() {
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
        //
        // [mfs] It looks like there's a lot of copying baked into the API?
        // [esl] I was getting an error with the code that was previously
        // here. The optional was throwing a bad_optional_access exception.
        // I think the optional had to be constructed with a value. It might
        // be better to refactor Message to make it moveable and construct
        // in a series of steps instead of one long statement...
        sss::StatusVal<Message> res = {
            sss::Status::Ok(),
            std::make_optional(Message{std::make_unique<uint8_t[]>(wc.byte_len),
                                       wc.byte_len})};
        std::memcpy(res.val->buffer.get(), recv_next_, wc.byte_len);
        ROME_TRACE("{} {}", res.val->buffer.get(), res.val->length);

        // If the tail reached the end of the receive buffer then all posted
        // wrs have been consumed and we can post new ones.
        // `PrepareRecvBuffer` also handles resetting `recv_next_` to point
        // to the base address of the receive buffer.
        recv_total_++;
        recv_next_ += kRecvMaxBytes;
        if (recv_next_ > recv_base_ + (recv_cap_ - kRecvMaxBytes)) {
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

  // Reset the receive buffer and post `ibv_recv_wr` on the RQ. This should
  // only be called when all posted receives have corresponding completions,
  // otherwise there may be a race on memory by posted recvs.
  void PrepareRecvBuffer() {
    ROME_ASSERT(recv_total_ % (recv_cap_ / kRecvMaxBytes) == 0,
                "Unexpected number of completions from RQ");
    // Prepare the recv buffer for incoming messages with the assumption
    // that the maximum received message will be `max_recv_` bytes long.
    for (auto curr = recv_base_;
         curr <= recv_base_ + (recv_cap_ - kRecvMaxBytes);
         curr += kRecvMaxBytes) {
      RDMA_CM_ASSERT(rdma_post_recv, id_, nullptr, curr, kRecvMaxBytes,
                     recv_mr_);
    }
    recv_next_ = recv_base_;
  }

  template <typename ProtoType> sss::StatusVal<ProtoType> TryDeliver() {
    auto msg_or = TryDeliverMessage();
    if (msg_or.status.t == sss::Ok) {
      ProtoType proto;
      proto.ParseFromArray(msg_or.val.value().buffer.get(),
                           msg_or.val.value().length);
      return {sss::Status::Ok(), proto};
    } else {
      return {msg_or.status, {}};
    }
  }

public:
  // [mfs] It seems to me that the savings involved in splitting rm_ is not
  // worth the complexity?
  Connection(uint32_t src_id, uint32_t dst_id, rdma_cm_id *channel_id)
      : rm_(kCapacity, channel_id->pd), id_(channel_id),
        send_cap_(kCapacity / 2), recv_cap_(kCapacity / 2), src_id_(src_id),
        dst_id_(dst_id) {
    send_mr_ = (rm_.RegisterMemoryRegion(kSendId, 0, send_cap_));
    recv_mr_ = (rm_.RegisterMemoryRegion(kRecvId, send_cap_, recv_cap_));
    send_base_ = reinterpret_cast<uint8_t *>(send_mr_->addr);
    send_next_ = send_base_;
    recv_base_ = reinterpret_cast<uint8_t *>(recv_mr_->addr);
    recv_next_ = recv_base_;
    PrepareRecvBuffer();
  }

  Connection(const Connection &) = delete;
  Connection(Connection &&c) = delete;

  template <typename ProtoType> sss::Status Send(const ProtoType &proto) {
    // two callers.  One is fail-stop.  The other is never called in IHT.  Can
    // this be fail-stop?
    Message msg{std::make_unique<uint8_t[]>(proto.ByteSizeLong()),
                proto.ByteSizeLong()};
    proto.SerializeToArray(msg.buffer.get(), msg.length);
    return SendMessage(msg);
  }

  template <typename ProtoType> sss::StatusVal<ProtoType> Deliver() {
    auto p = this->TryDeliver<ProtoType>();
    while (p.status.t == sss::Unavailable) {
      p = this->TryDeliver<ProtoType>();
    }
    return p;
  }

  /// This used to be a "cleanup" lambda used by ~ConnectionManager.  The key is
  /// the key of the established_ map, and the caller is the ConnectionManager's
  /// my_id_ field.
  ///
  /// TODO: Document this better  In particular, what are key and caller?  Are
  ///       they indicating that we actually do know who is the listening side?
  void cleanup(uint32_t key, uint32_t caller) {
    // A loopback connection is made manually, so we do not need to deal with
    // the regular `rdma_cm` handling. Similarly, we avoid destroying the
    // event channel below since it is destroyed along with the id.
    if (key != caller) {
      rdma_disconnect(id_);
      rdma_cm_event *event;
      auto result = rdma_get_cm_event(id_->channel, &event);
      while (result == 0) {
        RDMA_CM_ASSERT(rdma_ack_cm_event, event);
        result = rdma_get_cm_event(id_->channel, &event);
      }
    }

    // We only allocate contexts for connections that were created by the
    // `RdmaReceiver` callbacks. Otherwise, we created an event channel so
    // that we could asynchronously connect (and avoid distributed deadlock).
    auto *context = id_->context;
    auto *channel = id_->channel;
    rdma_destroy_ep(id_);

    if (key != caller && context != nullptr) {
      free(context);
    } else if (key != caller) {
      rdma_destroy_event_channel(channel);
    }
  }

  /// Test if this Connection's ID matches some other Connection's ID.  I don't
  /// understand why ConnectionManager::OnDisconnect needs this.
  bool matches(rdma_cm_id *other) { return id_ == other; }

  /// Send a write request.  This encapsulates so that id_ can be private
  void send_onesided(ibv_send_wr *send_wr_) {
    ibv_send_wr *bad = nullptr;
    RDMA_CM_ASSERT(ibv_post_send, id_->qp, send_wr_, &bad);
  }

  /// Poll to see if anything new arrived on the completion queue
  int poll_cq(int num, ibv_wc *wc) {
    return ibv_poll_cq(id_->send_cq, num, wc);
  }
};
} // namespace rome::rdma::internal