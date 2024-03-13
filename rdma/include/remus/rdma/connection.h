#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <ranges>
#include <span>

#include "connection_utils.h"
#include "segment.h"

// TODO: Many parts of this file need better documentation

namespace remus::rdma::internal {

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
class Connection {
  /// A byte stream representing data sent or received via two-sided
  /// communication.
  ///
  /// NB: The use of a unique_ptr ensures that the memory gets cleaned up when
  ///     the Message goes out of scope.
  ///
  /// TODO: We need to think more carefully about the design.  Right now, the
  ///       public interface of Send/Recv only allows for transmitting protos.
  ///       It also seems that those protos have a fixed maximum size, which is
  ///       determined by the sizes of pinned memory regions.  
  ///
  /// TODO: OTOH, we don't really use Send/Recv much, so maybe it doesn't
  ///       matter?

  // TODO: Should these constants migrate to connection_utils.h?

  /// The maximum size of a message sent or received
  ///
  /// TODO: Should we enforce that this evenly divides kCapacity?
  ///
  /// TODO: Is this even necessary?  How do we know how big it should be?
  ///       When/where/how do we nicely enforce this maximum message size?
  ///
  /// TODO: Make this an argument to the constructor?
  static constexpr const uint32_t kRecvMaxBytes = 1ul << 8;

  /// TODO: I don't understand why this is separate from kCapacity
  ///
  /// TODO: Make this an argument to the constructor?
  static constexpr size_t kMemoryPoolMessengerCapacity = 1 << 12;

  /// TODO: I don't understand why this is separate from kRecvMaxBytes
  ///
  /// TODO: Make this an argument to the constructor?
  static constexpr size_t kMemoryPoolMessageSize = 1 << 8;

  rdma_cm_id *id_;          // Pointer to the QP for sends/receives
  Segment send_seg_;        // Remotely accessible memory for send buffer
  uint8_t *send_base_;      // Base address of send buffer
  uint8_t *send_next_;      // Next un-posted address within send buffer
  uint32_t send_total_ = 0; // Number of sends; also generates send ids
  Segment recv_seg_;        // Remotely accessible memory for recv buffer
  uint8_t *recv_base_;      // Base address of recv buffer
  uint8_t *recv_next_;      // Next un-posted address within recv buffer
  uint32_t recv_total_ = 0; // Completed receives; helps track completion

  /// Internal method for sending a Message (byte array) over RDMA as a
  /// two-sided operation.
  ///
  /// TODO: Should this be fail-stop?
  ///
  /// TODO: Why do we enforce length so late?
  ///
  /// TODO: Why not let the caller deal with protos, instead of using them
  ///       internally?
  ///
  /// TODO: Should this be fail-stop?
  template<typename T>
    requires std::ranges::contiguous_range<T>
  remus::util::Status SendMessage(T&& msg) {

    static_assert(std::ranges::sized_range<T>);

    auto msg_ptr = std::ranges::data(std::forward<T>(msg));
    auto msg_elements = std::ranges::size(std::forward<T>(msg)); // elements
    auto msg_size = msg_elements * sizeof(decltype(*msg_ptr)); // bytes

    // The proto we send may not be larger than the maximum size received at
    // the peer. We assume that everyone uses the same value, so we check
    // what we know locally instead of doing something fancy to ask the
    // remote node.
    if (msg_size >= kRecvMaxBytes) {
      remus::util::Status err = {remus::util::ResourceExhausted, ""};
      err << "Message too large: expected<=" << kRecvMaxBytes
          << ", actual=" << msg_size;
      return err;
    }

    // If the new message will not fit in remaining memory, then we reset
    // the head pointer to the beginning.
    //
    // TODO: Are we sure there are no pending completions on it?
    auto tail = send_next_ + msg_size;
    auto end = send_base_ + kCapacity;
    if (tail > end) {
      send_next_ = send_base_;
    }
    std::memcpy(send_next_, msg_ptr, msg_size); // TODO can this be subsituted with std::ranges::copy?

    // Copy the proto into the send buffer.
    ibv_sge sge;
    std::memset(&sge, 0, sizeof(sge));
    sge.addr = reinterpret_cast<uint64_t>(send_next_);
    sge.length = msg_size;
    sge.lkey = send_seg_.mr()->lkey;

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
    if (ibv_post_send(id_->qp, &wr, &bad_wr) != 0) {
      remus::util::Status err = {remus::util::InternalError, ""};
      err << "ibv_post_send(): " << strerror(errno);
      return err;
    }

    // Assumes that the CQ associated with the SQ is synchronous.
    //
    // TODO:  This doesn't check if the *caller*'s event completed, just that
    //        *some* event completed.  Is that OK?
    ibv_wc wc;
    int comps = rdma_get_send_comp(id_, &wc);
    while (comps < 0 && errno == EAGAIN) {
      comps = rdma_get_send_comp(id_, &wc);
    }
    if (comps < 0) {
      remus::util::Status e = {remus::util::InternalError, {}};
      return e << "rdma_get_send_comp: {}" << strerror(errno);
    } else if (wc.status != IBV_WC_SUCCESS) {
      remus::util::Status e = {remus::util::InternalError, {}};
      return e << "rdma_get_send_comp(): " << ibv_wc_status_str(wc.status);
    }
    send_next_ += msg_size;
    return remus::util::Status::Ok();
  }

  /// Internal method for receiving a Message (byte array) over RDMA as a
  /// two-sided operation.
  /// TODO is it possible we already know the message size we expect to serialize?
  remus::util::StatusVal<std::vector<uint8_t>> TryDeliverMessage() {
    ibv_wc wc;
    auto ret = rdma_get_recv_comp(id_, &wc);
    if (ret < 0 && errno != EAGAIN) {
      remus::util::Status e = {remus::util::InternalError, {}};
      e << "rdma_get_recv_comp: " << strerror(errno);
      return {e, {}};
    } else if (ret < 0 && errno == EAGAIN) {
      return {{remus::util::Unavailable, "Retry"}, {}};
    } else {
      switch (wc.status) {
      case IBV_WC_WR_FLUSH_ERR:
        return {{remus::util::Aborted, "QP in error state"}, {}};
      case IBV_WC_SUCCESS: {

        // [dsm] This may be fine now?
        // vvvvv

        // Prepare the response.
        //
        // [mfs] It looks like there's a lot of copying baked into the API?
        //
        // [esl]  I was getting an error with the code that was previously here.
        //        The optional was throwing a bad_optional_access exception. I
        //        think the optional had to be constructed with a value. It
        //        might be better to refactor Message to make it moveable and
        //        construct in a series of steps instead of one long
        //        statement...

        std::span<uint8_t> recv_span{recv_next_, wc.byte_len};

        remus::util::StatusVal<std::vector<uint8_t>> res = {
            remus::util::Status::Ok(),
            std::make_optional(std::vector<uint8_t>(recv_span.begin(), recv_span.end()))};

        REMUS_TRACE("{} {}", res.val->data(), res.val->size());

        // If the tail reached the end of the receive buffer then all posted
        // wrs have been consumed and we can post new ones.
        // `PrepareRecvBuffer` also handles resetting `recv_next_` to point
        // to the base address of the receive buffer.
        recv_total_++;
        recv_next_ += kRecvMaxBytes;
        if (recv_next_ > recv_base_ + (kCapacity - kRecvMaxBytes)) {
          PrepareRecvBuffer();
        }
        return res;
      }
      default: {
        remus::util::Status err = {remus::util::InternalError, {}};
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
    REMUS_ASSERT(recv_total_ % (kCapacity / kRecvMaxBytes) == 0,
                "Unexpected number of completions from RQ");
    // Prepare the recv buffer for incoming messages with the assumption
    // that the maximum received message will be `max_recv_` bytes long.
    for (auto curr = recv_base_;
         curr <= recv_base_ + (kCapacity - kRecvMaxBytes);
         curr += kRecvMaxBytes) {
      RDMA_CM_ASSERT(rdma_post_recv, id_, nullptr, curr, kRecvMaxBytes,
                     recv_seg_.mr());
    }
    recv_next_ = recv_base_;
  }

  template <typename ProtoType> 
  remus::util::StatusVal<ProtoType> TryDeliver() {
    remus::util::StatusVal<std::vector<uint8_t>> msg_or = TryDeliverMessage();
    if (msg_or.status.t == remus::util::Ok) {
      ProtoType proto;
      proto.ParseFromArray(msg_or.val.value().data(),
                           msg_or.val.value().size());
      return {remus::util::Status::Ok(), proto};
    } else {
      return {msg_or.status, {}};
    }
  }

  remus::util::StatusVal<std::vector<uint8_t>> TryDeliverBytes() {
    return TryDeliverMessage();
  }

public:
  /// Construct a connection object
  Connection(uint32_t src_id, uint32_t dst_id, rdma_cm_id *channel_id)
      : send_seg_(kCapacity, channel_id->pd),
        recv_seg_(kCapacity, channel_id->pd), id_(channel_id) {
    // [mfs]  There's a secret rule here, that the send/recv are using the same
    //        pd as the channel.  Document it?
    send_next_ = send_base_ = reinterpret_cast<uint8_t *>(send_seg_.mr()->addr);
    recv_next_ = recv_base_ = reinterpret_cast<uint8_t *>(recv_seg_.mr()->addr);
    PrepareRecvBuffer();
  }

  Connection(const Connection &) = delete;
  Connection(Connection &&c) = delete;

  template <typename ProtoType> 
  remus::util::Status Send(const ProtoType &proto) {
    // two callers.  One is fail-stop.  The other is never called in IHT.  Can
    // this be fail-stop?
    std::string str = proto.SerializeAsString();
    return SendMessage(str);
  }

  template <typename T> 
    requires std::ranges::contiguous_range<T>
  remus::util::Status Send(T&& msg) {
    return SendMessage(std::forward<T>(msg));
  }

  // TODO: rename to Recv?
  template <typename ProtoType> 
  remus::util::StatusVal<ProtoType> Deliver() {
    auto p = this->TryDeliver<ProtoType>();
    while (p.status.t == remus::util::Unavailable) {
      p = this->TryDeliver<ProtoType>();
    }
    return p;
  }

  remus::util::StatusVal<std::vector<uint8_t>> DeliverBytes() {
    auto p = this->TryDeliverBytes();
    while (p.status.t == remus::util::Unavailable) {
      p = this->TryDeliverBytes();
    }
    return p;
  }

  /// This used to be a "cleanup" lambda used by ~ConnectionManager.  The key is
  /// the key of the established_ map, and the caller is the ConnectionManager's
  /// my_id_ field.
  ///
  /// TODO: Document this better  In particular, what are key and caller?  Are
  ///       they indicating that we actually do know who is the listening side?
  ///
  /// TODO: Could this be a destructor?
  void cleanup(uint32_t key, uint32_t caller) {
    // A loopback connection is made manually, so we do not need to deal with
    // the regular `rdma_cm` handling. Similarly, we avoid destroying the
    // event channel below since it is destroyed along with the id.
    if (key != caller) {
      rdma_disconnect(id_);
      rdma_cm_event *event;
      auto result = rdma_get_cm_event(id_->channel, &event);
      while (result == 0) {
        // TODO: Stop using RDMA_CM_ASSERT?
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
    // TODO:  I feel like we should be calling rdma_destroy_event_channel in the
    //        if and in the else if, but that leads to occasional double frees.
    //        Is there a race somewhere?  Note that the destructor used to hold
    //        a lock while making all calls to cleanup, so maybe there's
    //        something strange leading to the destructor getting called more
    //        than once?
    if (context != nullptr)
      free(context);
    else if (key != caller)
      rdma_destroy_event_channel(channel);
  }

  /// Send a write request.  This encapsulates so that id_ can be private
  ///
  /// TODO: Can we remove this?
  void send_onesided(ibv_send_wr *send_wr_) {
    ibv_send_wr *bad = nullptr;
    RDMA_CM_ASSERT(ibv_post_send, id_->qp, send_wr_, &bad);
  }

  /// Poll to see if anything new arrived on the completion queue.  This
  /// encapsulates so that id_ can be private.
  ///
  /// TODO: Can we remove this?
  int poll_cq(int num, ibv_wc *wc) {
    return ibv_poll_cq(id_->send_cq, num, wc);
  }
};
} // namespace remus::rdma::internal
