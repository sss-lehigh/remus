#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <netdb.h>
#include <ranges>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <span>

#include "segment.h"
#include "util.h"

namespace remus::internal {
 

/// @brief A connection object for RDMA communication.
/// @details
/// Connection encapsulates an RDMA communication identifier (rdma_cm_id*)
/// between two endpoints.  When a ComputeNode connects to a remote MemoryNode,
/// both machines will create a new Connection.  When a ComputeNode connects to
/// a local MemoryNode, we'll get a Connection object on the ComputeNode side,
/// but not on the MemoryNode side.  Once made, Connections are oblivious to the
/// side where they were made.
/// 
/// It is necessary to have a Connection between two machines before one-sided
/// operations can be issued between those machines.  A side effect is that the
/// connection allows two-sided operations between those machines.  This is
/// realized in our design by a Connection only having public methods to send,
/// receive, post one-sided requests, and poll for completions.
 
class Connection {
  rdma_cm_id *id_;         // Pointer to the QP for sends/receives
  const bool is_loopback_; // Track if this is a Loopback (self) connection

  /// Internal method for sending a Message (byte array) over RDMA as a
  /// two-sided operation.
  ///
  /// TODO: Move to rdma_ops.h?
  ///
  /// @tparam T TODO
  /// @param msg TODO
  /// @param seg TODO
  /// @param mr TODO
  /// @return TODO
  template <typename T>
    requires std::ranges::contiguous_range<T>
  remus::Status SendMessage(T &&msg, Segment &seg, ibv_mr *mr) {
    static_assert(std::ranges::sized_range<T>);

    auto msg_ptr = std::ranges::data(std::forward<T>(msg));
    auto msg_elements = std::ranges::size(std::forward<T>(msg)); // elements
    auto msg_size = msg_elements * sizeof(decltype(*msg_ptr));   // bytes

    // Copy the data into the send segment
    //
    // NB:  We assume that the receiver knows exactly how many bytes are going
    //      to be sent, and has a suitably large buffer prepared to receive
    //      them.  We also assume that seg is big enough to hold the message.
    //
    // NB:  We could ask the programmer to put the message into a Segment, which
    //      would avoid this copy, but we don't, because Send throughput isn't
    //      our priority.
    std::memcpy(seg.raw(), msg_ptr, msg_size);

    // Describe the command we're going to send to the RNIC
    ibv_sge sge; // NB: we don't zero it because we set all its fields below
    sge.addr = reinterpret_cast<uint64_t>(seg.raw());
    sge.length = msg_size;
    sge.lkey = mr->lkey;

    // TODO: Could we use rdma_post_send() instead of ibv_post_send()?
    //
    // TODO: Document this bit of code a bit better?
    ibv_send_wr wr;
    std::memset(&wr, 0, sizeof(wr));
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.num_sge = 1;
    wr.sg_list = &sge;
    wr.opcode = IBV_WR_SEND;
    wr.wr_id = 1; // NB: Unique IDs aren't needed for synchronous sends
    ibv_send_wr *bad_wr;
    if (ibv_post_send(id_->qp, &wr, &bad_wr) != 0) {
      // TODO: Should this be fatal?
      remus::Status err = {remus::InternalError, ""};
      err << "ibv_post_send(): " << strerror(errno);
      return err;
    }

    // NB: This assumes that the CQ associated with the SQ is synchronous.
    //
    // TODO: Document this
    ibv_wc wc;
    int comps = rdma_get_send_comp(id_, &wc);
    while (comps < 0 && errno == EAGAIN) {
      comps = rdma_get_send_comp(id_, &wc);
    }
    // TODO: Are these errors recoverable, or should we panic?
    if (comps < 0) {
      // TODO: Is operator << really better than just using std::format here?
      remus::Status e = {remus::InternalError, {}};
      return e << "rdma_get_send_comp: {}" << strerror(errno);
    } else if (wc.status != IBV_WC_SUCCESS) {
      remus::Status e = {remus::InternalError, {}};
      return e << "rdma_get_send_comp(): " << ibv_wc_status_str(wc.status);
    }
    return remus::Status::Ok();
  }

  /// Internal method for receiving a Message (byte array) over RDMA as a
  /// two-sided operation.
  ///
  /// TODO: Move to rdma_ops.h?
  ///
  /// @param seg TODO
  /// @return TODO
  remus::StatusVal<std::vector<uint8_t>> TryDeliverMessage(Segment &seg) {
    ibv_wc wc;
    auto ret = rdma_get_recv_comp(id_, &wc);
    if (ret < 0 && errno != EAGAIN) {
      remus::Status e = {remus::InternalError, {}};
      e << "rdma_get_recv_comp: " << strerror(errno);
      return {e, {}};
    } else if (ret < 0 && errno == EAGAIN) {
      return {{remus::Unavailable, "Retry"}, {}};
    } else {
      switch (wc.status) {
      case IBV_WC_WR_FLUSH_ERR:
        return {{remus::Aborted, "QP in error state"}, {}};
      case IBV_WC_SUCCESS: {
        // Prepare the response.
        std::span<uint8_t> recv_span{seg.raw(), wc.byte_len};
        remus::StatusVal<std::vector<uint8_t>> res = {
            remus::Status::Ok(), std::make_optional(std::vector<uint8_t>(
                                     recv_span.begin(), recv_span.end()))};
        return res;
      }
      default: {
        remus::Status err = {remus::InternalError, {}};
        err << "rdma_get_recv_comp(): " << ibv_wc_status_str(wc.status);
        return {err, {}};
      }
      }
    }
  }

  /// TODO
  ///
  /// TODO: Move to rdma_ops.h?
  ///
  /// @tparam T TODO
  /// @param seg TODO
  /// @return TODO
  template <typename T>
  remus::StatusVal<std::vector<T>> TryDeliverVec(Segment &seg) {
    remus::StatusVal<std::vector<uint8_t>> msg_or = TryDeliverMessage(seg);
    if (msg_or.status.t == remus::Ok) {
      std::vector<T> vec(msg_or.val.value().size() / sizeof(T));
      std::memcpy(vec.data(), msg_or.val.value().data(),
                  msg_or.val.value().size());
      return {remus::Status::Ok(), vec};
    }
    return {msg_or.status, {}};
  }

public:
  /// Construct a connection object
  ///
  /// @param src_id
  /// @param dst_id
  /// @param channel_id
  Connection(uint32_t src_id, uint32_t dst_id, rdma_cm_id *channel_id)
      : id_(channel_id), is_loopback_(src_id == dst_id) {}

  Connection(const Connection &) = delete;
  Connection(Connection &&c) = delete;

  /// TODO
  ///
  /// TODO: This should become SendVec, and then it should be simplified
  ///       accordingly?
  ///
  /// @tparam T  TODO
  /// @param msg TODO
  /// @param seg TODO
  /// @param mr TODO
  /// @return TODO
  template <typename T>
    requires std::ranges::contiguous_range<T>
  remus::Status Send(T &&msg, Segment &seg, ibv_mr *mr) {
    return SendMessage(std::forward<T>(msg), seg, mr);
  }

  /// TODO
  ///
  /// TODO: Rename to receive_vec?
  ///
  /// @tparam T TODO
  /// @param seg TODO
  /// @return TODO
  template <typename T>
  remus::StatusVal<std::vector<T>> DeliverVec(Segment &seg) {
    auto p = this->TryDeliverVec<T>(seg);
    while (p.status.t == remus::Unavailable) {
      p = this->TryDeliverVec<T>(seg);
    }
    return p;
  }

  /// TODO
  ~Connection() {
    // A loopback connection is made manually, so we do not need to deal with
    // the regular `rdma_cm` handling. Similarly, we avoid destroying the event
    // channel below since it is destroyed along with the id.
    if (!is_loopback_) {
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
    if (context != nullptr)
      free(context);
    else if (!is_loopback_)
      rdma_destroy_event_channel(channel);
  }

  /// Send a write request.  This encapsulates so that id_ can be private
  ///
  /// NB: These are issued by compute_thread, which means there's a trust
  ///     issue regarding the send_wr, and also regarding using the right PD.
  ///
  /// @param send_wr_ TODO
  void send_onesided(ibv_send_wr *send_wr) {
    ibv_send_wr *bad = nullptr;
    RDMA_CM_ASSERT(ibv_post_send, id_->qp, send_wr, &bad);
  }

  /// Poll to see if anything new arrived on the completion queue.  This
  /// encapsulates so that id_ can be private.
  ///
  /// @param num
  /// @param wc
  /// @return
  int poll_cq(int num, ibv_wc *wc) {
    return ibv_poll_cq(id_->send_cq, num, wc);
  }

  /// Return the protection domain associated with this Connection
  ibv_pd *pd() { return id_->pd; }
};
} // namespace remus::internal