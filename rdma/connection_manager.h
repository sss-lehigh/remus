#pragma once

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <infiniband/verbs.h>
#include <limits>
#include <memory>
#include <netdb.h>
#include <random>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <unordered_map>
#include <unordered_set>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"
#include "broker.h"
#include "memory_pool.h"

#define LOOPBACK_PORT_NUM 1

namespace rome::rdma::internal {

// Contains a copy of a received message by a `RdmaChannel` for the caller of
// `RdmaChannel::TryDeliver`.
//
// [mfs]  This could be put inside of TwoSidedRdmaMessenger, because it's only
//        used in this file?
struct Message {
  std::unique_ptr<uint8_t[]> buffer;
  size_t length;
};

///
///
/// NB: This was formerly called TwoSidedRdmaMessenger
class TwoSidedRdmaMessenger {
  // [mfs] TODO: Make these arguments to the ctor?
  static constexpr const uint32_t kCapacity = 1ul << 12;
  static constexpr const uint32_t kRecvMaxBytes = 1ul << 8;
  static_assert(kCapacity % 2 == 0, "Capacity must be divisible by two.");

  RdmaMemory rm_;           // Remotely accessible memory for send/recv buffers.
  rdma_cm_id *id_;          // (unowned) pointer to the QP for sends/rcvs
  ibv_mr *send_mr_;         // Memory region identified by `kSendId`
  const int send_cap_;      // Capacity (in bytes) of the send buffer
  uint8_t *send_base_;      // Base address of send buffer
  uint8_t *send_next_;      // Next un-posted address within send buffer
  uint32_t send_total_ = 0; // Number of sends that were performed
  ibv_mr *recv_mr_;         // Memory region identified by `kRecvId`
  const int recv_cap_;      // Capacity (in bytes) of recv buffer
  uint8_t *recv_base_;      // Base address of recv buffer
  uint8_t *recv_next_;      // Next unposted address within recv buffer
  uint32_t recv_total_ = 0; // Completed receives; helps track completion

public:
  explicit TwoSidedRdmaMessenger(rdma_cm_id *id)
      : rm_(kCapacity, std::nullopt, id->pd), id_(id), send_cap_(kCapacity / 2),
        recv_cap_(kCapacity / 2) {
    OK_OR_FAIL(rm_.RegisterMemoryRegion(kSendId, 0, send_cap_));
    OK_OR_FAIL(rm_.RegisterMemoryRegion(kRecvId, send_cap_, recv_cap_));
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

  sss::Status SendMessage(const Message &msg) {
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

  // Attempts to deliver a sent message by checking for completed receives and
  // then returning a `Message` containing a copy of the received buffer.
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

private:
  // Memory region IDs.
  static constexpr char kSendId[] = "send";
  static constexpr char kRecvId[] = "recv";

  // Reset the receive buffer and post `ibv_recv_wr` on the RQ. This should only
  // be called when all posted receives have corresponding completions,
  // otherwise there may be a race on memory by posted recvs.
  void PrepareRecvBuffer() {
    ROME_ASSERT(recv_total_ % (recv_cap_ / kRecvMaxBytes) == 0,
                "Unexpected number of completions from RQ");
    // Prepare the recv buffer for incoming messages with the assumption that
    // the maximum received message will be `max_recv_` bytes long.
    for (auto curr = recv_base_;
         curr <= recv_base_ + (recv_cap_ - kRecvMaxBytes);
         curr += kRecvMaxBytes) {
      RDMA_CM_ASSERT(rdma_post_recv, id_, nullptr, curr, kRecvMaxBytes,
                     recv_mr_);
    }
    recv_next_ = recv_base_;
  }
};

// Contains the necessary information for communicating between nodes. This
// class wraps a unique pointer to the `rdma_cm_id` that holds the QP used for
// communication, along with the `RdmaChannel` that represents the memory used
// for 2-sided message-passing.
class Connection {
  class RdmaChannel {

    TwoSidedRdmaMessenger messenger;

    // A pointer to the QP used to post sends and receives.
    rdma_cm_id *id_; //! NOT OWNED

  public:
    ~RdmaChannel() {}
    explicit RdmaChannel(rdma_cm_id *id) : messenger(id), id_(id) {}

    // No copy or move.
    RdmaChannel(const RdmaChannel &c) = delete;
    RdmaChannel(RdmaChannel &&c) = delete;

    // Getters.
    rdma_cm_id *id() const { return id_; }

    template <typename ProtoType> sss::Status Send(const ProtoType &proto) {
      Message msg{std::make_unique<uint8_t[]>(proto.ByteSizeLong()),
                  proto.ByteSizeLong()};
      proto.SerializeToArray(msg.buffer.get(), msg.length);
      return messenger.SendMessage(msg);
    }

  private:
    template <typename ProtoType> sss::StatusVal<ProtoType> TryDeliver() {
      auto msg_or = messenger.TryDeliverMessage();
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
    template <typename ProtoType> sss::StatusVal<ProtoType> Deliver() {
      auto p = this->TryDeliver<ProtoType>();
      while (p.status.t == sss::Unavailable) {
        p = this->TryDeliver<ProtoType>();
      }
      return p;
    }
  };

  // [mfs] These should probably be template parameters
  static constexpr size_t kMemoryPoolMessengerCapacity = 1 << 12;
  static constexpr size_t kMemoryPoolMessageSize = 1 << 8;

  uint32_t src_id_;
  uint32_t dst_id_;

  // Remotely accessible memory that is used for 2-sided message-passing.
  //
  // [mfs] What does "2-sided" mean here?
  //
  // [mfs]  Since we only ever instantiate this with one type from messenger.h,
  //        why not just hard-code it?
  RdmaChannel channel_;

public:
  Connection()
      : src_id_(std::numeric_limits<uint32_t>::max()),
        dst_id_(std::numeric_limits<uint32_t>::max()), channel_(nullptr) {}
  Connection(uint32_t src_id, uint32_t dst_id, rdma_cm_id *channel_id)
      : src_id_(src_id), dst_id_(dst_id), channel_(channel_id) {}

  Connection(const Connection &) = delete;
  Connection(Connection &&c) = delete;
  // : src_id_(c.src_id_), dst_id_(c.dst_id_),
  //   channel_(std::move(c.channel_)) {}

  uint32_t src_id() const { return src_id_; }
  uint32_t dst_id() const { return dst_id_; }
  rdma_cm_id *id() const { return channel_.id(); }
  RdmaChannel *channel() { return &channel_; }
};

/// [mfs] This should be a has-a RdmaReceiverInterface, since I was able to make
///       the inheritance private.
class ConnectionManager {
  // Whether or not to stop handling requests.
  //
  // [mfs] Why volatile instead of atomic?
  //
  // [mfs]  This seems like it should be part of the broker, because the only
  //        time its value is used is when the broker calls OnConnectRequest()
  volatile bool accepting_;

  // Current status
  sss::Status status_;

  uint32_t my_id_;
  std::unique_ptr<RdmaBroker<ConnectionManager>> broker_;

  // Maintains connection information for a given Internet address. A connection
  // manager only maintains a single connection per node. Nodes are identified
  // by a string representing their IP address.

  /// A mutex for protecting accesses to established_ and requested_
  /// TODO: use std::mutex instead?  Fix the locking bugs and races?
  std::atomic<int> mu_;

  std::unordered_map<uint32_t, std::unique_ptr<Connection>> requested_;
  std::unordered_map<uint32_t, std::unique_ptr<Connection>> established_;

  uint32_t backoff_us_{0};

  rdma_cm_id *loopback_id_ = nullptr;

public:
  ~ConnectionManager() {
    ROME_DEBUG("Shutting down: {}", fmt::ptr(this));
    Acquire(my_id_);
    // Shutdown()
    accepting_ = false;
    // end Shutdown()

    ROME_DEBUG("Stopping broker...");
    if (broker_ != nullptr)
      auto s = broker_->Stop();

    auto cleanup = [this](auto &iter) {
      // A loopback connection is made manually, so we do not need to deal with
      // the regular `rdma_cm` handling. Similarly, we avoid destroying the
      // event channel below since it is destroyed along with the id.
      auto id = iter.second->id();
      if (iter.first != my_id_) {
        rdma_disconnect(id);
        rdma_cm_event *event;
        auto result = rdma_get_cm_event(id->channel, &event);
        while (result == 0) {
          RDMA_CM_ASSERT(rdma_ack_cm_event, event);
          result = rdma_get_cm_event(id->channel, &event);
        }
      }

      // We only allocate contexts for connections that were created by the
      // `RdmaReceiver` callbacks. Otherwise, we created an event channel so
      // that we could asynchronously connect (and avoid distributed deadlock).
      auto *context = id->context;
      auto *channel = id->channel;
      rdma_destroy_ep(id);

      if (iter.first != my_id_ && context != nullptr) {
        free(context);
      } else if (iter.first != my_id_) {
        rdma_destroy_event_channel(channel);
      }
    };

    std::for_each(established_.begin(), established_.end(), cleanup);
    Release();
  }

  explicit ConnectionManager(uint32_t my_id)
      : accepting_(false), my_id_(my_id), broker_(nullptr), mu_(kUnlocked) {}

  sss::Status Start(std::string_view addr, std::optional<uint16_t> port) {
    if (accepting_) {
      return {sss::InternalError, "Cannot start broker twice"};
    }
    accepting_ = true;

    broker_ = RdmaBroker<ConnectionManager>::Create(addr, port, this);
    if (broker_ == nullptr) {
      return {sss::InternalError, "Failed to create broker"};
    }
    return {sss::Ok, {}};
  }

  // Getters.
  std::string address() const { return broker_->address(); }
  uint16_t port() const { return broker_->port(); }
  ibv_pd *pd() const { return broker_->pd(); }

  // `RdmaReceiverInterface` implementation
  void OnConnectRequest(rdma_cm_id *id, rdma_cm_event *event) {
    if (!accepting_)
      return;

    // The private data is used to understand from what node the connection
    // request is coming from.
    ROME_ASSERT_DEBUG(event->param.conn.private_data != nullptr,
                      "Received connect request without private data.");
    uint32_t peer_id =
        *reinterpret_cast<const uint32_t *>(event->param.conn.private_data);
    ROME_DEBUG("[OnConnectRequest] (Node {}) Got connection request from: {}",
               my_id_, peer_id);

    if (peer_id != my_id_) {
      // Attempt to acquire lock when not originating from same node
      if (!Acquire(peer_id)) {
        ROME_DEBUG("Lock acquisition failed: {}", mu_);
        rdma_reject(event->id, nullptr, 0);
        rdma_destroy_ep(id);
        rdma_ack_cm_event(event);
        return;
      }

      // Check if the connection has already been established.
      if (auto conn = established_.find(peer_id);
          conn != established_.end() || requested_.contains(peer_id)) {
        rdma_reject(event->id, nullptr, 0);
        rdma_destroy_ep(id);
        rdma_ack_cm_event(event);
        if (peer_id != my_id_)
          Release();
        std::string message = "[OnConnectRequest] (Node ";
        message = message + std::to_string(my_id_) + ") Connection already " +
                  (conn != established_.end() ? "established" : "requested") +
                  ": " + std::to_string(peer_id);
        ROME_DEBUG(message);
        return;
      }

      // Create a new QP for the connection.
      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      ROME_ASSERT(id->qp == nullptr, "QP already allocated...?");
      RDMA_CM_ASSERT(rdma_create_qp, id, pd(), &init_attr);
    } else {
      // rdma_destroy_id(id);
      id = loopback_id_;
    }

    // Prepare the necessary resources for this connection. Includes a
    // `RdmaChannel` that holds the QP and memory for 2-sided communication.
    // The underlying QP is RC, so we reuse it for issuing 1-sided RDMA too. We
    // also store the `peer_id` associated with this id so that we can reference
    // it later.
    auto context = new IdContext{peer_id, {}};
    // [mfs]  `memset()` shouldn't be needed for struct initialization in C++.
    //        Replace {} with {0} in the following line if necessary?
    std::memset(&context->conn_param, 0, sizeof(context->conn_param));
    context->conn_param.private_data = &context->node_id;
    context->conn_param.private_data_len = sizeof(context->node_id);
    context->conn_param.rnr_retry_count = 1; // Retry forever
    context->conn_param.retry_count = 7;
    context->conn_param.responder_resources = 8;
    context->conn_param.initiator_depth = 8;
    id->context = context;

    auto it =
        established_.emplace(peer_id, new Connection(my_id_, peer_id, id));
    ROME_ASSERT_DEBUG(it.second, "Insertion failed");

    ROME_DEBUG("[OnConnectRequest] (Node {}) peer={}, id={}", my_id_, peer_id,
               fmt::ptr(id));
    RDMA_CM_ASSERT(rdma_accept, id,
                   peer_id == my_id_ ? nullptr : &context->conn_param);
    rdma_ack_cm_event(event);
    if (peer_id != my_id_)
      Release();
  }

  // [mfs]  Is it necessary for the removal from erased() to precede destroying
  //        the endpoint?  If not, perhaps we could do all the rdma_* operations
  //        in the caller (broker), and then this code would only need to manage
  //        the established_ map?  If not, could the map operation come first,
  //        and then the caller calls rdma_disconnect (et al.)?
  void OnDisconnect(rdma_cm_id *id) {
    // This disconnection originated from the peer, so we simply disconnect the
    // local endpoint and clean it up.
    //
    // NOTE: The event is already ack'ed by the caller.
    rdma_disconnect(id);

    uint32_t peer_id = IdContext::GetNodeId(id->context);
    Acquire(peer_id);
    // [mfs] How could this ever be the case?
    if (auto conn = established_.find(peer_id);
        conn != established_.end() && conn->second->id() == id) {
      ROME_DEBUG("(Node {}) Disconnected from node {}", my_id_, peer_id);
      established_.erase(peer_id);
    }
    Release();
    auto *event_channel = id->channel;
    rdma_destroy_ep(id);
    rdma_destroy_event_channel(event_channel);
  }

  // `RdmaClientInterface` implementation
  sss::StatusVal<Connection *> Connect(uint32_t peer_id,
                                       std::string_view server, uint16_t port) {
    if (Acquire(my_id_)) {
      auto conn = established_.find(peer_id);
      if (conn != established_.end()) {
        Release();
        return {sss::Status::Ok(), conn->second.get()};
      }

      auto port_str = std::to_string(htons(port));
      rdma_cm_id *id = nullptr;
      rdma_addrinfo hints, *resolved = nullptr;

      std::memset(&hints, 0, sizeof(hints));
      hints.ai_port_space = RDMA_PS_TCP;
      hints.ai_qp_type = IBV_QPT_RC;
      hints.ai_family = AF_IB;

      struct sockaddr_in src;
      std::memset(&src, 0, sizeof(src));
      src.sin_family = AF_INET;
      auto src_addr_str = broker_->address();
      inet_aton(src_addr_str.data(), &src.sin_addr);

      hints.ai_src_addr = reinterpret_cast<sockaddr *>(&src);
      hints.ai_src_len = sizeof(src);

      // Resolve the server's address. If this connection request is for the
      // loopback connection, then we are going to
      int gai_ret =
          rdma_getaddrinfo(server.data(), port_str.data(), &hints, &resolved);
      if (gai_ret != 0) {
        sss::Status err = {sss::InternalError, "rdma_getaddrinfo(): "};
        err << gai_strerror(gai_ret);
        return {err, {}};
      }

      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      auto err = rdma_create_ep(&id, resolved, pd(), &init_attr);
      rdma_freeaddrinfo(resolved);
      if (err) {
        Release();
        sss::Status ee = {sss::InternalError, "rdma_create_ep(): "};
        ee << strerror(errno) << " (" << errno << ")";
        return {ee, {}};
      }
      ROME_DEBUG("[Connect] (Node {}) Trying to connect to: {} (id={})", my_id_,
                 peer_id, fmt::ptr(id));

      if (peer_id == my_id_)
        // [mfs]  I don't like having ConnectLoopback do the release...  Is the
        //        lock granularity a problem?
        return ConnectLoopback(id);

      auto *event_channel = rdma_create_event_channel();
      RDMA_CM_CHECK_TOVAL(fcntl, event_channel->fd, F_SETFL,
                          fcntl(event_channel->fd, F_GETFL) | O_NONBLOCK);
      RDMA_CM_CHECK_TOVAL(rdma_migrate_id, id, event_channel);

      rdma_conn_param conn_param;
      std::memset(&conn_param, 0, sizeof(conn_param));
      conn_param.private_data = &my_id_;
      conn_param.private_data_len = sizeof(my_id_);
      conn_param.retry_count = 7;
      conn_param.rnr_retry_count = 1;
      conn_param.responder_resources = 8;
      conn_param.initiator_depth = 8;

      RDMA_CM_CHECK_TOVAL(rdma_connect, id, &conn_param);

      // Handle events.
      while (true) {
        rdma_cm_event *event;
        auto result = rdma_get_cm_event(id->channel, &event);
        while (result < 0 && errno == EAGAIN) {
          result = rdma_get_cm_event(id->channel, &event);
        }
        ROME_DEBUG("[Connect] (Node {}) Got event: {} (id={})", my_id_,
                   rdma_event_str(event->event), fmt::ptr(id));

        switch (event->event) {
        case RDMA_CM_EVENT_ESTABLISHED: {
          RDMA_CM_CHECK_TOVAL(rdma_ack_cm_event, event);
          auto conn = established_.find(peer_id);
          if (bool is_established = (conn != established_.end());
              is_established && peer_id != my_id_) {
            Release();

            // Since we are initiating the disconnection, we must get and ack
            // the event.
            ROME_DEBUG("[Connect] (Node {}) Disconnecting: (id={})", my_id_,
                       fmt::ptr(id));
            RDMA_CM_CHECK_TOVAL(rdma_disconnect, id);
            rdma_cm_event *event;
            auto result = rdma_get_cm_event(id->channel, &event);
            while (result < 0 && errno == EAGAIN) {
              result = rdma_get_cm_event(id->channel, &event);
            }
            RDMA_CM_CHECK_TOVAL(rdma_ack_cm_event, event);

            rdma_destroy_ep(id);
            rdma_destroy_event_channel(event_channel);

            if (is_established) {
              ROME_DEBUG("[Connect] Already connected: {}", peer_id);
              return {sss::Status::Ok(), conn->second.get()};
            } else {
              sss::Status err = {sss::Unavailable, "[Connect (Node "};
              err << my_id_ << ") Connection is already requested: " << peer_id;
              return {err, {}};
            }
          }

          // If this code block is reached, then the connection established by
          // this call is the first successful connection to be established and
          // therefore we must add it to the set of established connections.
          ROME_DEBUG(
              "Connected: dev={}, addr={}, port={}", id->verbs->device->name,
              inet_ntoa(reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(id))
                            ->sin_addr),
              rdma_get_src_port(id));

          RDMA_CM_CHECK_TOVAL(fcntl, event_channel->fd, F_SETFL,
                              fcntl(event_channel->fd, F_GETFL) | O_SYNC);
          RDMA_CM_CHECK_TOVAL(fcntl, id->recv_cq->channel->fd, F_SETFL,
                              fcntl(id->recv_cq->channel->fd, F_GETFL) |
                                  O_NONBLOCK);
          RDMA_CM_CHECK_TOVAL(fcntl, id->send_cq->channel->fd, F_SETFL,
                              fcntl(id->send_cq->channel->fd, F_GETFL) |
                                  O_NONBLOCK);

          // Allocate a new control channel to be used with this connection
          auto iter = established_.emplace(peer_id,
                                           new Connection(my_id_, peer_id, id));
          ROME_ASSERT(iter.second, "Unexepected error");
          auto *new_conn = established_[peer_id].get();
          Release();
          return {sss::Status::Ok(), new_conn};
        }
        case RDMA_CM_EVENT_ADDR_RESOLVED:
          ROME_WARN("Got addr resolved...");
          RDMA_CM_CHECK_TOVAL(rdma_ack_cm_event, event);
          break;
        default: {
          auto cm_event = event->event;
          RDMA_CM_CHECK_TOVAL(rdma_ack_cm_event, event);
          backoff_us_ =
              backoff_us_ > 0
                  ? std::min((backoff_us_ + (100 * my_id_)) * 2, kMaxBackoffUs)
                  : kMinBackoffUs;
          Release();
          rdma_destroy_ep(id);
          rdma_destroy_event_channel(event_channel);
          if (cm_event == RDMA_CM_EVENT_REJECTED) {
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us_));
            return {{sss::Unavailable, "Connection request rejected"}, {}};
          }
          sss::Status err = {sss::InternalError, "Got unexpected event: "};
          err << rdma_event_str(cm_event);
          return {err, {}};
        }
        }
      }
    } else {
      return {{sss::Unavailable, "Lock acquisition failed"}, {}};
    }
  }

  sss::StatusVal<Connection *> GetConnection(uint32_t peer_id) {
    while (!Acquire(my_id_)) {
      std::this_thread::yield();
    }
    auto conn = established_.find(peer_id);
    if (conn != established_.end()) {
      auto result = conn->second.get();
      Release();
      return {sss::Status::Ok(), result};
    } else {
      Release();
      sss::Status err = {sss::NotFound, "Connection not found: "};
      err << peer_id;
      return {err, {}};
    }
  }

private:
  // The size of each memory region dedicated to a single connection.
  static constexpr int kCapacity = 1 << 12; // 4 KiB
  static constexpr int kMaxRecvBytes = 64;

  static constexpr int kMaxWr = kCapacity / kMaxRecvBytes;
  static constexpr int kMaxSge = 1;
  static constexpr int kMaxInlineData = 0;

  static constexpr char kPdId[] = "ConnectionManager";

  static constexpr int kUnlocked = -1;

  static constexpr uint32_t kMinBackoffUs = 100;
  static constexpr uint32_t kMaxBackoffUs = 5000000;

  // Each `rdma_cm_id` can be associated with some context, which is represented
  // by `IdContext`. `node_id` is the numerical identifier for the peer node of
  // the connection and `conn_param` is used to provide private data during the
  // connection set up to send the local node identifier upon connection setup.
  struct IdContext {
    uint32_t node_id;
    rdma_conn_param conn_param;

    static inline uint32_t GetNodeId(void *ctx) {
      return reinterpret_cast<IdContext *>(ctx)->node_id;
    }
  };

  // Lock acquisition will spin until either the lock is acquired successfully
  // or the locker is an outgoing connection request from this node.
  inline bool Acquire(int peer_id) {
    for (int expected = kUnlocked;
         !mu_.compare_exchange_weak(expected, peer_id); expected = kUnlocked) {
      if (expected == my_id_) {
        ROME_DEBUG("[Acquire] (Node {}) Giving up lock acquisition: actual={}, "
                   "swap={}",
                   my_id_, expected, peer_id);
        return false;
      }
    }
    return true;
  }

  inline void Release() { mu_ = kUnlocked; }

  static ibv_qp_init_attr DefaultQpInitAttr() {
    ibv_qp_init_attr init_attr;
    std::memset(&init_attr, 0, sizeof(init_attr));
    init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
    init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = kMaxSge;
    init_attr.cap.max_inline_data = kMaxInlineData;
    init_attr.sq_sig_all = 0; // Must request completions.
    init_attr.qp_type = IBV_QPT_RC;
    return init_attr;
  }

  static ibv_qp_attr DefaultQpAttr() {
    ibv_qp_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    attr.max_dest_rd_atomic = 8;
    attr.path_mtu = IBV_MTU_4096;
    attr.min_rnr_timer = 12;
    attr.rq_psn = 0;
    attr.sq_psn = 0;
    attr.timeout = 12;
    attr.retry_cnt = 7;
    attr.rnr_retry = 1;
    attr.max_rd_atomic = 8;
    return attr;
  }

  sss::StatusVal<Connection *> ConnectLoopback(rdma_cm_id *id) {
    ROME_ASSERT_DEBUG(id->qp != nullptr, "No QP associated with endpoint");
    ROME_DEBUG("Connecting loopback...");
    ibv_qp_attr attr;
    int attr_mask;

    attr = DefaultQpAttr();
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = LOOPBACK_PORT_NUM; // id->port_num;
    attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    ROME_TRACE("Loopback: IBV_QPS_INIT");
    if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
      // [mfs] Error: returns without releasing locks
      sss::Status err = {sss::InternalError, ""};
      err << "ibv_modify_qp(): " << strerror(errno);
      return {err, {}};
    }

    ibv_port_attr port_attr;
    RDMA_CM_CHECK_TOVAL(ibv_query_port, id->verbs, LOOPBACK_PORT_NUM,
                        &port_attr); // RDMA_CM_CHECK(ibv_query_port, id->verbs,
                                     // id->port_num, &port_attr);
    attr.ah_attr.dlid = port_attr.lid;
    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = id->qp->qp_num;
    attr.ah_attr.port_num = LOOPBACK_PORT_NUM; // id->port_num;
    attr_mask =
        (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
         IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    ROME_TRACE("Loopback: IBV_QPS_RTR");
    if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
      // [mfs] Error: returns without releasing locks
      sss::Status err = {sss::InternalError, ""};
      err << "ibv_modify_qp(): " << strerror(errno);
      return {err, {}};
    }

    attr.qp_state = IBV_QPS_RTS;
    attr_mask = (IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                 IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
    ROME_TRACE("Loopback: IBV_QPS_RTS");
    RDMA_CM_CHECK_TOVAL(ibv_modify_qp, id->qp, &attr, attr_mask);

    RDMA_CM_CHECK_TOVAL(fcntl, id->recv_cq->channel->fd, F_SETFL,
                        fcntl(id->recv_cq->channel->fd, F_GETFL) | O_NONBLOCK);
    RDMA_CM_CHECK_TOVAL(fcntl, id->send_cq->channel->fd, F_SETFL,
                        fcntl(id->send_cq->channel->fd, F_GETFL) | O_NONBLOCK);

    // Allocate a new control channel to be used with this connection
    auto it = established_.emplace(my_id_, new Connection(my_id_, my_id_, id));
    ROME_ASSERT(it.second, "Unexepected error");
    Release();
    // TODO: isn't it racy to access established_ after releasing the lock?
    return {{sss::Status::Ok()}, established_[my_id_].get()};
  }
};
} // namespace rome::rdma