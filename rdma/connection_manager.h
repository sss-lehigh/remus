
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
#include <unordered_map>
#include <unordered_set>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"
#include "broker.h"
#include "channel.h"
#include "messenger.h"
#include "receiver.h"

namespace rome::rdma {

// Contains the necessary information for communicating between nodes. This
// class wraps a unique pointer to the `rdma_cm_id` that holds the QP used for
// communication, along with the `RdmaChannel` that represents the memory used
// for 2-sided message-passing.
template <typename Channel = RdmaChannel<EmptyRdmaMessenger>> class Connection {
public:
  typedef Channel channel_type;

  Connection()
      : terminated_(false), src_id_(std::numeric_limits<uint32_t>::max()),
        dst_id_(std::numeric_limits<uint32_t>::max()), channel_(nullptr) {}
  Connection(uint32_t src_id, uint32_t dst_id,
             std::unique_ptr<channel_type> channel)
      : terminated_(false), src_id_(src_id), dst_id_(dst_id),
        channel_(std::move(channel)) {}

  Connection(const Connection &) = delete;
  Connection(Connection &&c)
      : terminated_(c.terminated_), src_id_(c.src_id_), dst_id_(c.dst_id_),
        channel_(std::move(c.channel_)) {}

  // Getters.
  inline bool terminated() const { return terminated_; }
  uint32_t src_id() const { return src_id_; }
  uint32_t dst_id() const { return dst_id_; }
  rdma_cm_id *id() const { return channel_->id(); }
  channel_type *channel() const { return channel_.get(); }

  void Terminate() { terminated_ = true; }

private:
  volatile bool terminated_;

  uint32_t src_id_;
  uint32_t dst_id_;

  // Remotely accessible memory that is used for 2-sided message-passing.
  std::unique_ptr<channel_type> channel_;
};

#define LOOPBACK_PORT_NUM 1

template <typename ChannelType>
class ConnectionManager : public RdmaReceiverInterface {
public:
  typedef Connection<ChannelType> conn_type;

  ~ConnectionManager();
  explicit ConnectionManager(uint32_t my_id);

  sss::Status Start(std::string_view addr, std::optional<uint16_t> port);

  // Getters.
  std::string address() const { return broker_->address(); }
  uint16_t port() const { return broker_->port(); }
  ibv_pd *pd() const { return broker_->pd(); }

  int GetNumConnections() {
    Acquire(my_id_);
    int size = established_.size();
    Release();
    return size;
  }

  // `RdmaReceiverInterface` implementaiton
  void OnConnectRequest(rdma_cm_id *id, rdma_cm_event *event) override;
  void OnEstablished(rdma_cm_id *id, rdma_cm_event *event) override;
  void OnDisconnect(rdma_cm_id *id) override;

  // `RdmaClientInterface` implementation
  sss::StatusVal<conn_type *> Connect(uint32_t node_id, std::string_view server,
                                      uint16_t port);

  sss::StatusVal<conn_type *> GetConnection(uint32_t node_id);

  void Shutdown();

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
    ChannelType *channel;

    static inline uint32_t GetNodeId(void *ctx) {
      return reinterpret_cast<IdContext *>(ctx)->node_id;
    }

    static inline ChannelType *GetRdmaChannel(void *ctx) {
      return reinterpret_cast<IdContext *>(ctx)->channel;
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

  constexpr ibv_qp_init_attr DefaultQpInitAttr() {
    ibv_qp_init_attr init_attr;
    std::memset(&init_attr, 0, sizeof(init_attr));
    init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
    init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = kMaxSge;
    init_attr.cap.max_inline_data = kMaxInlineData;
    init_attr.sq_sig_all = 0; // Must request completions.
    init_attr.qp_type = IBV_QPT_RC;
    return init_attr;
  }

  constexpr ibv_qp_attr DefaultQpAttr() {
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

  sss::StatusVal<conn_type *> ConnectLoopback(rdma_cm_id *id);

  // Whether or not to stop handling requests.
  volatile bool accepting_;

  // Current status
  sss::Status status_;

  uint32_t my_id_;
  std::unique_ptr<RdmaBroker> broker_;
  ibv_pd *pd_; // Convenience ptr to protection domain of `broker_`

  // Maintains connection information for a given Internet address. A connection
  // manager only maintains a single connection per node. Nodes are identified
  // by a string representing their IP address.
  std::atomic<int> mu_;
  std::unordered_map<uint32_t, std::unique_ptr<conn_type>> requested_;
  std::unordered_map<uint32_t, std::unique_ptr<conn_type>> established_;

  uint32_t backoff_us_{0};

  rdma_cm_id *loopback_id_ = nullptr;
};

template <typename ChannelType>
ConnectionManager<ChannelType>::~ConnectionManager() {
  ROME_DEBUG("Shutting down: {}", fmt::ptr(this));
  Acquire(my_id_);
  Shutdown();

  ROME_DEBUG("Stopping broker...");
  if (broker_ != nullptr)
    auto s = broker_->Stop();

  auto cleanup = [this](auto &iter) {
    // A loopback connection is made manually, so we do not need to deal with
    // the regular `rdma_cm` handling. Similarly, we avoid destroying the event
    // channel below since it is destroyed along with the id.
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

template <typename ChannelType>
void ConnectionManager<ChannelType>::Shutdown() {
  // Stop accepting new requests.
  accepting_ = false;
}

template <typename ChannelType>
ConnectionManager<ChannelType>::ConnectionManager(uint32_t my_id)
    : accepting_(false), my_id_(my_id), broker_(nullptr), mu_(-1) {}

template <typename ChannelType>
sss::Status
ConnectionManager<ChannelType>::Start(std::string_view addr,
                                      std::optional<uint16_t> port) {
  if (accepting_) {
    return {sss::InternalError, "Cannot start broker twice"};
  }
  accepting_ = true;

  broker_ = RdmaBroker::Create(addr, port, this);
  if (broker_ == nullptr) {
    return {sss::InternalError, "Failed to create broker"};
  }
  return {sss::Ok, {}};
}

namespace {

[[maybe_unused]] inline std::string GetDestinationAsString(rdma_cm_id *id) {
  char addr_str[INET_ADDRSTRLEN];
  ROME_ASSERT(inet_ntop(AF_INET, &(id->route.addr.dst_sin.sin_addr), addr_str,
                        INET_ADDRSTRLEN) != nullptr,
              "inet_ntop(): {}", strerror(errno));
  std::stringstream ss;
  ss << addr_str << ":" << rdma_get_dst_port(id);
  return ss.str();
}

} // namespace

template <typename ChannelType>
void ConnectionManager<ChannelType>::OnConnectRequest(rdma_cm_id *id,
                                                      rdma_cm_event *event) {
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
  auto context = new IdContext{peer_id, {}, {}};
  std::memset(&context->conn_param, 0, sizeof(context->conn_param));
  context->conn_param.private_data = &context->node_id;
  context->conn_param.private_data_len = sizeof(context->node_id);
  context->conn_param.rnr_retry_count = 1; // Retry forever
  context->conn_param.retry_count = 7;
  context->conn_param.responder_resources = 8;
  context->conn_param.initiator_depth = 8;
  id->context = context;

  auto iter = established_.emplace(
      peer_id,
      new Connection{my_id_, peer_id, std::make_unique<ChannelType>(id)});
  ROME_ASSERT_DEBUG(iter.second, "Insertion failed");

  ROME_DEBUG("[OnConnectRequest] (Node {}) peer={}, id={}", my_id_, peer_id,
             fmt::ptr(id));
  RDMA_CM_ASSERT(rdma_accept, id,
                 peer_id == my_id_ ? nullptr : &context->conn_param);
  rdma_ack_cm_event(event);
  if (peer_id != my_id_)
    Release();
}

template <typename ChannelType>
void ConnectionManager<ChannelType>::OnEstablished(rdma_cm_id *id,
                                                   rdma_cm_event *event) {
  rdma_ack_cm_event(event);
}

template <typename ChannelType>
void ConnectionManager<ChannelType>::OnDisconnect(rdma_cm_id *id) {
  // This disconnection originated from the peer, so we simply disconnect the
  // local endpoint and clean it up.
  //
  // NOTE: The event is already ack'ed by the caller.
  rdma_disconnect(id);

  uint32_t peer_id = IdContext::GetNodeId(id->context);
  Acquire(peer_id);
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

template <typename ChannelType>
sss::StatusVal<typename ConnectionManager<ChannelType>::conn_type *>
ConnectionManager<ChannelType>::ConnectLoopback(rdma_cm_id *id) {
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
  auto channel = std::make_unique<ChannelType>(id);
  auto iter = established_.emplace(
      my_id_, new Connection{my_id_, my_id_, std::move(channel)});
  ROME_ASSERT(iter.second, "Unexepected error");
  Release();
  return {{sss::Status::Ok()}, established_[my_id_].get()};
}

template <typename ChannelType>
sss::StatusVal<typename ConnectionManager<ChannelType>::conn_type *>
ConnectionManager<ChannelType>::Connect(uint32_t peer_id,
                                        std::string_view server,
                                        uint16_t port) {
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

      sss::StatusVal<ChannelType *> conn_or;
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
        auto channel = std::make_unique<ChannelType>(id);
        auto iter = established_.emplace(
            peer_id, new Connection{my_id_, peer_id, std::move(channel)});
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

template <typename ChannelType>
sss::StatusVal<typename ConnectionManager<ChannelType>::conn_type *>
ConnectionManager<ChannelType>::GetConnection(uint32_t peer_id) {
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

} // namespace rome::rdma