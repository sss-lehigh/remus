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
#include <unordered_map>
#include <vector>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"
#include "connection.h"
#include "memory_pool.h"

namespace rome::rdma::internal {

constexpr uint32_t LOOPBACK_PORT_NUM = 1;

class ConnectionManager {

public:
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

private:
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

  // Maintains connection information for a given Internet address. A connection
  // manager only maintains a single connection per node. Nodes are identified
  // by a string representing their IP address.

  /// A mutex for protecting accesses to established_
  ///
  /// TODO: use std::mutex instead?  Fix the locking bugs and races?
  std::atomic<int> mu_;
  /// established_ is a map holding all of the connections between threads.
  ///
  /// TODO: The structure of the code right now is concerning, because we have a
  ///       map from id to connection, meaning that this map can only hold one
  ///       qp between this machine and another.  I think what we want is for
  ///       the map to be from an id to a vector of connections?  But I also
  ///       think that most of the functionality in this file needs to migrate
  ///       to the broker?
  std::unordered_map<uint32_t, std::unique_ptr<Connection>> established_;

  rdma_cm_id *loopback_id_ = nullptr;

public:
  ~ConnectionManager() {
    ROME_TRACE("Shutting down: {}", fmt::ptr(this));
    Acquire(my_id_);
    // Shutdown()
    accepting_ = false;
    // end Shutdown()

    for (auto &iter : established_) {
      iter.second->cleanup(iter.first, my_id_);
    }

    Release();
  }

  explicit ConnectionManager(uint32_t my_id)
      : accepting_(false), my_id_(my_id), mu_(kUnlocked) {}

  /// TODO: I don't think we still need this.  I also think accepting_ can go
  ///       away?
  sss::Status Start(std::string_view addr, std::optional<uint16_t> port) {
    // TODO: This method is fail-stop in the caller, so instead it should be
    // fail-stop here
    if (accepting_) {
      return {sss::InternalError, "Cannot start broker twice"};
    }
    accepting_ = true;
    return {sss::Ok, {}};
  }

  // TODO: Migrate to listener.h?
  void OnConnectRequest(rdma_cm_id *id, rdma_cm_event *event, ibv_pd *pd) {
    if (!accepting_)
      return;

    // The private data is used to understand from what node the connection
    // request is coming from.
    ROME_ASSERT_DEBUG(event->param.conn.private_data != nullptr,
                      "Received connect request without private data.");
    uint32_t peer_id =
        *reinterpret_cast<const uint32_t *>(event->param.conn.private_data);
    ROME_TRACE("[OnConnectRequest] (Node {}) Got connection request from: {}",
               my_id_, peer_id);

    if (peer_id != my_id_) {
      // Attempt to acquire lock when not originating from same node
      if (!Acquire(peer_id)) {
        ROME_TRACE("Lock acquisition failed: {}", mu_);
        rdma_reject(event->id, nullptr, 0);
        rdma_destroy_ep(id);
        rdma_ack_cm_event(event);
        return;
      }

      // Check if the connection has already been established.
      if (auto conn = established_.find(peer_id); conn != established_.end()) {
        rdma_reject(event->id, nullptr, 0);
        rdma_destroy_ep(id);
        rdma_ack_cm_event(event);
        if (peer_id != my_id_)
          Release();
        std::string message = "[OnConnectRequest] (Node ";
        message = message + std::to_string(my_id_) + ") Connection already " +
                  (conn != established_.end() ? "established" : "requested") +
                  ": " + std::to_string(peer_id);
        ROME_TRACE(message);
        return;
      }

      // Create a new QP for the connection.
      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      ROME_ASSERT(id->qp == nullptr, "QP already allocated...?");
      RDMA_CM_ASSERT(rdma_create_qp, id, pd, &init_attr);
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

    ROME_TRACE("[OnConnectRequest] (Node {}) peer={}, id={}", my_id_, peer_id,
               fmt::ptr(id));
    RDMA_CM_ASSERT(rdma_accept, id,
                   peer_id == my_id_ ? nullptr : &context->conn_param);
    rdma_ack_cm_event(event);
    if (peer_id != my_id_)
      Release();
  }

  // TODO: Migrate to listener.h?
  //
  // [mfs]  Is it necessary for the removal from erased() to precede destroying
  //        the endpoint?  If not, perhaps we could do all the rdma_* operations
  //        in the caller (broker), and then this code would only need to manage
  //        the established_ map?  If not, could the map operation come first,
  //        and then the caller calls rdma_disconnect (et al.)?
  //
  // TODO: why does this deal with an rdma_cm_id instead of a Connection?
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
        conn != established_.end() && conn->second->matches(id)) {
      ROME_TRACE("(Node {}) Disconnected from node {}", my_id_, peer_id);
      established_.erase(peer_id);
    }
    Release();
    auto *event_channel = id->channel;
    rdma_destroy_ep(id);
    rdma_destroy_event_channel(event_channel);
  }

  // [mfs]  I think this should not go to broker, nor should ConnectLoopback,
  //        but everything else should, because everything else is for receiving
  //        a connection, not creating one.
  sss::StatusVal<Connection *> Connect(uint32_t peer_id,
                                       std::string_view server, uint16_t port,
                                       ibv_pd *pd, std::string my_address) {
    // It's OK for this to return OK or Unavailable.  Otherwise, it is going to
    // lead to the whole program crashing, so why not just crash here?
    if (Acquire(my_id_)) {
      auto conn = established_.find(peer_id);
      if (conn != established_.end()) {
        Release();
        return {sss::Status::Ok(), conn->second.get()};
      }

      auto port_str = std::to_string(htons(port));
      rdma_cm_id *id = nullptr;
      rdma_addrinfo hints = {0}, *resolved = nullptr;

      hints.ai_port_space = RDMA_PS_TCP;
      hints.ai_qp_type = IBV_QPT_RC;
      hints.ai_family = AF_IB;

      struct sockaddr_in src = {0};
      src.sin_family = AF_INET;
      auto src_addr_str = my_address;
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
      auto err = rdma_create_ep(&id, resolved, pd, &init_attr);
      rdma_freeaddrinfo(resolved);
      if (err) {
        Release();
        sss::Status ee = {sss::InternalError, "rdma_create_ep(): "};
        ee << strerror(errno) << " (" << errno << ")";
        return {ee, {}};
      }
      ROME_TRACE("[Connect] (Node {}) Trying to connect to: {} (id={})", my_id_,
                 peer_id, fmt::ptr(id));

      if (peer_id == my_id_)
        // [mfs]  I don't like having ConnectLoopback do the release...  Is the
        //        lock granularity a problem?
        return ConnectLoopback(id);

      auto *event_channel = rdma_create_event_channel();
      RDMA_CM_CHECK_TOVAL(fcntl, event_channel->fd, F_SETFL,
                          fcntl(event_channel->fd, F_GETFL) | O_NONBLOCK);
      RDMA_CM_CHECK_TOVAL(rdma_migrate_id, id, event_channel);

      rdma_conn_param conn_param = {0};
      conn_param.private_data = &my_id_;
      conn_param.private_data_len = sizeof(my_id_);
      conn_param.retry_count = 7;
      conn_param.rnr_retry_count = 1;
      conn_param.responder_resources = 8;
      conn_param.initiator_depth = 8;

      RDMA_CM_CHECK_TOVAL(rdma_connect, id, &conn_param);

      // Handle events.
      uint32_t backoff_us_{0}; // for backoff
      while (true) {
        rdma_cm_event *event;
        auto result = rdma_get_cm_event(id->channel, &event);
        while (result < 0 && errno == EAGAIN) {
          result = rdma_get_cm_event(id->channel, &event);
        }
        ROME_TRACE("[Connect] (Node {}) Got event: {} (id={})", my_id_,
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
            ROME_TRACE("[Connect] (Node {}) Disconnecting: (id={})", my_id_,
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
              ROME_TRACE("[Connect] Already connected: {}", peer_id);
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
          ROME_TRACE(
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

  // [mfs]  This should stay, because we need to keep connections after the
  //        broker goes away.
  //
  // TODO: Are errors always fatal here?  Yes.  So we can go ahead and make this
  // fail-stop.
  sss::StatusVal<Connection *> GetConnection(uint32_t peer_id) {
    // TODO: use a regular mutex
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

  static constexpr int kUnlocked = -1;

  static constexpr uint32_t kMinBackoffUs = 100;
  static constexpr uint32_t kMaxBackoffUs = 5000000;

  // Lock acquisition will spin until either the lock is acquired successfully
  // or the locker is an outgoing connection request from this node.
  inline bool Acquire(int peer_id) {
    for (int expected = kUnlocked;
         !mu_.compare_exchange_weak(expected, peer_id); expected = kUnlocked) {
      if (expected == my_id_) {
        ROME_TRACE("[Acquire] (Node {}) Giving up lock acquisition: actual={}, "
                   "swap={}",
                   my_id_, expected, peer_id);
        return false;
      }
    }
    return true;
  }

  inline void Release() { mu_ = kUnlocked; }

  // TODO:  This is used by Connect and also OnConnectRequest.  Need to migrate
  //        to util?
  static ibv_qp_init_attr DefaultQpInitAttr() {
    ibv_qp_init_attr init_attr = {0};
    init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
    init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = kMaxSge;
    init_attr.cap.max_inline_data = kMaxInlineData;
    init_attr.sq_sig_all = 0; // Must request completions.
    init_attr.qp_type = IBV_QPT_RC;
    return init_attr;
  }

  // TODO: This is only used in ConnectLoopback
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

  // TODO: This belongs wherever Connect() goes
  sss::StatusVal<Connection *> ConnectLoopback(rdma_cm_id *id) {
    ROME_ASSERT_DEBUG(id->qp != nullptr, "No QP associated with endpoint");
    ROME_TRACE("Connecting loopback...");
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
} // namespace rome::rdma::internal