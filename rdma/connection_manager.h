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

#define LOOPBACK_PORT_NUM 1

namespace rome::rdma::internal {

class ConnectionManager {

  /// A broker handles the connection setup using the RDMA CM library. It is
  /// single threaded but communicates with all other brokers in the system to
  /// exchange information regarding the underlying RDMA memory configurations.
  class RdmaBroker {
    /// Produce a vector of active ports, or None if none are found
    static std::optional<std::vector<int>>
    FindActivePorts(ibv_context *context) {
      // Find the first active port, failing if none exists.
      ibv_device_attr dev_attr;
      ibv_query_device(context, &dev_attr);
      std::vector<int> ports;
      for (int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
        ibv_port_attr port_attr;
        ibv_query_port(context, i, &port_attr);
        if (port_attr.state != IBV_PORT_ACTIVE) {
          continue;
        } else {
          ports.push_back(i);
        }
      }
      if (ports.empty())
        return {};
      return ports;
    }

    // Returns a vector of device name and active port pairs that are accessible
    // on this machine, or None if no devices are found
    //
    // TODO: this function name is misleading... It is stateful, since it
    // *opens* devices.  This means that its return value doesn't tell the whole
    // story.
    static std::optional<std::vector<std::pair<std::string, int>>>
    GetAvailableDevices() {
      int num_devices;
      auto **device_list = ibv_get_device_list(&num_devices);
      if (num_devices <= 0)
        return {};
      std::vector<std::pair<std::string, int>> active;
      for (int i = 0; i < num_devices; ++i) {
        auto *context = ibv_open_device(device_list[i]);
        if (context) {
          auto ports = FindActivePorts(context);
          if (!ports.has_value())
            continue;
          for (auto p : ports.value()) {
            active.emplace_back(context->device->name, p);
          }
        }
      }

      ibv_free_device_list(device_list);
      return active;
    }

    /// The worker thread that listens and responds to incoming messages.
    struct thread_deleter {
      void operator()(std::thread *thread) {
        thread->join();
        free(thread);
      }
    };

    std::string address_;
    uint16_t port_;

    // Flag to indicate that the worker thread should terminate.
    std::atomic<bool> terminate_;
    std::unique_ptr<std::thread, thread_deleter> runner_;

    // Status of the broker at any given time.
    //
    // [mfs] I don't think we need this
    sss::Status status_;

    // RDMA CM related members.
    rdma_event_channel *listen_channel_;
    rdma_cm_id *listen_id_;

    // The total number of connections made by this broker.
    // [mfs] Why atomic?
    std::atomic<uint32_t> num_connections_;

    // Maintains connections that are forwarded by the broker.
    ConnectionManager *receiver_; //! NOT OWNED

  public:
    ~RdmaBroker() {
      [[maybe_unused]] auto s = Stop();
      rdma_destroy_ep(listen_id_);
    }

    // Creates a new broker on the given `device` and `port` using the provided
    // `receiver`. If the initialization fails, then the status is propagated to
    // the caller. Otherwise, a unique pointer to the newly created `RdmaBroker`
    // is returned.
    static std::unique_ptr<RdmaBroker>
    Create(std::optional<std::string_view> address,
           std::optional<uint16_t> port, ConnectionManager *receiver) {
      // TODO: This method is fail-stop in the caller, so instead it should be
      // fail-stop here

      auto *broker = new RdmaBroker(receiver);
      auto status = broker->Init(address, port);
      if (status.t == sss::Ok) {
        return std::unique_ptr<RdmaBroker>(broker);
      } else {
        ROME_ERROR("{}: {}:{}", status.message.value(), address.value_or(""),
                   port.value_or(-1));
        return nullptr;
      }
    }

    RdmaBroker(const RdmaBroker &) = delete;
    RdmaBroker(RdmaBroker &&) = delete;

    // Getters.
    std::string address() const { return address_; }
    uint16_t port() const { return port_; }
    ibv_pd *pd() const { return listen_id_->pd; }

    // When shutting down the broker we must be careful. First, we signal to the
    // connection request handler that we are not accepting new requests.
    sss::Status Stop() {
      terminate_ = true;
      runner_.reset();
      return status_;
    }

  private:
    static constexpr int kMaxRetries = 100;

    RdmaBroker(ConnectionManager *receiver)
        : terminate_(false), status_(sss::Status::Ok()),
          listen_channel_(nullptr), listen_id_(nullptr), num_connections_(0),
          receiver_(receiver) {}

    // Start the broker listening on the given `device` and `port`. If `port` is
    // `nullopt`, then the first available port is used.
    sss::Status Init(std::optional<std::string_view> address,
                     std::optional<uint16_t> port) {
      // TODO: This method is fail-stop in the caller, so instead it should be
      // fail-stop here

      // Check that devices exist before trying to set things up.
      auto devices = GetAvailableDevices();
      if (!devices.has_value())
        return {sss::NotFound, "no devices found"}; // No devices found...
      // [mfs] Does the above call do anything?  Weird...

      rdma_addrinfo hints = {0}, *resolved;

      // Get the local connection information.
      hints.ai_flags = RAI_PASSIVE;
      hints.ai_port_space = RDMA_PS_TCP;

      auto port_str =
          port.has_value() ? std::to_string(htons(port.value())) : "";
      int gai_ret = rdma_getaddrinfo(
          address.has_value() ? address.value().data() : nullptr,
          port_str.data(), &hints, &resolved);
      if (gai_ret != 0) {
        sss::Status err = {sss::InternalError, "rdma_getaddrinfo(): "};
        err << gai_strerror(gai_ret);
        return err;
      }

      // Create an endpoint to receive incoming requests on.
      ibv_qp_init_attr init_attr = {0};
      init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = 16;
      init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = 1;
      init_attr.cap.max_inline_data = 0;
      init_attr.sq_sig_all = 1;
      auto err = rdma_create_ep(&listen_id_, resolved, nullptr, &init_attr);
      rdma_freeaddrinfo(resolved);
      if (err) {
        sss::Status err = {sss::InternalError, "rdma_create_ep(): "};
        err << strerror(errno);
        return err;
      }

      // Migrate the new endpoint to an async channel
      listen_channel_ = rdma_create_event_channel();
      RDMA_CM_CHECK(rdma_migrate_id, listen_id_, listen_channel_);
      RDMA_CM_CHECK(fcntl, listen_id_->channel->fd, F_SETFL,
                    fcntl(listen_id_->channel->fd, F_GETFL) | O_NONBLOCK);

      // Start listening for incoming requests on the endpoint.
      RDMA_CM_CHECK(rdma_listen, listen_id_, 0);

      address_ = std::string(inet_ntoa(
          reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(listen_id_))
              ->sin_addr));

      port_ = rdma_get_src_port(listen_id_);
      ROME_INFO("Listening: {}:{}", address_, port_);

      runner_.reset(new std::thread([&]() { this->Run(); }));

      return sss::Status::Ok();
    }

    void HandleConnectionRequests() {
      rdma_cm_event *event = nullptr;
      int ret;
      while (true) {
        do {
          // If we are shutting down, and there are no connections left then we
          // should finish.
          if (terminate_)
            return;

          // Attempt to read from `listen_channel_`
          ret = rdma_get_cm_event(listen_channel_, &event);
          if (ret != 0 && errno != EAGAIN) {
            status_ = {sss::InternalError, "rdma_get_cm_event(): "};
            status_ << strerror(errno);
            return;
          }
          std::this_thread::yield(); // TODO: is this right?

          // It was this at top, then an await/suspend here
          // std::this_thread::sleep_for(std::chrono::milliseconds(10))
        } while ((ret != 0 && errno == EAGAIN));

        ROME_TRACE("({}) Got event: {} (id={})", fmt::ptr(this),
                   rdma_event_str(event->event), fmt::ptr(event->id));
        switch (event->event) {
        case RDMA_CM_EVENT_TIMEWAIT_EXIT: // Nothing to do.
          rdma_ack_cm_event(event);
          break;
        case RDMA_CM_EVENT_CONNECT_REQUEST: {
          rdma_cm_id *id = event->id;
          receiver_->OnConnectRequest(id, event);
          break;
        }
        case RDMA_CM_EVENT_ESTABLISHED: {
          rdma_cm_id *id = event->id;
          // Now that we've established the connection, we can transition to
          // using it to communicate with the other node
          rdma_ack_cm_event(event);

          num_connections_.fetch_add(1);
          ROME_TRACE("({}) Num connections: {}", fmt::ptr(this),
                     num_connections_);
        } break;
        case RDMA_CM_EVENT_DISCONNECTED: {
          rdma_cm_id *id = event->id;
          rdma_ack_cm_event(event);
          receiver_->OnDisconnect(id);

          // `num_connections_` will only reach zero once all connections have
          // received their disconnect messages.
          num_connections_.fetch_add(-1);
          ROME_TRACE("({}) Num connections: {}", fmt::ptr(this),
                     num_connections_);
        } break;
        case RDMA_CM_EVENT_DEVICE_REMOVAL:
          // TODO: Cleanup
          ROME_ERROR("event: {}, error: {}\n", rdma_event_str(event->event),
                     event->status);
          break;
        case RDMA_CM_EVENT_ADDR_ERROR:
        case RDMA_CM_EVENT_ROUTE_ERROR:
        case RDMA_CM_EVENT_UNREACHABLE:
        case RDMA_CM_EVENT_ADDR_RESOLVED:
        case RDMA_CM_EVENT_REJECTED:
        case RDMA_CM_EVENT_CONNECT_ERROR:
          // These signals are sent to a connecting endpoint, so we should not
          // see them here. If they appear, abort.
          ROME_FATAL("Unexpected signal: {}", rdma_event_str(event->event));
        default:
          ROME_FATAL("Not implemented");
        }

        // TODO: is this right?  It used to be co_await suspend_always
        std::this_thread::yield();
        // co_await std::suspend_always{}; // Suspend after handling a given
        // message.
      }
    }

    void Run() {
      HandleConnectionRequests();
      ROME_TRACE("Finished: {}",
                 (status_.t == sss::Ok) ? "Ok" : status_.message.value());
    }
  };

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
  std::unique_ptr<RdmaBroker> broker_;

  // Maintains connection information for a given Internet address. A connection
  // manager only maintains a single connection per node. Nodes are identified
  // by a string representing their IP address.

  /// A mutex for protecting accesses to established_ and requested_
  /// TODO: use std::mutex instead?  Fix the locking bugs and races?
  std::atomic<int> mu_;

  std::unordered_map<uint32_t, std::unique_ptr<Connection>> requested_;
  std::unordered_map<uint32_t, std::unique_ptr<Connection>> established_;

  rdma_cm_id *loopback_id_ = nullptr;

public:
  ~ConnectionManager() {
    ROME_TRACE("Shutting down: {}", fmt::ptr(this));
    Acquire(my_id_);
    // Shutdown()
    accepting_ = false;
    // end Shutdown()

    ROME_TRACE("Stopping broker...");
    if (broker_ != nullptr)
      auto s = broker_->Stop();

    for (auto &iter : established_) {
      iter.second->cleanup(iter.first, my_id_);
    }

    Release();
  }

  explicit ConnectionManager(uint32_t my_id)
      : accepting_(false), my_id_(my_id), broker_(nullptr), mu_(kUnlocked) {}

  sss::Status Start(std::string_view addr, std::optional<uint16_t> port) {
    // TODO: This method is fail-stop in the caller, so instead it should be
    // fail-stop here
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
        ROME_TRACE(message);
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

  // `RdmaClientInterface` implementation
  sss::StatusVal<Connection *> Connect(uint32_t peer_id,
                                       std::string_view server, uint16_t port) {
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
        ROME_TRACE("[Acquire] (Node {}) Giving up lock acquisition: actual={}, "
                   "swap={}",
                   my_id_, expected, peer_id);
        return false;
      }
    }
    return true;
  }

  inline void Release() { mu_ = kUnlocked; }

  static ibv_qp_init_attr DefaultQpInitAttr() {
    ibv_qp_init_attr init_attr = {0};
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