#pragma once

#include <atomic>
#include <functional>
#include <rdma/rdma_cma.h>
#include <thread>

#include "remus/logging/logging.h"
#include "connection_utils.h"
#include "connection.h"

namespace remus::rdma::internal {

/// A listening endpoint and a thread that listens for new connections
class Listener {
  /// A context to associate with an `rdma_cm_id`.  `node_id` is the numerical
  /// identifier for the peer node of the connection and `conn_param` is used to
  /// provide private data during the connection set up to send the local node
  /// identifier.
  struct IdContext {
    uint32_t node_id;           // The peer's id
    rdma_conn_param conn_param; // Private data to send during config

    /// Extract a node id from a context object
    static inline uint32_t GetNodeId(void *ctx) {
      return reinterpret_cast<IdContext *>(ctx)->node_id;
    }
  };

  /// A functor for joining a thread on deletion
  struct thread_deleter {
    void operator()(std::thread *thread) {
      thread->join();
      free(thread);
    }
  };

  /// A thread that joins itself on deletion
  using receiving_thread = std::unique_ptr<std::thread, thread_deleter>;

  /// A function type; used for saving a connection to the Connection Map
  using saver_t = std::function<void(uint32_t, Connection *)>;

  std::atomic<bool> stop_listening_;             // For stopping worker thread
  rdma_cm_id *listen_id_ = nullptr;              // The listening endpoint
  rdma_event_channel *listen_channel_ = nullptr; // The channel for listen_id_
  receiving_thread runner_;                      // The worker thread
  std::string address_;                          // This node's IP address
  uint16_t port_;                                // This node's port
  uint32_t my_id_;                               // Node ID of this thread
  const saver_t connection_saver; // Saves a connection to the Connection Map

  /// Create a listening endpoint on RDMA and start listening on it.  Terminate
  /// the program if anything goes wrong.
  ///
  /// NB: Since we are using an async socket, we call `listen()` here, but it
  ///     won't block or run any code for handling a listening request.  We'll
  ///     do that later, in a separate thread.
  void CreateListeningEndpoint(const std::string &address, uint16_t port) {
    using namespace std::string_literals;

    // Check that devices exist before trying to set things up.
    auto devices = GetAvailableDevices();
    if (!devices.has_value()) {
      REMUS_FATAL("CreateListeningEndpoint :: no RDMA-capable devices found");
    }

    // Get the local connection information.
    rdma_addrinfo hints = {0}, *resolved;
    hints.ai_flags = RAI_PASSIVE;
    hints.ai_port_space = RDMA_PS_TCP;
    auto port_str = std::to_string(htons(port));
    int gai_ret =
        rdma_getaddrinfo(address.c_str(), port_str.c_str(), &hints, &resolved);
    if (gai_ret != 0) {
      REMUS_FATAL("rdma_getaddrinfo(): "s + gai_strerror(gai_ret));
    }

    REMUS_ASSERT(resolved != nullptr, "Did not find an appropriate RNIC");

    // Create an endpoint to receive incoming requests
    ibv_qp_init_attr init_attr = {0};
    init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = 16;
    init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 0;
    init_attr.sq_sig_all = 1;
    auto err = rdma_create_ep(&listen_id_, resolved, nullptr, &init_attr);
    rdma_freeaddrinfo(resolved);
    if (err != 0) {
      REMUS_FATAL("rdma_create_ep(): "s + strerror(errno));
    }

    REMUS_ASSERT(listen_id_->pd != nullptr, "Should initialize a protection domain");

    // Migrate the new endpoint to an async channel
    listen_channel_ = rdma_create_event_channel();
    if (rdma_migrate_id(listen_id_, listen_channel_) != 0) {
      REMUS_FATAL("rdma_migrate_id(): "s + strerror(errno));
    }
    make_nonblocking(listen_id_->channel->fd);

    // Start listening for incoming requests on the endpoint.
    if (rdma_listen(listen_id_, 0) != 0) {
      REMUS_FATAL("rdma_listen(): "s + strerror(errno));
    }

    // Record and report the node's address/port
    address_ = std::string(inet_ntoa(
        reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(listen_id_))
            ->sin_addr));
    port_ = rdma_get_src_port(listen_id_);
    REMUS_INFO("Listening: {}:{}", address_, port_);
  }

  /// The main loop run by the listening thread.  Polls for new events on the
  /// listening endpoint and handles them.  This typically means receiving new
  /// connections, configuring the new endpoints, and then putting them into the
  /// map.
  void HandleConnectionRequests() {
    using namespace std::string_literals;

    while (true) {
      // Check if shutdown was requested
      if (stop_listening_) {
        return;
      }

      // Attempt to read an event from `listen_channel_`
      rdma_cm_event *event = nullptr;
      {
        int ret = rdma_get_cm_event(listen_channel_, &event);
        if (ret != 0 && errno != EAGAIN) {
          REMUS_FATAL("rdma_get_cm_event(): "s + strerror(errno));
        }
        // On EAGAIN, yield and then try again
        //
        // TODO: Is yielding the right thing to do?  Sleep?  Tight spin loop?
        if (ret != 0) {
          std::this_thread::yield();
          continue;
        }
      }
      REMUS_TRACE("({}) Got event: {} (id={})", fmt::ptr(this),
                 rdma_event_str(event->event), fmt::ptr(event->id));

      // Handle whatever event we just received
      rdma_cm_id *id = event->id;
      switch (event->event) {
      case RDMA_CM_EVENT_TIMEWAIT_EXIT:
        // We aren't currently getting CM_EVENT_TIMEWAIT_EXIT events, but if we
        // did, we'd probably just ACK them and continue
        //
        // TODO:  Why doesn't this code check for ACK errors, when the Connector
        //        does?
        rdma_ack_cm_event(event);
        break;

      case RDMA_CM_EVENT_CONNECT_REQUEST:
        // This is the main thing we expect: a request for a new connection
        OnConnectRequest(id, event, listen_id_->pd);
        break;

      case RDMA_CM_EVENT_ESTABLISHED:
        // Once the connection is fully established, we just ack it, and then
        // it's ready for use.
        //
        // TODO:  Should we be updating the map?  What if another thread tries
        //        to use it before we ack?  Are we just getting lucky with
        //        potential timing bugs, because we set up the whole clique
        //        before we use connections?
        rdma_ack_cm_event(event);
        break;

      case RDMA_CM_EVENT_DISCONNECTED:
        // Since we're polling on a *listening* channel, we're never going to
        // see disconnected events.  If we did, we have some code for knowing
        // what to do with them.
        rdma_ack_cm_event(event);
        OnDisconnect(id);
        break;

      case RDMA_CM_EVENT_DEVICE_REMOVAL:
        // We don't expect to ever see a device removal event
        REMUS_ERROR("event: {}, error: {}\n", rdma_event_str(event->event),
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
        REMUS_FATAL("Unexpected signal: {}", rdma_event_str(event->event));

      default:
        // We did not design for other events, so crash if another event arrives
        REMUS_FATAL("Not implemented");
      }
    }
  }

  /// Handler to run when a connection request arrives
  void OnConnectRequest(rdma_cm_id *id, rdma_cm_event *event, ibv_pd *pd) {
    using namespace std::string_literals;

    // The private data is used to figure out the node that made the request
    REMUS_ASSERT_DEBUG(event->param.conn.private_data != nullptr,
                      "Received connect request without private data.");
    uint32_t peer_id = *(uint32_t *)(event->param.conn.private_data);
    REMUS_TRACE("[OnConnectRequest] (Node {}) Got connection request from: {}",
               my_id_, peer_id);

    if (peer_id != my_id_) {
      // Create a new QP for the connection.
      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      REMUS_ASSERT(id->qp == nullptr, "QP already allocated...?");
      RDMA_CM_ASSERT(rdma_create_qp, id, pd, &init_attr);
    } else {
      REMUS_FATAL("OnConnectionRequest called for self-connection");
    }

    // Prepare the necessary resources for this connection.  Includes a QP and
    // memory for 2-sided communication. The underlying QP is RC, so we reuse it
    // for issuing 1-sided RDMA too. We also store the `peer_id` associated with
    // this id so that we can reference it later.
    auto context = new IdContext{peer_id, {}};
    context->conn_param.private_data = &context->node_id;
    context->conn_param.private_data_len = sizeof(context->node_id);
    context->conn_param.rnr_retry_count = 1; // Retry forever
    context->conn_param.retry_count = 7;
    context->conn_param.responder_resources = 8;
    context->conn_param.initiator_depth = 8;
    id->context = context;
    make_nonblocking(id->recv_cq->channel->fd);
    make_nonblocking(id->send_cq->channel->fd);
    // Save the connection and ack it
    connection_saver(peer_id, new Connection(my_id_, peer_id, id));

    REMUS_TRACE("[OnConnectRequest] (Node {}) peer={}, id={}", my_id_, peer_id,
               fmt::ptr(id));
    RDMA_CM_ASSERT(rdma_accept, id,
                   peer_id == my_id_ ? nullptr : &context->conn_param);
    rdma_ack_cm_event(event);
  }

  /// This is not live code, we just have it here for reference.  If we wanted
  /// to handle dropped connections gracefully, we could do something like this
  /// (but not exactly this).
  void OnDisconnect(rdma_cm_id *id) {
    // NB:  The event is already ack'ed by the caller, and the remote peer
    //      already disconnected, so we just clean up...
    rdma_disconnect(id);
    uint32_t peer_id = IdContext::GetNodeId(id->context);
    auto *event_channel = id->channel;
    rdma_destroy_ep(id);
    rdma_destroy_event_channel(event_channel);
    // ... And don't forget to remove peer_id from the map of connections ...
  }

public:
  /// Construct a Listener by saving its node Id and the function for saving
  /// connections
  Listener(uint32_t my_id, std::function<void(uint32_t, Connection *)> saver)
      : my_id_(my_id), connection_saver(saver) {}

  /// Initialize a listening endpoint, and then park a thread on it, so the
  /// thread can receive new connection requests
  void StartListeningThread(const std::string &address, uint16_t port) {
    // Don't allow multiple threads at once
    if (runner_ != nullptr) {
      REMUS_FATAL("Cannot start more than one listener");
    }

    // Create the endpoint, park a thread on it
    CreateListeningEndpoint(address, port);
    runner_.reset(new std::thread([&]() { this->HandleConnectionRequests(); }));
  }

  /// Stop listening for new connections, and terminate the listening thread
  ///
  /// NB: This blocks the caller until the listening thread has been joined
  void StopListeningThread() {
    stop_listening_ = true;      // Tell the thread to stop
    runner_.reset();             // This will join on the thread
    rdma_destroy_ep(listen_id_); // Now we can destroy the endpoint
  }

  /// Report the listener's RDMA protection domain (PD)
  ///
  /// TODO: Why not just return it from StartListeningThread?
  ibv_pd *pd() const { return listen_id_->pd; }

  /// Report the listener's address
  ///
  /// TODO: Why not just return it from StartListeningThread?
  std::string address() { return address_; }
};

} // namespace remus::rdma::internal
