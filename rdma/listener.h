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
#include "connection_manager.h"
#include "memory_pool.h"

namespace rome::rdma::internal {

/// A broker handles the connection setup using the RDMA CM library. It is
/// single threaded but communicates with all other brokers in the system to
/// exchange information regarding the underlying RDMA memory configurations.
///
/// TODO: Rename to Listener?
///
/// TODO: I feel like the behavior we really want out of this code is that it
///       should create a listening socket, and every time a new request comes
///       in on the socket, it should *fully connect* and then hand that
///       connection over to the connection manager.
class RdmaBroker {

  std::string address_;
  uint16_t port_;

  // Flag to indicate that the worker thread should terminate.
  std::atomic<bool> terminate_;

  /// The worker thread that listens and responds to incoming messages.
  struct thread_deleter {
    void operator()(std::thread *thread) {
      thread->join();
      free(thread);
    }
  };
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

  /// Produce a vector of active ports, or None if none are found
  static std::optional<std::vector<int>> FindActivePorts(ibv_context *context) {
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

  static constexpr int kMaxRetries = 100;

  RdmaBroker(ConnectionManager *receiver)
      : terminate_(false), status_(sss::Status::Ok()), listen_channel_(nullptr),
        listen_id_(nullptr), num_connections_(0), receiver_(receiver) {}

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

    auto port_str = port.has_value() ? std::to_string(htons(port.value())) : "";
    int gai_ret =
        rdma_getaddrinfo(address.has_value() ? address.value().data() : nullptr,
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
        receiver_->OnConnectRequest(id, event, listen_id_->pd);
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
  Create(std::optional<std::string_view> address, std::optional<uint16_t> port,
         ConnectionManager *receiver) {
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
};

} // namespace rome::rdma::internal