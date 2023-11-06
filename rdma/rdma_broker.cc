// A broker handles the connection setup using the RDMA CM library. It is single
// threaded but communicates with all other brokers in the system to exchange
// information regarding the underlying RDMA memory configurations.
//
// In the future, this could potentially be replaced with a more robust
// component (e.g., Zookeeper) but for now we stick with a simple approach.
#include "rdma_broker.h"

#include <arpa/inet.h>
#include <asm-generic/errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <sys/stat.h>

#include <barrier>
#include <chrono>
#include <cmath>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ratio>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "../logging/logging.h"
#include "../util/coroutine.h"
#include "../util/status_util.h"
#include "rdma_device.h"
#include "rdma_receiver.h"
#include "rdma_util.h"

namespace rome::rdma {

using ::util::Coro;

RdmaBroker ::~RdmaBroker() {
  [[maybe_unused]] auto s = Stop();
  rdma_destroy_ep(listen_id_);
  // if (listen_channel_ != nullptr) {
  //   rdma_destroy_event_channel(listen_channel_);
  // }
}

std::unique_ptr<RdmaBroker>
RdmaBroker::Create(std::optional<std::string_view> address,
                   std::optional<uint16_t> port,
                   RdmaReceiverInterface *receiver) {
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

RdmaBroker::RdmaBroker(RdmaReceiverInterface *receiver)
    : terminate_(false), status_(sss::Status::Ok()), listen_channel_(nullptr),
      listen_id_(nullptr), num_connections_(0), receiver_(receiver) {}

sss::Status RdmaBroker::Init(std::optional<std::string_view> address,
                             std::optional<uint16_t> port) {
  // Check that devices exist before trying to set things up.
  auto devices = RdmaDevice::GetAvailableDevices();
  if (devices.status.t != sss::Ok)
    return devices.status;

  rdma_addrinfo hints, *resolved;

  // Get the local connection information.
  std::memset(&hints, 0, sizeof(hints));
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
  ibv_qp_init_attr init_attr;
  std::memset(&init_attr, 0, sizeof(init_attr));
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

  address_ = std::string(
      inet_ntoa(reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(listen_id_))
                    ->sin_addr));

  port_ = rdma_get_src_port(listen_id_);
  ROME_INFO("Listening: {}:{}", address_, port_);

  runner_.reset(new std::thread([&]() { this->Run(); }));

  return sss::Status::Ok();
}

// When shutting down the broker we must be careful. First, we signal to the
// connection request handler that we are not accepting new requests.
sss::Status RdmaBroker::Stop() {
  terminate_ = true;
  runner_.reset();
  return status_;
}

Coro RdmaBroker::HandleConnectionRequests() {
  rdma_cm_event *event = nullptr;
  int ret;
  while (true) {
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      // If we are shutting down, and there are no connections left then we
      // should finish.
      if (terminate_)
        co_return;

      // Attempt to read from `listen_channel_`
      ret = rdma_get_cm_event(listen_channel_, &event);
      if (ret != 0 && errno != EAGAIN) {
        status_ = {sss::InternalError, "rdma_get_cm_event(): "};
        status_ << strerror(errno);
        co_return;
      }
      co_await std::suspend_always{};
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
      // using it to communicate with the other node. This is handled in
      // another coroutine that we can resume every round.
      receiver_->OnEstablished(id, event);

      num_connections_.fetch_add(1);
      ROME_DEBUG("({}) Num connections: {}", fmt::ptr(this), num_connections_);
    } break;
    case RDMA_CM_EVENT_DISCONNECTED: {
      rdma_cm_id *id = event->id;
      rdma_ack_cm_event(event);
      receiver_->OnDisconnect(id);

      // `num_connections_` will only reach zero once all connections have
      // recevied their disconnect messages.
      num_connections_.fetch_add(-1);
      ROME_DEBUG("({}) Num connections: {}", fmt::ptr(this), num_connections_);
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
      // These signals are sent to a connecting endpoint, so we should not see
      // them here. If they appear, abort.
      ROME_FATAL("Unexpected signal: {}", rdma_event_str(event->event));
    default:
      ROME_FATAL("Not implemented");
    }

    co_await std::suspend_always{}; // Suspend after handling a given message.
  }
}

void RdmaBroker::Run() {
  scheduler_.Schedule(HandleConnectionRequests());
  scheduler_.Run();
  ROME_TRACE("Finished: {}",
             (status_.t == sss::Ok) ? "Ok" : status_.message.value());
}

} // namespace rome::rdma