#pragma once

#include <arpa/inet.h>
#include <atomic>
#include <coroutine>
#include <cstdint>
#include <exception>
#include <memory>
#include <netdb.h>
#include <rdma/rdma_cma.h>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"
#include "device.h"
#include "receiver.h"

namespace util {

// Forward declaration necessary so that `from_promise()` is defined for our
// coroutine handle. There may be a cleaner way to accomplish this, but this how
// its done here, https://en.cppreference.com/w/cpp/language/coroutines.
class Promise;

// For our purposes, the return object of a coroutine is just a wrapper for the
// `promise_type`. Any coroutine that is used with the scheduler must return a
// `Task`.
class Coro : public std::coroutine_handle<Promise> {
public:
  using promise_type = Promise;
  using handler_type = std::coroutine_handle<Promise>;
};

// The promise object of a coroutine dictates behavior when the coroutine first
// starts, and when it returns. It also can save some state to be queried later.
class Promise {
public:
  Coro get_return_object() {
    return {std::coroutine_handle<Promise>::from_promise(*this)};
  }

  std::suspend_always initial_suspend() { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void unhandled_exception() {
    std::rethrow_exception(std::current_exception());
  }
  void return_void() {}
};

// The interface for all coroutine schedulers. A scheduler can add new
// coroutines, start running, and cancel running. So
//
// TODO: Make this a concept?
template <typename PromiseT> class Scheduler {
public:
  // Adds a new coroutine to the runner to be run with a given policy.
  virtual void Schedule(std::coroutine_handle<PromiseT> task) = 0;

  // Starts running the coroutines until `Stop()` is called or some other
  // temination condition is reached.
  virtual void Run() = 0;

  // Cancels running the coroutines.
  virtual void Cancel() = 0;
};

using Cancellation = std::atomic<bool>;

// [mfs]  If we only use one scheduler, should it be hard-coded (or at least not
//        virtual dispatch?)
template <typename PromiseT>
class RoundRobinScheduler : public Scheduler<PromiseT> {
  struct CoroWrapper {
    ~CoroWrapper() { handle.destroy(); }
    std::coroutine_handle<PromiseT> handle;
    CoroWrapper *prev;
    CoroWrapper *next;
  };

  int task_count_;
  CoroWrapper *curr_;
  CoroWrapper *last_;
  std::atomic<bool> canceled_;

public:
  ~RoundRobinScheduler() { ROME_TRACE("Task count: {}", task_count_); }
  RoundRobinScheduler() : task_count_(0), curr_(nullptr), canceled_(false) {}

  // Getters.
  int task_count() const { return task_count_; }

  // Inserts the given task as the next task to run.
  void Schedule(std::coroutine_handle<PromiseT> task) override {
    if (canceled_)
      return;
    auto coro = new CoroWrapper{task, nullptr, nullptr};
    if (curr_ == nullptr) {
      coro->prev = coro;
      coro->next = coro;
      curr_ = coro;
      last_ = curr_;
    } else {
      coro->prev = last_;
      coro->next = last_->next;
      last_->next->prev = coro;
      last_->next = coro;
      last_ = coro;
    }
    ++task_count_;
  }

  void Run() override {
    ROME_ASSERT(curr_ != nullptr,
                "You must schedule at least one task before running");
    while (curr_ != nullptr) {
      if (curr_->handle.done()) {
        if (curr_->next == curr_) {
          // Only one coroutine was left...
          --task_count_;
          delete curr_;
          last_ = nullptr;
          std::atomic_thread_fence(std::memory_order_release);
          curr_ = nullptr;
          continue;
        }

        // Unlink `curr_` by setting its predecessor's next to `curr_`'s next.
        // And setting the successors's prev to `curr_`'s prev. Finally, update
        // the tail of the list to point to `curr_`'s predecessor.
        curr_->prev->next = curr_->next;
        curr_->next->prev = curr_->prev;
        if (last_ == curr_) {
          last_ = curr_->prev;
        }

        // Cleanup the removed `Task` by deleting the `TaskWrapper`, which in
        // turn takes care of destroying the underlying coroutine handle.
        auto *temp = curr_;
        curr_ = curr_->next;
        --task_count_;
        delete temp;
      } else {
        curr_->handle.resume();
        curr_ = curr_->next;
      }
    }
  }

  // Cancels the scheduler and then waits for all currently scheduled tasks to
  // complete. Coroutines can obtain a pointer to the Cancellation flag using
  // `Cancellation()` and then check if it has been canceled.
  void Cancel() override {
    canceled_ = true;
    while (curr_ != nullptr)
      ;
  }

  const Cancellation &Cancellation() const { return canceled_; }
};

} // namespace util
namespace rome::rdma {

using Scheduler = util::RoundRobinScheduler<util::Promise>;

// A broker handles the connection setup using the RDMA CM library. It is single
// threaded but communicates with all other brokers in the system to exchange
// information regarding the underlying RDMA memory configurations.
//
// In the future, this could potentially be replaced with a more robust
// component (e.g., Zookeeper) but for now we stick with a simple approach.
class RdmaBroker {
  // The working thread that listens and responds to incoming messages.
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
  sss::Status status_;

  // RDMA CM related members.
  rdma_event_channel *listen_channel_;
  rdma_cm_id *listen_id_;

  // The total number of connections made by this broker.
  std::atomic<uint32_t> num_connections_;

  // Maintains connections that are forwarded by the broker.
  RdmaReceiverInterface *receiver_; //! NOT OWNED

  // Runs connection handler coroutine.
  Scheduler scheduler_;

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

  RdmaBroker(RdmaReceiverInterface *receiver)
      : terminate_(false), status_(sss::Status::Ok()), listen_channel_(nullptr),
        listen_id_(nullptr), num_connections_(0), receiver_(receiver) {}

  // Start the broker listening on the given `device` and `port`. If `port` is
  // `nullopt`, then the first available port is used.
  sss::Status Init(std::optional<std::string_view> address,
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

    address_ = std::string(inet_ntoa(
        reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(listen_id_))
            ->sin_addr));

    port_ = rdma_get_src_port(listen_id_);
    ROME_INFO("Listening: {}:{}", address_, port_);

    runner_.reset(new std::thread([&]() { this->Run(); }));

    return sss::Status::Ok();
  }

  // NB: Coroutine
  util::Coro HandleConnectionRequests() {
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

      ROME_DEBUG("({}) Got event: {} (id={})", fmt::ptr(this),
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
        ROME_DEBUG("({}) Num connections: {}", fmt::ptr(this),
                   num_connections_);
      } break;
      case RDMA_CM_EVENT_DISCONNECTED: {
        rdma_cm_id *id = event->id;
        rdma_ack_cm_event(event);
        receiver_->OnDisconnect(id);

        // `num_connections_` will only reach zero once all connections have
        // received their disconnect messages.
        num_connections_.fetch_add(-1);
        ROME_DEBUG("({}) Num connections: {}", fmt::ptr(this),
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
        // These signals are sent to a connecting endpoint, so we should not see
        // them here. If they appear, abort.
        ROME_FATAL("Unexpected signal: {}", rdma_event_str(event->event));
      default:
        ROME_FATAL("Not implemented");
      }

      co_await std::suspend_always{}; // Suspend after handling a given message.
    }
  }

  void Run() {
    scheduler_.Schedule(HandleConnectionRequests());
    scheduler_.Run();
    ROME_TRACE("Finished: {}",
               (status_.t == sss::Ok) ? "Ok" : status_.message.value());
  }
};

} // namespace rome::rdma