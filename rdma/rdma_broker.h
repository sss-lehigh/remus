#pragma once
// A broker handles the connection setup using the RDMA CM library. It is single
// threaded but communicates with all other brokers in the system to exchange
// information regarding the underlying RDMA memory configurations.
//
// In the future, this could potentially be replaced with a more robust
// component (e.g., Zookeeper) but for now we stick with a simple approach.

#include <arpa/inet.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include <barrier>
#include <chrono>
#include <cmath>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "../logging/logging.h"
#include "../util/coroutine.h"
#include "../util/status_util.h"
#include "../vendor/sss/status.h"
#include "rdma_receiver.h"

namespace rome::rdma {

using ::util::Coro;
using ::util::RoundRobinScheduler;
using Scheduler = RoundRobinScheduler<::util::Promise>;

class RdmaBroker {
public:
  ~RdmaBroker();

  // Creates a new broker on the given `device` and `port` using the provided
  // `receiver`. If the initialization fails, then the status is propagated to
  // the caller. Otherwise, a unique pointer to the newly created `RdmaBroker`
  // is returned.
  static std::unique_ptr<RdmaBroker>
  Create(std::optional<std::string_view> address, std::optional<uint16_t> port,
         RdmaReceiverInterface *receiver);

  RdmaBroker(const RdmaBroker &) = delete;
  RdmaBroker(RdmaBroker &&) = delete;

  // Getters.
  std::string address() const { return address_; }
  uint16_t port() const { return port_; }
  ibv_pd *pd() const { return listen_id_->pd; }

  sss::Status Stop();

private:
  static constexpr int kMaxRetries = 100;

  RdmaBroker(RdmaReceiverInterface *receiver);

  // Start the broker listening on the given `device` and `port`. If `port` is
  // `nullopt`, then the first available port is used.
  sss::Status Init(std::optional<std::string_view> addr,
                   std::optional<uint16_t> port);

  Coro HandleConnectionRequests();

  void Run();

  std::string address_;
  uint16_t port_;

  // Flag to indicate that the worker thread should terminate.
  std::atomic<bool> terminate_;

  // The working thread that listens and responds to incoming messages.
  struct thread_deleter {
    void operator()(std::thread *thread) {
      thread->join();
      free(thread);
    }
  };
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
};

} // namespace rome::rdma