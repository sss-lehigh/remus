#include <algorithm>

#include "structures/hashtable.h"
#include "structures/iht_ds.h"
#include "structures/test_map.h"

#include "../rdma/connection_manager/connection_manager.h"
#include "../rdma/memory_pool/memory_pool.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "common.h"
#include "protos/experiment.pb.h"

using ::rome::rdma::MemoryPool;

// typedef RdmaIHT<int, int, CNF_ELIST_SIZE, CNF_PLIST_SIZE> IHT;
// typedef Hashtable<int, int, CNF_PLIST_SIZE> IHT;
typedef TestMap<int, int> IHT;

class Server {
public:
  ~Server() = default;

  static std::unique_ptr<Server> Create(MemoryPool::Peer server,
                                        std::vector<MemoryPool::Peer> clients,
                                        ExperimentParams params,
                                        MemoryPool *pool) {
    return std::unique_ptr<Server>(new Server(server, clients, params, pool));
  }

  /// @brief Start the server
  /// @param pool the memory pool to use
  /// @param done a bool for inter-thread communication
  /// @param runtime_s how long to wait before listening for finishing messages
  /// @return the status
  absl::Status Launch(volatile bool *done, int runtime_s,
                      std::function<void()> cleanup) {
    // Sleep while clients are running if there is a set runtime.
    if (runtime_s > 0) {
      ROME_INFO("SERVER :: Sleeping for {}", runtime_s);

      for (int it = 0; it < runtime_s * 10; it++) {
        // We sleep for 1 second, runtime times, and do intermitten cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check for server related cleanup
        cleanup();
      }
    }

    // Sync with the clients
    while (!(*done)) {
      // Sleep for 1/10 second while not done
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Check for server related cleanup
      cleanup();
    }

    // Wait for all clients to be done.
    for (auto &p : peers_) {
      if (p.id == self_.id)
        continue; // ignore self since joining threads will force client and
                  // server to end at the same time
      ROME_INFO("SERVER :: receiving ack from {}", p.id);
      auto conn_or = pool_->connection_manager()->GetConnection(p.id);
      if (!conn_or.ok())
        return conn_or.status();

      auto *conn = conn_or.value();
      auto msg = conn->channel()->TryDeliver<AckProto>();
      while ((!msg.ok() &&
              msg.status().code() == absl::StatusCode::kUnavailable)) {
        msg = conn->channel()->TryDeliver<AckProto>();
      }
      ROME_INFO("SERVER :: received ack");
    }

    // Let all clients know that we are done
    for (auto &p : peers_) {
      if (p.id == self_.id)
        continue; // ignore self since joining threads will force client and
                  // server to end at the same time
      ROME_INFO("SERVER :: sending ack to {}", p.id);
      auto conn_or = pool_->connection_manager()->GetConnection(p.id);
      if (!conn_or.ok())
        return conn_or.status();
      auto *conn = conn_or.value();
      AckProto e;
      // Send back an ack proto let the client know that all the other clients
      // are done
      auto sent = conn->channel()->Send(e);
      ROME_INFO("SERVER :: sent ack");
    }
    return absl::OkStatus();
  }

private:
  Server(MemoryPool::Peer self, std::vector<MemoryPool::Peer> peers,
         ExperimentParams params, MemoryPool *pool)
      : self_(self), peers_(peers), params_(params), pool_(pool) {}

  const MemoryPool::Peer self_;
  std::vector<MemoryPool::Peer> peers_;
  const ExperimentParams params_;
  MemoryPool *pool_;
};