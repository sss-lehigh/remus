#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../logging/logging.h"
#include "connection.h"

namespace rome::rdma::internal {

/// A map holding all of the open Connection objects between this node and all
/// nodes in the system.  Each node is identified by its numerical id.  This
/// includes the connections between this node and itself (via loopback).
///
/// TODO: This is a multimap... it supports multiple connections between a pair
///       of nodes.  However, it does not yet *return* any but the first
///       connection in response to a get().  Supporting such a feature will
///       require some significant work, because it should involve some degree
///       of thread affinity.
class ConnectionMap {
  /// Identifier for this ConnectionMap's node
  uint32_t my_id_;

  /// a map holding all of the connections involving this machine.  Note that
  /// this includes loopback connections
  std::unordered_map<uint32_t, std::vector<std::unique_ptr<Connection>>>
      connections_;

  /// A mutex protecting connections_
  std::mutex con_mu_;

public:
  /// Save a connection by putting it in the map with its associated Id.
  ///
  /// NB: Takes ownership of the provided `conn`
  bool put_connection(uint32_t id, Connection *conn) {
    std::lock_guard<std::mutex> lg(con_mu_);
    auto vec = connections_.find(id);
    // case 1: there is no vector for this id yet, so make one
    if (vec == connections_.end()) {
      std::vector<std::unique_ptr<Connection>> v;
      v.emplace_back(conn);
      connections_.emplace(id, std::move(v));
      return true;
    }
    // case 2: otherwise we can just add this connection to the map
    bool res = (vec->second.size() == 0);
    vec->second.emplace_back(conn);
    return res;
  }

  /// Destruct the ConnectionMap by iterating through its open connections,
  /// and for each one, shutting it down.
  ~ConnectionMap() {
    ROME_TRACE("Shutting down: {}", fmt::ptr(this));
    // NB: By the time this runs, we should be sequential, so no synchronization
    // is needed
    for (auto &vec : connections_)
      for (auto &i : vec.second)
        i->cleanup(vec.first, my_id_);
  }

  /// Construct the ConnectionMap by assigning it its node Id
  explicit ConnectionMap(uint32_t my_id) : my_id_(my_id) {}

  /// Get a connection from the map of connections.  This assumes the map cannot
  /// be mutated, and thus no synchronization is needed.  Terminates if no
  /// connection is found.
  ///
  /// TODO: The map is capable of holding >1 connection, but we always return
  ///       the first one.
  Connection *GetConnection(uint32_t peer_id) {
    using namespace std::string_literals;

    std::lock_guard<std::mutex> lg(con_mu_);
    auto vec = connections_.find(peer_id);
    if (vec == connections_.end() || vec->second.size() == 0)
      ROME_FATAL("Connection not found: "s + std::to_string(peer_id));
    return vec->second[0].get();
  }
};
} // namespace rome::rdma::internal