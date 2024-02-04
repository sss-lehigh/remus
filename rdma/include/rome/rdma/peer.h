#pragma once

#include <cstdint>
#include <string>

namespace rome::rdma {
/// Peer is used to describe a node in the system
///
/// NB: It appears that all nodes agree on the mapping between an id and an
///     address:port
struct Peer {
  const uint16_t id = 0;          // A unique Id for the peer
  const std::string address = ""; // The public address of the peer
  const uint16_t port = 0;        // The port on which the peer listens

  /// Construct a Peer
  ///
  /// TODO: We probably don't need a constructor since this is so trivial
  ///
  /// @param id       The Id for the peer (default 0)
  /// @param address  The ip address as a string (default "")
  /// @param port     The port (default 0)
  Peer(uint16_t id = 0, std::string address = "", uint16_t port = 0)
      : id(id), address(address), port(port) {}
};
} // namespace rome::rdma