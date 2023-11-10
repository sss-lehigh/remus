#pragma once

#include <cstdint>
#include <string>

namespace rome::rdma {

/// Peer is used to describe another node in the system
///
/// NB: It appears that all nodes agree on the mapping between an id and an
///     address:port
struct Peer {
  uint16_t id;         // A unique Id for the peer
  std::string address; // The public address of the peer
  uint16_t port;       // The port on which the peer accepts connections

  /// Construct a Peer
  ///
  /// @param id       The Id for the peer (default 0)
  /// @param address  The ip address as a string (default "")
  /// @param port     The port (default 0)
  Peer(uint16_t id = 0, std::string address = "", uint16_t port = 0)
      : id(id), address(address), port(port) {}
};

}