#pragma once

#include <cstdint>
#include <string>

namespace rome::rdma {
/// Peer describes another node in the system
///
/// TODO: do all nodes agree on the mapping between id and address:port?
///
/// TODO: This really doesn't have much to do with a MemoryPool, but it's in
///       this file?
struct Peer {
  uint16_t id;
  std::string address;
  uint16_t port;

  Peer() : Peer(0, "", 0) {}
  Peer(uint16_t id, std::string address, uint16_t port)
      : id(id), address(address), port(port) {}
};

} // namespace rome::rdma