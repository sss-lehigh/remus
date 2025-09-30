#pragma once

#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "logging.h"

namespace remus {
/// @brief A simple struct that contains the information needed to identify a
/// machine in the remus system.
/// @details
/// MachineInfo is used for translating between a machine's numerical id and its
/// DNS name.  Numerical ids start at zero and are contiguous.  Thus on
/// CloudLab, MachineInfo is not very important, because numbers start at 0, and
/// the names are things like "node0".  But on other systems, figuring out names
/// could be trickier, so having MachineInfo lets us avoid hard-coding an
/// association from numerical ids to DNS names.
///
/// TODO: field names should end with '_'
struct MachineInfo {
  const uint16_t id;          // A unique, 0-based Id for the machine
  const std::string address;  // The public address of the machine
};
}  // namespace remus

// TODO: This file is in pretty good shape:
// 1. The documentation needs a bit of additional work
// 2. Some code maybe should migrate here?  Or should single-use functions
//    migrate to the files where their callers reside?
// 3. There's a chance that there is code in here that we should be using more
//    broadly
// 4. There are some constants that we might not need.
namespace remus::internal {

// TODO:  I think all of these numbers need to be revisited
constexpr int kCapacity = 1 << 16;  // Send/Recv buffers are 4 KiB
constexpr int kMaxSge = 32;         // Max # SGEs in one RDMA write
constexpr int kMaxRecvSge = 1;      // Max # SGEs in one RDMA receive
constexpr int kMaxInlineData = 0;   // We aren't using INLINE data
constexpr int kMaxRecvBytes = 64;   // Max message size
constexpr int kMaxWr = kCapacity / kMaxRecvBytes;  // Max # outstanding writes

/// Set the file descriptor `fd` as O_NONBLOCK
inline void make_nonblocking(int fd) {
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) != 0) {
    REMUS_FATAL("fcntl(): {}", strerror(errno));
  }
}

/// Set the file descriptor `fd` as O_SYNC
inline void make_sync(int fd) {
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_SYNC) != 0) {
    REMUS_FATAL("fcntl(): {}", strerror(errno));
  }
}

/// Produce a vector of active RDMA ports, or None if none are found
inline std::vector<int> find_active_ports(ibv_context *context) {
  // Find the first active port, failing if none exists.
  //
  // NB:  port 0 is a control port, so we start at 1
  ibv_device_attr dev_attr;
  ibv_query_device(context, &dev_attr);
  std::vector<int> ports;
  for (int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
    ibv_port_attr port_attr;
    ibv_query_port(context, i, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE) {
      ports.push_back(i);
    }
  }
  return ports;
}

/// Returns a vector of device name and active port pairs that are accessible
/// on this machine, or None if no devices are found
inline std::vector<std::pair<std::string, int>> get_avail_devices() {
  int num_devices;
  auto **device_list = ibv_get_device_list(&num_devices);
  if (num_devices <= 0) return {};
  std::vector<std::pair<std::string, int>> active;
  for (int i = 0; i < num_devices; ++i) {
    if (auto *context = ibv_open_device(device_list[i]); context) {
      for (auto p : find_active_ports(context)) {
        active.emplace_back(context->device->name, p);
      }
    }
  }
  ibv_free_device_list(device_list);
  return active;
}

/// Configure the minimum attributes for a QP
///
/// TODO: Should this be used more broadly?
inline ibv_qp_init_attr make_default_qp_init_attrs() {
  // TODO: Where do these numbers come from?  Are they still valid?
  ibv_qp_init_attr init_attr;
  std::memset(&init_attr, 0, sizeof(ibv_qp_init_attr));
  init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
  init_attr.cap.max_send_sge = kMaxSge;
  init_attr.cap.max_recv_sge = kMaxRecvSge;
  init_attr.cap.max_inline_data = kMaxInlineData;
  init_attr.sq_sig_all = 0;  // Must request completions.
  init_attr.qp_type = IBV_QPT_RC;
  return init_attr;
}

// TODO: Documentation?
inline rdma_cm_id *make_listen_id(const std::string &address, uint16_t port) {
  rdma_cm_id *listen_id;
  // Check that devices exist before trying to set things up.
  auto devices = get_avail_devices();
  if (devices.empty()) {
    REMUS_FATAL("CreateListeningEndpoint :: no RDMA-capable devices found");
  }

  // Get the local connection information.
  rdma_addrinfo hints, *resolved;
  std::memset(&hints, 0, sizeof(rdma_addrinfo));
  hints.ai_flags = RAI_PASSIVE;
  hints.ai_port_space = RDMA_PS_TCP;
  auto port_str = std::to_string(htons(port));
  int gai_ret =
      rdma_getaddrinfo(address.c_str(), port_str.c_str(), &hints, &resolved);
  if (gai_ret != 0) {
    REMUS_FATAL("rdma_getaddrinfo(): {}", gai_strerror(gai_ret));
  }

  REMUS_ASSERT(resolved != nullptr, "Did not find an appropriate RNIC");

  // Create an endpoint to receive incoming requests
  ibv_qp_init_attr init_attr;
  std::memset(&init_attr, 0, sizeof(ibv_qp_init_attr));
  init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
  init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_inline_data = 0;
  init_attr.sq_sig_all = 1;
  auto err = rdma_create_ep(&listen_id, resolved, nullptr, &init_attr);
  rdma_freeaddrinfo(resolved);
  if (err != 0) {
    REMUS_FATAL("listener rdma_create_ep():{} for {}:{}", strerror(errno),
                address, port);
  }
  return listen_id;
}

/// @brief A structure to hold information about a memory region
/// @details
/// RegionInfo is a helper for passing data about a MemoryNode's segments to a
/// ComputeNode.
struct RegionInfo {
  uint64_t raddr;  // The base address of the segment
  uint32_t rkey;   // The rkey to use when accessing the segment
};

/// @brief A deleter for ibv_mr objects
/// @details
/// ibv_mr_deleter wraps a call to ibv_dereg_mr.  It is used by `ibv_mr_ptr`,
/// so that a unique_ptr can hold an RDMA memory region (ibv_mr*) and
/// correctly deregister
struct ibv_mr_deleter {
  void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
};
using ibv_mr_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

/// @brief A control block for managing segments in the distributed memory
/// system.
/// @details
/// ControlBlock is a header for each Segment managed by a Memory Node.  It
/// supports bump allocation and graceful shutdown, and offers some optional
/// space for a barrier (for synchronizing compute threads) and root pointer
/// (cast to rdma_ptr<> to reach the workload's root)
struct alignas(64) ControlBlock {
  const uint64_t size_;                 // The size of the segment
  std::atomic<uint64_t> allocated_;     // The number of allocated bytes
  std::atomic<uint64_t> control_flag_;  // A control flag, for shutdown
  std::atomic<uint64_t> barrier_;       // An optional barrier
  std::atomic<uint64_t> root_;          // An optional root pointer

  /// Initialize a ControlBlock with the provided size
  ControlBlock(uint64_t size)
      : size_(size),
        allocated_(sizeof(ControlBlock)),
        control_flag_(0),
        barrier_(0),
        root_(0) {}
};

/// @brief A pseudorandom number generator based on rdtsc
/// @details
/// A good-faith re-implementation of Fraser's PRNG.  We use the same magic
/// constants, and we seed the PRNG with the result of rdtsc.
class rdtsc_rand_t {
  uint64_t
      seed;  // The seed... should be 64 bits, even though we return 32 bits
 public:
  /// Construct the PRNG by setting the seed to the value of rdtsc
  rdtsc_rand_t() : seed(__rdtsc()) {}

  /// Generate a random number and update the seed
  uint32_t rand() { return (seed = (seed * 1103515245) + 12345); }
};
}  // namespace remus::internal
