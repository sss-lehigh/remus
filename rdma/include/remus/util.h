#pragma once

#include <cstdint>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <string>
#include <vector>
#include <ifaddrs.h>
#include <arpa/inet.h>

#include "logging.h"

namespace remus {
/// MachineInfo is used for translating between a machine's numerical id and its
/// DNS name.  Numerical ids start at zero and are contiguous.  Thus on
/// CloudLab, MachineInfo is not very important, because numbers start at 0, and
/// the names are things like "node0".  But on other systems, figuring out names
/// could be trickier, so having MachineInfo lets us avoid hard-coding an
/// association from numerical ids to DNS names.
struct MachineInfo {
  const uint16_t id;         // A unique, 0-based Id for the machine
  const std::string address; // The public address of the machine
};
} // namespace remus

// [mfs] This file is in pretty good shape:
// 1. The documentation needs a bit of additional work
// 2. Some code maybe should migrate here?  Or should single-use functions
//    migrate to the files where their callers reside?
// 3. There's a chance that there is code in here that we should be using more
//    broadly
// 4. There are some constants that we might not need.
namespace remus::internal {

// [mfs]  I think all of these numbers need to be revisted
constexpr int kCapacity = 1 << 16; // Send/Recv buffers are 4 KiB
constexpr int kMaxSge = 32;        // Max # SGEs in one RDMA write
constexpr int kMaxRecvSge = 1;     // Max # SGEs in one RDMA receive
constexpr int kMaxInlineData = 0;  // We aren't using INLINE data
constexpr int kMaxRecvBytes = 64;  // Max message size
constexpr int kMaxWr = kCapacity / kMaxRecvBytes; // Max # outstanding writes

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
  if (num_devices <= 0)
    return {};
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
/// [mfs] Should this be used more broadly?
inline ibv_qp_init_attr make_default_qp_init_attrs() {
  // [mfs] Where do these numbers come from?  Are they still valid?
  ibv_qp_init_attr init_attr;
  std::memset(&init_attr, 0, sizeof(ibv_qp_init_attr));
  init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
  init_attr.cap.max_send_sge = kMaxSge;
  init_attr.cap.max_recv_sge = kMaxRecvSge;
  init_attr.cap.max_inline_data = kMaxInlineData;
  init_attr.sq_sig_all = 0; // Must request completions.
  init_attr.qp_type = IBV_QPT_RC;
  return init_attr;
}

// [mfs] Documentation?
// inline rdma_cm_id *make_listen_id(const std::string &address, uint16_t port) {
//   rdma_cm_id *listen_id;
//   // Check that devices exist before trying to set things up.
//   auto devices = get_avail_devices();
//   if (devices.empty()) {
//     REMUS_FATAL("CreateListeningEndpoint :: no RDMA-capable devices found");
//   }

//   // Get the local connection information.
//   rdma_addrinfo hints, *resolved;
//   std::memset(&hints, 0, sizeof(rdma_addrinfo));
//   hints.ai_flags = RAI_PASSIVE;
//   hints.ai_port_space = RDMA_PS_TCP;
//   auto port_str = std::to_string(htons(port));
//   int gai_ret =
//       rdma_getaddrinfo(address.c_str(), port_str.c_str(), &hints, &resolved);
//   if (gai_ret != 0) {
//     REMUS_FATAL("rdma_getaddrinfo(): {}", gai_strerror(gai_ret));
//   }

//   REMUS_ASSERT(resolved != nullptr, "Did not find an appropriate RNIC");

//   // Create an endpoint to receive incoming requests
//   ibv_qp_init_attr init_attr;
//   std::memset(&init_attr, 0, sizeof(ibv_qp_init_attr));
//   init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
//   init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = 1;
//   init_attr.cap.max_inline_data = 0;
//   init_attr.sq_sig_all = 1;
//   auto err = rdma_create_ep(&listen_id, resolved, nullptr, &init_attr);
//   rdma_freeaddrinfo(resolved);
//   if (err != 0) {
//     REMUS_FATAL("listener rdma_create_ep():{} for {}:{}", strerror(errno), address,
//                 port);
//   }
//   return listen_id;
// }

/// Get the network interface name for an IP address
inline std::string get_interface_for_ip(const std::string& ip_address) {
  // Use getifaddrs to enumerate all interfaces
  struct ifaddrs *ifaddr, *ifa;
  if (getifaddrs(&ifaddr) == -1) {
    return "";
  }

  std::string result;
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    
    // Only check IPv4 addresses
    if (ifa->ifa_addr->sa_family == AF_INET) {
      char host[NI_MAXHOST];
      if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                      host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
        if (std::string(host) == ip_address) {
          result = ifa->ifa_name;
          break;
        }
      }
    }
  }

  freeifaddrs(ifaddr);
  return result;
}

/// Find RDMA device by network interface name
inline std::string find_device_by_netdev(const std::string& netdev_name) {
  int num_devices;
  auto **device_list = ibv_get_device_list(&num_devices);
  if (num_devices <= 0) {
    return "";
  }

  for (int i = 0; i < num_devices; ++i) {
    auto *context = ibv_open_device(device_list[i]);
    if (!context) continue;

    // Check all ports
    ibv_device_attr dev_attr;
    if (ibv_query_device(context, &dev_attr) != 0) {
      ibv_close_device(context);
      continue;
    }

    for (int port = 1; port <= dev_attr.phys_port_cnt; ++port) {
      ibv_port_attr port_attr;
      if (ibv_query_port(context, port, &port_attr) != 0) continue;
      if (port_attr.state != IBV_PORT_ACTIVE) continue;

      // Check sysfs to find associated network device
      char path[256];
      snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/gid_attrs/ndevs/0",
               context->device->name, port);
      
      FILE* f = fopen(path, "r");
      if (f) {
        char ndev[64];
        if (fgets(ndev, sizeof(ndev), f)) {
          // Remove newline
          ndev[strcspn(ndev, "\n")] = 0;
          
          if (std::string(ndev) == netdev_name) {
            std::string dev_name = context->device->name;
            fclose(f);
            ibv_close_device(context);
            ibv_free_device_list(device_list);
            return dev_name;
          }
        }
        fclose(f);
      }
    }

    ibv_close_device(context);
  }

  ibv_free_device_list(device_list);
  return "";
}

/// Find the RDMA device for a given IP address
inline std::string find_device_for_address(const std::string& ip_address) {
  // First, find which network interface has this IP
  std::string netdev = get_interface_for_ip(ip_address);
  if (netdev.empty()) {
    return "";
  }

  // Then find which RDMA device corresponds to that interface
  return find_device_by_netdev(netdev);
}

/// Find the experimental device using the node's address
inline std::string find_experimental_device(const std::string& my_address) {
  auto dev = find_device_for_address(my_address);
  if (!dev.empty()) {
    return dev;
  }

  // Fallback: look for any device on 10.10.1.x subnet
  for (int i = 1; i < 255; i++) {
    std::string test_ip = "10.10.1." + std::to_string(i);
    dev = find_device_for_address(test_ip);
    if (!dev.empty()) {
      return dev;
    }
  }

  // Last resort
  auto devices = get_avail_devices();
  if (!devices.empty()) {
    return devices[0].first;
  }

  return "";
}

inline rdma_cm_id *make_listen_id(const std::string &address, uint16_t port,
                                   const std::string &device_name) {
  REMUS_INFO("Binding listener to device {} at {}:{}", device_name, address, port);
  
  // Resolve the address properly
  rdma_addrinfo hints, *resolved;
  std::memset(&hints, 0, sizeof(rdma_addrinfo));
  hints.ai_flags = RAI_PASSIVE;
  hints.ai_port_space = RDMA_PS_TCP;
  
  auto port_str = std::to_string(port);
  int gai_ret = rdma_getaddrinfo(address.c_str(), port_str.c_str(), &hints, &resolved);
  if (gai_ret != 0) {
    REMUS_FATAL("rdma_getaddrinfo(): {}", gai_strerror(gai_ret));
  }

  // Create endpoint with proper QP attributes
  ibv_qp_init_attr init_attr = make_default_qp_init_attrs();
  rdma_cm_id *listen_id = nullptr;
  
  auto err = rdma_create_ep(&listen_id, resolved, nullptr, &init_attr);
  rdma_freeaddrinfo(resolved);
  
  if (err != 0) {
    REMUS_FATAL("rdma_create_ep() failed for {}:{}: {}", address, port, strerror(errno));
  }

  // Create event channel and migrate
  rdma_event_channel *channel = rdma_create_event_channel();
  if (rdma_migrate_id(listen_id, channel) != 0) {
    REMUS_FATAL("rdma_migrate_id(): {}", strerror(errno));
  }
  make_nonblocking(channel->fd);
  
  if (rdma_listen(listen_id, 128) != 0) {
    REMUS_FATAL("rdma_listen() failed: {}", strerror(errno));
  }

  REMUS_INFO("Listener bound on device {} port {}", 
             listen_id->verbs->device->name, port);

  return listen_id;
}

/// RegionInfo is a helper for passing data about a MemoryNode's segments to a
/// ComputeNode.
struct RegionInfo {
  uint64_t raddr; // The base address of the segment
  uint32_t rkey;  // The rkey to use when accessing the segment
};

/// ibv_mr_deleter wraps a call to ibv_dereg_mr.  It is used by `ibv_mr_ptr`,
/// so that a unique_ptr can hold an RDMA memory region (ibv_mr*) and
/// correctly deregister
struct ibv_mr_deleter {
  void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
};
using ibv_mr_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

/// ControlBlock is a header for each Segment managed by a Memory Node.  It
/// supports bump allocation and graceful shutdown, and offers some optional
/// space for a barrier (for synchronizing compute threads) and root pointer
/// (cast to rdma_ptr<> to reach the workload's root)
struct alignas(64) ControlBlock {
  const uint64_t size_;                // The size of the segment
  std::atomic<uint64_t> allocated_;    // The number of allocated bytes
  std::atomic<uint64_t> control_flag_; // A control flag, for shutdown
  std::atomic<uint64_t> barrier_;      // An optional barrier
  std::atomic<uint64_t> root_;         // An optional root pointer

  /// Initialize a ControlBlock with the provided size
  ControlBlock(uint64_t size)
      : size_(size), allocated_(sizeof(ControlBlock)), control_flag_(0),
        barrier_(0), root_(0) {}
};

/// A good-faith re-implementation of Fraser's PRNG.  We use the same magic
/// constants, and we seed the PRNG with the result of rdtsc.
class rdtsc_rand_t {
  uint64_t seed; // The seed... should be 64 bits, even though we return 32 bits
public:
  /// Construct the PRNG by setting the seed to the value of rdtsc
  rdtsc_rand_t() : seed(__rdtsc()) {}

  /// Generate a random number and update the seed
  uint32_t rand() { return (seed = (seed * 1103515245) + 12345); }
};
} // namespace remus::internal
