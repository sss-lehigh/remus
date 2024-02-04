#pragma once

#include <fcntl.h>
#include <infiniband/verbs.h>
#include <optional>
#include <string>
#include <vector>

#include "rome/logging/logging.h"

namespace rome::rdma::internal {

/// Send/Recv buffers are 4 KiB
constexpr int kCapacity = 1 << 12;

/// Max # SGEs in a single RDMA write
constexpr int kMaxSge = 1;

/// We aren't using INLINE data
constexpr int kMaxInlineData = 0;

/// Max message size
constexpr int kMaxRecvBytes = 64;

/// Max # outstanding writes
constexpr int kMaxWr = kCapacity / kMaxRecvBytes;

/// Set the file descriptor `fd` as O_NONBLOCK
inline void make_nonblocking(int fd) {
  using namespace std::string_literals;
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) != 0) {
    ROME_FATAL("fcntl():"s + strerror(errno));
  }
}

/// Set the file descriptor `fd` as O_SYNC
///
/// TODO: Why do we use this?
inline void make_sync(int fd) {
  using namespace std::string_literals;
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_SYNC) != 0) {
    ROME_FATAL("fcntl():"s + strerror(errno));
  }
}

/// Produce a vector of active RDMA ports, or None if none are found
inline std::optional<std::vector<int>> FindActivePorts(ibv_context *context) {
  // Find the first active port, failing if none exists.
  ibv_device_attr dev_attr;
  ibv_query_device(context, &dev_attr);
  std::vector<int> ports;
  for (int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
    ibv_port_attr port_attr;
    ibv_query_port(context, i, &port_attr);
    if (port_attr.state != IBV_PORT_ACTIVE) {
      continue;
    } else {
      ports.push_back(i);
    }
  }
  if (ports.empty())
    return {};
  return ports;
}

/// Returns a vector of device name and active port pairs that are accessible
/// on this machine, or None if no devices are found
///
/// TODO: This function name is misleading... It is stateful, since it
///       *opens* devices.  This means that its return value doesn't tell the
///       whole story.
inline std::optional<std::vector<std::pair<std::string, int>>>
GetAvailableDevices() {
  int num_devices;
  auto **device_list = ibv_get_device_list(&num_devices);
  if (num_devices <= 0)
    return {};
  std::vector<std::pair<std::string, int>> active;
  for (int i = 0; i < num_devices; ++i) {
    auto *context = ibv_open_device(device_list[i]);
    if (context) {
      auto ports = FindActivePorts(context);
      if (!ports.has_value())
        continue;
      for (auto p : ports.value()) {
        active.emplace_back(context->device->name, p);
      }
    }
  }
  ibv_free_device_list(device_list);
  return active;
}

/// Configure the minimum attributes for a QP
inline ibv_qp_init_attr DefaultQpInitAttr() {
  ibv_qp_init_attr init_attr = {0};
  init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = kMaxWr;
  init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = kMaxSge;
  init_attr.cap.max_inline_data = kMaxInlineData;
  init_attr.sq_sig_all = 0; // Must request completions.
  init_attr.qp_type = IBV_QPT_RC;
  return init_attr;
}

} // namespace rome::rdma::internal
