#pragma once

#include <cstddef>
#include <infiniband/verbs.h>
#include <optional>
#include <string>
#include <vector>

#include "../logging/logging.h"

namespace rome::rdma {

class RdmaDevice {
  static sss::StatusVal<std::vector<int>>
  FindActivePorts(ibv_context *context) {
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

    if (ports.empty()) {
      return {{sss::Unavailable, "No active ports"}, {}};
    } else {
      return {sss::Status::Ok(), ports};
    }
  }

public:
  // Returns a vector of device name and active port pairs that are accessible
  // on this machine.
  static sss::StatusVal<std::vector<std::pair<std::string, int>>>
  GetAvailableDevices() {
    int num_devices;
    auto **device_list = ibv_get_device_list(&num_devices);
    if (num_devices <= 0) {
      return {{sss::NotFound, "No devices found"}, {}};
    }
    std::vector<std::pair<std::string, int>> active;
    for (int i = 0; i < num_devices; ++i) {
      auto *context = ibv_open_device(device_list[i]);
      if (context) {
        auto ports_or = FindActivePorts(context);
        if (ports_or.status.t != sss::Ok)
          continue;
        for (auto p : ports_or.val.value()) {
          active.emplace_back(context->device->name, p);
        }
      }
    }

    ibv_free_device_list(device_list);
    return {sss::Status::Ok(), active};
  }
};

} // namespace rome::rdma