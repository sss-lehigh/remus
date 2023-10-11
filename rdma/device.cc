#include "device.h"

#include <infiniband/verbs.h>

#include <optional>
#include <vector>

#include "../logging/logging.h"
#include "../util/status_util.h"

namespace rome::rdma {

namespace {

bool IsActivePort(const ibv_port_attr &port_attr) {
  return port_attr.state == IBV_PORT_ACTIVE;
}

sss::StatusVal<std::vector<int>> FindActivePorts(ibv_context *context) {
  // Find the first active port, failing if none exists.
  ibv_device_attr dev_attr;
  ibv_query_device(context, &dev_attr);
  std::vector<int> ports;
  for (int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
    ibv_port_attr port_attr;
    ibv_query_port(context, i, &port_attr);
    if (!IsActivePort(port_attr)) {
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

} // namespace

/* static */ sss::StatusVal<std::vector<std::pair<std::string, int>>>
RdmaDevice::GetAvailableDevices() {
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

/* static */ sss::Status RdmaDevice::LookupDevice(std::string_view name) {
  auto devices_or = GetAvailableDevices();
  if (devices_or.status.t != sss::Ok)
    return devices_or.status;
  for (const auto &d : devices_or.val.value()) {
    if (name == d.first)
      return sss::Status::Ok();
  }
  sss::Status err = {sss::NotFound, "Device not found: "};
  err << name;
  return err;
}

RdmaDevice::~RdmaDevice() { protection_domains_.clear(); }

sss::Status RdmaDevice::OpenDevice(std::string_view dev_name) {
  int num_devices;
  auto device_list =
      ibv_device_list_unique_ptr(ibv_get_device_list(&num_devices));
  if (num_devices <= 0 || device_list == nullptr) {
    return {sss::Unavailable, "No available devices"};
  }

  // Find device called `dev_name`
  ibv_device *found = nullptr;
  for (int i = 0; i < num_devices; ++i) {
    auto *dev = device_list.get()[i];
    if (dev->name == dev_name) {
      ROME_DEBUG("Device found: {}", dev->name);
      found = dev;
      continue;
    }
  }

  if (found == nullptr) {
    sss::Status err = {sss::NotFound, "Device not found: "};
    err << dev_name;
    return err;
  }

  // Try opening the device on the provided port, or the first available.
  dev_context_ = ibv_context_unique_ptr(ibv_open_device(found));
  if (dev_context_ == nullptr) {
    sss::Status err = {sss::Unavailable, "Could not open device: "};
    err << dev_name;
    return err;
  }
  return sss::Status::Ok();
}

sss::Status RdmaDevice::ResolvePort(std::optional<int> port) {
  auto ports_or = FindActivePorts(dev_context_.get());
  if (ports_or.status.t != sss::Ok) {
    sss::Status err = {sss::Unavailable, "No active ports:"};
    err << dev_context_->device->name;
    return err;
  }
  auto ports = ports_or.val.value();
  if (port.has_value()) {
    for (auto p : ports) {
      if (p == port.value()) {
        port_ = p;
        return sss::Status::Ok();
      }
    }
    sss::Status err = {sss::Unavailable, "Port not active: "};
    err << port.value();
    return err;
  } else {
    port_ = ports[0]; // Use the first active port
    return sss::Status::Ok();
  }
}

sss::Status RdmaDevice::CreateProtectionDomain(std::string_view id) {
  if (protection_domains_.find(std::string(id)) != protection_domains_.end()) {
    sss::Status err = {sss::AlreadyExists, "PD already exists: "};
    err << id;
    return err;
  }
  auto pd = ibv_pd_unique_ptr(ibv_alloc_pd(dev_context_.get()));
  if (pd == nullptr) {
    sss::Status err = {sss::Unknown, "Failed to allocated PD: "};
    err << id;
    return err;
  }
  protection_domains_.emplace(id, std::move(pd));
  return sss::Status::Ok();
}

sss::StatusVal<ibv_pd *> RdmaDevice::GetProtectionDomain(std::string_view id) {
  auto iter = protection_domains_.find(std::string(id));
  if (iter == protection_domains_.end()) {
    sss::Status err = {sss::NotFound, "PD not found: "};
    err << id;
    return {err, {}};
  }
  return {sss::Status::Ok(), iter->second.get()};
}

} // namespace rome::rdma