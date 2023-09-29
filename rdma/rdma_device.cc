#include "rdma_device.h"

#include <infiniband/verbs.h>

#include <optional>
#include <vector>

#include "../logging/logging.h"
#include "../util/status_util.h"

namespace rome::rdma {

using ::util::AlreadyExistsErrorBuilder;
using ::util::NotFoundErrorBuilder;
using ::util::UnavailableErrorBuilder;
using ::util::UnknownErrorBuilder;

namespace {

bool IsActivePort(const ibv_port_attr &port_attr) {
  return port_attr.state == IBV_PORT_ACTIVE;
}

absl::StatusOr<std::vector<int>> FindActivePorts(ibv_context *context) {
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
    return absl::UnavailableError("No active ports");
  } else {
    return ports;
  }
}

} // namespace

/* static */ absl::StatusOr<std::vector<std::pair<std::string, int>>>
RdmaDevice::GetAvailableDevices() {
  int num_devices;
  auto **device_list = ibv_get_device_list(&num_devices);
  ROME_CHECK_QUIET(ROME_RETURN(absl::NotFoundError("No devices found")),
                   num_devices > 0);
  std::vector<std::pair<std::string, int>> active;
  for (int i = 0; i < num_devices; ++i) {
    auto *context = ibv_open_device(device_list[i]);
    if (context) {
      auto ports_or = FindActivePorts(context);
      if (!ports_or.ok())
        continue;
      for (auto p : ports_or.value()) {
        active.emplace_back(context->device->name, p);
      }
    }
  }

  ibv_free_device_list(device_list);
  return active;
}

/* static */ absl::Status RdmaDevice::LookupDevice(std::string_view name) {
  auto devices_or = GetAvailableDevices();
  if (!devices_or.ok())
    return devices_or.status();
  for (const auto &d : devices_or.value()) {
    if (name == d.first)
      return absl::OkStatus();
  }
  return util::NotFoundErrorBuilder() << "Device not found: " << name;
}

RdmaDevice::~RdmaDevice() { protection_domains_.clear(); }

absl::Status RdmaDevice::OpenDevice(std::string_view dev_name) {
  int num_devices;
  auto device_list =
      ibv_device_list_unique_ptr(ibv_get_device_list(&num_devices));
  ROME_CHECK_QUIET(
      ROME_RETURN(UnavailableErrorBuilder() << "No available devices"),
      num_devices > 0 && device_list != nullptr);

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
  ROME_CHECK_QUIET(
      ROME_RETURN(NotFoundErrorBuilder() << "Device not found: " << dev_name),
      found != nullptr);

  // Try opening the device on the provided port, or the first available.
  dev_context_ = ibv_context_unique_ptr(ibv_open_device(found));
  ROME_CHECK_QUIET(ROME_RETURN(UnavailableErrorBuilder()
                               << "Could not open device:" << dev_name),
                   dev_context_ != nullptr);
  return absl::OkStatus();
}

absl::Status RdmaDevice::ResolvePort(std::optional<int> port) {
  auto ports_or = FindActivePorts(dev_context_.get());
  ROME_CHECK_QUIET(ROME_RETURN(UnavailableErrorBuilder()
                               << "No active ports:"
                               << dev_context_->device->name),
                   ports_or.ok());
  auto ports = ports_or.value();
  if (port.has_value()) {
    for (auto p : ports) {
      if (p == port.value()) {
        port_ = p;
        return absl::OkStatus();
      }
    }
    return UnavailableErrorBuilder() << "Port not active: " << port.value();
  } else {
    port_ = ports[0]; // Use the first active port
    return absl::OkStatus();
  }
}

absl::Status RdmaDevice::CreateProtectionDomain(std::string_view id) {
  if (protection_domains_.find(std::string(id)) != protection_domains_.end()) {
    return AlreadyExistsErrorBuilder() << "PD already exists: {}" << id;
  }
  auto pd = ibv_pd_unique_ptr(ibv_alloc_pd(dev_context_.get()));
  if (pd == nullptr) {
    return UnknownErrorBuilder() << "Failed to allocated PD: {}" << id;
  }
  protection_domains_.emplace(id, std::move(pd));
  return absl::OkStatus();
}

absl::StatusOr<ibv_pd *> RdmaDevice::GetProtectionDomain(std::string_view id) {
  auto iter = protection_domains_.find(std::string(id));
  if (iter == protection_domains_.end()) {
    return NotFoundErrorBuilder() << "PD not found: {}" << id;
  }
  return iter->second.get();
}

} // namespace rome::rdma