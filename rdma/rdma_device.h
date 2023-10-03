#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>

#include "../logging/logging.h"
#include "infiniband/verbs.h"
#include "rdma_util.h"

namespace rome::rdma {

// RAII.
class RdmaDevice {
public:
  // Returns a vector of device name and active port pairs that are accessible
  // on this machine.
  static sss::StatusVal<std::vector<std::pair<std::string, int>>>
  GetAvailableDevices();

  static sss::Status LookupDevice(std::string_view name);

  static std::unique_ptr<RdmaDevice> Create(std::string_view name,
                                            std::optional<int> port) {
    auto *device = new RdmaDevice();
    if (device->OpenDevice(name).t != sss::Ok) {
      return nullptr;
    }
    if (device->ResolvePort(port).t != sss::Ok) {
      return nullptr;
    }
    ROME_INFO("Created device: dev_name={}, port={}", device->name(),
              device->port());
    return std::unique_ptr<RdmaDevice>(device);
  }

  ~RdmaDevice();

  RdmaDevice(const RdmaDevice &) = delete; //! No copies
  RdmaDevice(RdmaDevice &&) = delete;      //! No moves

  // Getters.
  std::string name() { return dev_context_->device->name; }
  int port() { return port_; }

  //  Creates a new protection domain registered with the device under the given
  //  `id`. This domain can then be retrieved to allocate memory regions.
  sss::Status CreateProtectionDomain(std::string_view id);

  // Returns a pointer to the protection domain registered under `id`, or
  // returns `NotFound`.
  sss::StatusVal<ibv_pd *> GetProtectionDomain(std::string_view id);

private:
  RdmaDevice() {} // Can only be constructed through `Create`

  // Tries to open the device with name `dev_name`. Returns `NotFound` if none
  // exists.
  sss::Status OpenDevice(std::string_view dev_name);

  // Tries to resolve the given `port` on the currently open device. If
  // `nullopt` then the first available port is used. Returns `NotFound`
  //
  // Requires: that `OpenDevice` returned `Ok`.
  sss::Status ResolvePort(std::optional<int> port);

  int port_;
  ibv_context_unique_ptr dev_context_;
  std::unordered_map<std::string, ibv_pd_unique_ptr> protection_domains_;
};

} // namespace rome::rdma