#pragma once

#include <rdma/rdma_cma.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace rome::rdma {

// Contains a copy of a received message by a `RdmaChannel` for the caller of
// `RdmaChannel::TryDeliver`.
struct Message {
  std::unique_ptr<uint8_t[]> buffer;
  size_t length;
};

class RdmaMessenger {
 public:
  virtual ~RdmaMessenger() = default;
  virtual absl::Status SendMessage(const Message& msg) = 0;
  virtual absl::StatusOr<Message> TryDeliverMessage() = 0;
};

class EmptyRdmaMessenger : public RdmaMessenger {
 public:
  ~EmptyRdmaMessenger() = default;
  explicit EmptyRdmaMessenger(rdma_cm_id* id) {}
  absl::Status SendMessage(const Message& msg) override {
    return absl::OkStatus();
  }
  absl::StatusOr<Message> TryDeliverMessage() override {
    return Message{nullptr, 0};
  }
};

}  // namespace rome