#pragma once

#include <rdma/rdma_cma.h>

#include "../../vendor/sss/status.h"

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
  virtual sss::Status SendMessage(const Message &msg) = 0;
  virtual sss::StatusVal<Message> TryDeliverMessage() = 0;
};

class EmptyRdmaMessenger : public RdmaMessenger {
public:
  ~EmptyRdmaMessenger() = default;
  explicit EmptyRdmaMessenger(rdma_cm_id *id) {}
  sss::Status SendMessage(const Message &msg) override {
    return sss::Status::Ok();
  }
  sss::StatusVal<Message> TryDeliverMessage() override {
    return {{sss::Ok, {}}, Message{nullptr, 0}};
  }
};

} // namespace rome::rdma