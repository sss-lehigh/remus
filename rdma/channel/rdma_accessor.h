#pragma once

#include <rdma/rdma_cma.h>

#include "absl/status/status.h"

namespace rome::rdma {

class RdmaAccessor {
 public:
  virtual ~RdmaAccessor() = default;
  virtual absl::Status PostInternal(ibv_send_wr* sge, ibv_send_wr** bad) = 0;
};

class EmptyRdmaAccessor : public RdmaAccessor {
 public:
  ~EmptyRdmaAccessor() = default;
  explicit EmptyRdmaAccessor(rdma_cm_id* id) {}
  absl::Status PostInternal(ibv_send_wr* sge, ibv_send_wr** bad) override {
    return absl::OkStatus();
  }
};

}  // namespace rome