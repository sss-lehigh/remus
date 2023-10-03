#pragma once

#include <rdma/rdma_cma.h>

#include "../../vendor/sss/status.h"

namespace rome::rdma {

class RdmaAccessor {
public:
  virtual ~RdmaAccessor() = default;
  virtual sss::Status PostInternal(ibv_send_wr *sge, ibv_send_wr **bad) = 0;
};

// [mfs] Do we ever use this?
class EmptyRdmaAccessor : public RdmaAccessor {
public:
  ~EmptyRdmaAccessor() = default;
  explicit EmptyRdmaAccessor(rdma_cm_id *id) {}
  sss::Status PostInternal(ibv_send_wr *sge, ibv_send_wr **bad) override {
    return sss::Status::Ok();
  }
};

} // namespace rome::rdma