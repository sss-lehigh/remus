#pragma once

#include <infiniband/verbs.h>

#include "../rdma/channel/rdma_accessor.h"
#include "../rdma/rdma_util.h"

namespace rome::rdma {

class SyncRdmaAccessor : public RdmaAccessor {
public:
  ~SyncRdmaAccessor() = default;
  explicit SyncRdmaAccessor(rdma_cm_id *id);
  absl::Status PostInternal(ibv_send_wr *wr, ibv_send_wr **bad) override;

private:
  rdma_cm_id *id_; //! NOT OWNED
};

} // namespace rome::rdma