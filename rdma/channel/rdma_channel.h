#pragma once

#include <asm-generic/errno-base.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include <cstddef>
#include <limits>
#include <type_traits>

#include "../../util/status_util.h"
#include "../rdma_memory.h"
#include "absl/status/status.h"
#include "rdma_messenger.h"

namespace rome::rdma {

template <typename Messenger, typename Accessor>
class RdmaChannel : public Messenger, Accessor {
public:
  ~RdmaChannel() {}
  explicit RdmaChannel(rdma_cm_id *id) : Messenger(id), Accessor(id), id_(id) {}

  // No copy or move.
  RdmaChannel(const RdmaChannel &c) = delete;
  RdmaChannel(RdmaChannel &&c) = delete;

  // Getters.
  rdma_cm_id *id() const { return id_; }

  template <typename ProtoType> absl::Status Send(const ProtoType &proto) {
    Message msg{std::make_unique<uint8_t[]>(proto.ByteSizeLong()),
                proto.ByteSizeLong()};
    proto.SerializeToArray(msg.buffer.get(), msg.length);
    return this->SendMessage(msg);
  }

  template <typename ProtoType> absl::StatusOr<ProtoType> TryDeliver() {
    absl::StatusOr<Message> msg_or = this->TryDeliverMessage();
    if (msg_or.ok()) {
      ProtoType proto;
      proto.ParseFromArray(msg_or->buffer.get(), msg_or->length);
      return proto;
    } else {
      return msg_or.status();
    }
  }

  template <typename ProtoType> absl::StatusOr<ProtoType> Deliver() {
    auto p = this->TryDeliver<ProtoType>();
    while (p.status().code() == absl::StatusCode::kUnavailable) {
      p = this->TryDeliver<ProtoType>();
    }
    return p;
  }

  absl::Status Post(ibv_send_wr *wr, ibv_send_wr **bad) {
    return this->PostInternal(wr, bad);
  }

private:
  // A pointer to the QP used to post sends and receives.
  rdma_cm_id *id_; //! NOT OWNED
};

} // namespace rome::rdma