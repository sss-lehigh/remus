#pragma once

#include <rdma/rdma_cma.h>

#include "../../vendor/sss/status.h"
#include "messenger.h"

namespace rome::rdma {

template <typename Messenger> class RdmaChannel : public Messenger {
public:
  ~RdmaChannel() {}
  explicit RdmaChannel(rdma_cm_id *id) : Messenger(id), id_(id) {}

  // No copy or move.
  RdmaChannel(const RdmaChannel &c) = delete;
  RdmaChannel(RdmaChannel &&c) = delete;

  // Getters.
  rdma_cm_id *id() const { return id_; }

  template <typename ProtoType> sss::Status Send(const ProtoType &proto) {
    Message msg{std::make_unique<uint8_t[]>(proto.ByteSizeLong()),
                proto.ByteSizeLong()};
    proto.SerializeToArray(msg.buffer.get(), msg.length);
    return this->SendMessage(msg);
  }

  template <typename ProtoType> sss::StatusVal<ProtoType> TryDeliver() {
    auto msg_or = this->TryDeliverMessage();
    if (msg_or.status.t == sss::Ok) {
      ProtoType proto;
      proto.ParseFromArray(msg_or.val.value().buffer.get(),
                           msg_or.val.value().length);
      return {sss::Status::Ok(), proto};
    } else {
      return {msg_or.status, {}};
    }
  }

  template <typename ProtoType> sss::StatusVal<ProtoType> Deliver() {
    auto p = this->TryDeliver<ProtoType>();
    while (p.status.t == sss::Unavailable) {
      p = this->TryDeliver<ProtoType>();
    }
    return p;
  }

  sss::Status Post(ibv_send_wr *wr, ibv_send_wr **bad) {
    return this->PostInternal(wr, bad);
  }

private:
  // A pointer to the QP used to post sends and receives.
  rdma_cm_id *id_; //! NOT OWNED
};

} // namespace rome::rdma