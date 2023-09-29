#pragma once

#include <rdma/rdma_verbs.h>

#include "../../logging/logging.h"
#include "../../util/status_util.h"
#include "../rdma_memory.h"
#include "../rdma_util.h"
#include "rdma_messenger.h"

namespace rome::rdma {

using ::util::InternalErrorBuilder;
using ::util::ResourceExhaustedErrorBuilder;

template <uint32_t kCapacity = 1ul << 12, uint32_t kRecvMaxBytes = 1ul << 8>
class TwoSidedRdmaMessenger : public RdmaMessenger {
  static_assert(kCapacity % 2 == 0, "Capacity must be divisible by two.");

public:
  explicit TwoSidedRdmaMessenger(rdma_cm_id *id);

  absl::Status SendMessage(const Message &msg) override;

  // Attempts to deliver a sent message by checking for completed receives and
  // then returning a `Message` containing a copy of the received buffer.
  absl::StatusOr<Message> TryDeliverMessage() override;

private:
  // Memorry region IDs.
  static constexpr char kSendId[] = "send";
  static constexpr char kRecvId[] = "recv";

  // Reset the receive buffer and post `ibv_recv_wr` on the RQ. This should only
  // be called when all posted receives have corresponding completions,
  // otherwise there may be a race on memory by posted recvs.
  void PrepareRecvBuffer();

  // The remotely accessible memory used for the send and recv buffers.
  RdmaMemory rm_;

  // A pointer to the QP used to post sends and receives.
  rdma_cm_id *id_; //! NOT OWNED

  // Pointer to memory region identified by `kSendId`.
  ibv_mr *send_mr_;

  // Size of send buffer.
  const int send_capacity_;

  // Pointer to the base address and next unposted address of send buffer.
  uint8_t *send_base_, *send_next_;

  uint32_t send_total_;

  // Pointer to memory region identified by `kRecvId`.
  ibv_mr *recv_mr_;

  // Size of the recv buffer.
  const int recv_capacity_;

  // Pointer to base address and next unposted address.
  uint8_t *recv_base_, *recv_next_;

  // Tracks the number of completed receives. This is used by
  // `PrepareRecvBuffer()` to ensure that it is only called when all posted
  // receives have been completed.
  uint32_t recv_total_;
};

} // namespace rome::rdma

#include "twosided_messenger_impl.h"