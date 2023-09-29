#pragma once

#include <rdma/rdma_cma.h>

#include "absl/status/statusor.h"

namespace rome::rdma {

// Interface used by `RdmaBroker` to handle connection requests. Each of the
// respective callbacks receives a pointer to the `rdma_cm_id` associated with
// the event and a pointer to the `rdma_cm_event` itself. This allows the
// callback to observe all of the necessary information associated with the
// signal. For example, the event contains connection paramaters that include
// private data during a connection request.
class RdmaReceiverInterface {
 public:
  virtual ~RdmaReceiverInterface() = default;

  // Prepares the new connection for the `rmda_accept` call. Receives `id`,
  // which is the new connection allocated upon receiving the connection
  // request. The implementation should then create or assign a QP to `id`.
  // TODO: Fix comment ^
  // TODO: https://github.com/jacnel/rome/issues/7
  virtual void OnConnectRequest(rdma_cm_id* id, rdma_cm_event* event) = 0;

  // The following are called whenever their respective signals are received by
  // an `RdmaBroker` object. Unlike `OnConnectRequest` there are no requirements
  // for the implementation. However, the `id` is destroyed after `OnDisconnect`
  // is called, and therefore the implementation should ensure that the endpoint
  // is not used after `OnDisconnect` is called.
  virtual void OnEstablished(rdma_cm_id* id, rdma_cm_event* event) = 0;
  virtual void OnDisconnect(rdma_cm_id* id) = 0;
};

}  // namespace rome::rdma