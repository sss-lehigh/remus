#pragma once

#include <rdma/rdma_cma.h>

#include <cstdint>
#include <memory>

#include "../channel/rdma_accessor.h"
#include "../channel/rdma_channel.h"
#include "../channel/twosided_messenger.h"

namespace rome::rdma {

// Contains the necessary information for communicating between nodes. This
// class wraps a unique pointer to the `rdma_cm_id` that holds the QP used for
// communication, along with the `RdmaChannel` that represents the memory used
// for 2-sided message-passing.
template <typename Channel = RdmaChannel<EmptyRdmaMessenger, EmptyRdmaAccessor>>
class Connection {
public:
  typedef Channel channel_type;

  Connection()
      : terminated_(false), src_id_(std::numeric_limits<uint32_t>::max()),
        dst_id_(std::numeric_limits<uint32_t>::max()), channel_(nullptr) {}
  Connection(uint32_t src_id, uint32_t dst_id,
             std::unique_ptr<channel_type> channel)
      : terminated_(false), src_id_(src_id), dst_id_(dst_id),
        channel_(std::move(channel)) {}

  Connection(const Connection &) = delete;
  Connection(Connection &&c)
      : terminated_(c.terminated_), src_id_(c.src_id_), dst_id_(c.dst_id_),
        channel_(std::move(c.channel_)) {}

  // Getters.
  inline bool terminated() const { return terminated_; }
  uint32_t src_id() const { return src_id_; }
  uint32_t dst_id() const { return dst_id_; }
  rdma_cm_id *id() const { return channel_->id(); }
  channel_type *channel() const { return channel_.get(); }

  void Terminate() { terminated_ = true; }

private:
  volatile bool terminated_;

  uint32_t src_id_;
  uint32_t dst_id_;

  // Remotely accessible memory that is used for 2-sided message-passing.
  std::unique_ptr<channel_type> channel_;
};

} // namespace rome::rdma