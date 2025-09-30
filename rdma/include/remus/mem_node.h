#pragma once

#include <rdma/rdma_cma.h>

#include <atomic>
#include <functional>
#include <thread>

#include "cfg.h"
#include "connection.h"
#include "logging.h"
#include "rdma_ops.h"
#include "rdma_ptr.h"
#include "util.h"

namespace remus {

/// @brief A MemoryNode is a machine that provides Segments for ComputeNodes
/// @details
/// MemoryNode provides a collection of Segments that can be accessed by Compute
/// Nodes.  Each Segment must be pinned, registered with the RNIC, and assigned
/// an rkey that Compute Nodes must provide in order to operate on it.
///
/// Memory Nodes are visible to Compute Nodes, but Compute Nodes are not
/// necessarily visible to Memory Nodes, and Memory Nodes are not visible to
/// each other.  Each Memory Node will have a listening id (think "socket"), and
/// Compute Nodes can connect to that id to create their own dedicated ids for
/// interacting with the Memory Node. Each such id will support both two-sided
/// and one-sided communication.  This is necessary, because one cannot perform
/// one-sided operations without knowing an rkey, and the only way to
/// communicate an rkey is via a two-sided operation.
///
/// All Segments within a MemoryNode are associated with the same protection
/// domain (PD), because all connections derive from a single listening id, and
/// thus inherit its PD.  Despite this simplicity, a Memory Node must maintain a
/// collection of the ids that are created in response to Compute Nodes
/// connecting to its listening id, so it can shut down gracefully.
///
/// Note that Remus allows a machine to function as both a Memory Node and a
/// Compute Node.  To avoid deadlock, the act of receiving connections on the
/// listening id is performed by a separate thread, so that a co-located Compute
/// Node context can try to connect to remote machines while receiving
/// connections from those same machines.
class MemoryNode {
  // TODO: Why protected?  Do we extend MemoryNode?
 protected:
  /// @brief A context to associate with a connection
  /// @details
  /// NB: This would be more important if we handled disconnects gracefully.  As
  ///     is, it just lets us know the machine id of the remote machine when a
  ///     connection is established.
  struct IdContext {
    uint32_t machine_id_;         // The connected machine's id
    rdma_conn_param conn_param_;  // Private data to send during config

    /// Extract the machine id from a context object
    ///
    /// @param ctx TODO
    /// @return TODO
    static inline uint32_t get_machine_id(void *ctx) {
      return reinterpret_cast<IdContext *>(ctx)->machine_id_;
    }
  };

  /// @brief SegInfo is a convenience type for associating a MemoryRegistration
  /// with one of the remote-accessible Segments stored in this MemoryNode
  struct SegInfo {
    std::unique_ptr<internal::Segment> seg_;  // A Segment
    internal::ibv_mr_ptr mr_;                 // The associated mr
  };

  /// A shared pointer to a Connection object
  using ConnPtr = std::shared_ptr<internal::Connection>;

  rdma_cm_id *listen_id_ = nullptr;               // The listening endpoint
  rdma_event_channel *listen_channel_ = nullptr;  // The channel for listen_id_
  std::thread runner_;                            // The thread who listens
  std::string address_;                           // This node's address
  uint16_t port_;                                 // This node's port
  std::vector<internal::RegionInfo> ris_;         // Info about the Segments
  uint64_t remaining_conns_;                      // # conns yet to receive
  MachineInfo self_;                              // This machine's id/addr
  std::vector<ConnPtr> conns_;                    // All open connections
  std::vector<SegInfo> segs_;                     // The RDMA heaps

  /// A segment that is registered with the RNIC in a manner that allows limited
  /// send() calls.
  ///
  /// NB: This is not optimized for high-performance 2-sided operations.  Its
  ///     current use is limited to sending rkeys to ComputeNodes.  Note that it
  ///     does not require thread safety, since all Sends are currently
  ///     blocking.
  internal::Segment send_seg_;
  internal::ibv_mr_ptr mr_;  // MemoryRegistration for send_seg_

  /// The main loop run by the listening thread.  Polls for new events on the
  /// listening endpoint and handles them.  This typically means receiving new
  /// connections, configuring the new endpoints, and then putting them into
  /// conns_.
  void handle_connections() {
    REMUS_INFO("MemoryNode {} listening on {}:{}", self_.id, address_, port_);
    while (remaining_conns_ > 0) {
      // Attempt to read an event from `listen_channel_`
      rdma_cm_event *event = nullptr;
      // Handle errors.  Hopefully it's EAGAIN so we can retry
      if (int ret = rdma_get_cm_event(listen_channel_, &event); ret != 0) {
        if (errno != EAGAIN) {
          REMUS_FATAL("rdma_get_cm_event(): {}", strerror(errno));
        } else {
          // TODO: Why yield?  Is there harm in trying again immediately?  If
          // so, then why not sleep?
          std::this_thread::yield();
          continue;
        }
      }

      // Handle whatever event we just received
      rdma_cm_id *id = event->id;
      switch (event->event) {
        case RDMA_CM_EVENT_TIMEWAIT_EXIT:
          // We aren't currently getting CM_EVENT_TIMEWAIT_EXIT events, but if
          // we did, we'd probably just ACK them and continue
          rdma_ack_cm_event(event);
          break;

        case RDMA_CM_EVENT_CONNECT_REQUEST:
          // This is the main thing we expect: a request for a new connection
          on_connect(id, event, listen_id_->pd);
          break;

        case RDMA_CM_EVENT_ESTABLISHED:
          // Once the connection is fully established, we just ack it, and then
          // it's ready for use.
          rdma_ack_cm_event(event);
          break;

        case RDMA_CM_EVENT_DISCONNECTED:
          // Since we're polling on a *listening* channel, we're never going to
          // see disconnected events.  If we did, we have some code for knowing
          // what to do with them.
          rdma_ack_cm_event(event);
          on_disconnect(id);
          break;

        case RDMA_CM_EVENT_DEVICE_REMOVAL:
          // We don't expect to ever see a device removal event
          REMUS_FATAL("event: {}, error: {}\n", rdma_event_str(event->event),
                      event->status);
          break;

        case RDMA_CM_EVENT_ADDR_ERROR:
        case RDMA_CM_EVENT_ROUTE_ERROR:
        case RDMA_CM_EVENT_UNREACHABLE:
        case RDMA_CM_EVENT_ADDR_RESOLVED:
        case RDMA_CM_EVENT_REJECTED:
        case RDMA_CM_EVENT_CONNECT_ERROR:
          // These signals are sent to a connecting endpoint, so we should not
          // see them here.~ If they appear, abort.
          REMUS_FATAL("Unexpected signal: {}", rdma_event_str(event->event));

        default:
          // We did not design for other events, so crash if another event
          // arrives
          REMUS_FATAL("Not implemented: {}", rdma_event_str(event->event));
      }
    }
  }

  /// Handler to run when a connection request arrives
  void on_connect(rdma_cm_id *id, rdma_cm_event *event, ibv_pd *pd) {
    // The private data is used to figure out the node that made the request
    REMUS_ASSERT(event->param.conn.private_data != nullptr,
                 "Received connect request without private data.");
    uint32_t machine_id = *(uint32_t *)(event->param.conn.private_data);
    if (machine_id == self_.id) {
      REMUS_FATAL("on_connect called for self-connection");
    }
    // Set the QP ACK timeout before creating the QP
    uint8_t timeout = 12;  // Example timeout value
    int ret = rdma_set_option(id, RDMA_OPTION_ID, RDMA_OPTION_ID_ACK_TIMEOUT,
                              &timeout, sizeof(timeout));
    REMUS_ASSERT(ret == 0, "rdma_set_option(): {}", strerror(errno));

    // Create a new QP for the connection.
    ibv_qp_init_attr init_attr = internal::make_default_qp_init_attrs();
    ret = rdma_create_qp(id, pd, &init_attr);
    REMUS_ASSERT(ret == 0, "rdma_create_qp(): {}", strerror(errno));

    // Prepare the necessary resources for this connection.  Includes a QP and
    // memory for 2-sided communication. The underlying QP is RC, so we reuse
    // it for issuing 1-sided RDMA too. We also store the `machine_id`
    // associated with this id so that we can reference it later.
    //
    // TODO: This config should be double-checked
    auto context = new IdContext{machine_id, {}};
    context->conn_param_.private_data = &context->machine_id_;
    context->conn_param_.private_data_len = sizeof(context->machine_id_);
    context->conn_param_.rnr_retry_count = 7;  // Retry forever
    context->conn_param_.retry_count = 255;
    context->conn_param_.responder_resources = 255;
    context->conn_param_.initiator_depth = 255;
    id->context = context;
    internal::make_nonblocking(id->recv_cq->channel->fd);
    internal::make_nonblocking(id->send_cq->channel->fd);

    // Save the connection and ack it
    auto conn = new internal::Connection(self_.id, machine_id, id);
    conns_.emplace_back(conn);

    ret = rdma_accept(id,
                      machine_id == self_.id ? nullptr : &context->conn_param_);
    REMUS_ASSERT(ret == 0, "rdma_accept(): {}",
                 strerror((*__errno_location())));

    // TODO: We're not checking for errors on this ack?!?
    rdma_ack_cm_event(event);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // TODO:  Should Send have fail-stop semantics instead of returning a status
    //        that needs to be checked?
    auto status = conn->Send(ris_, send_seg_, mr_.get());
    if (status.t != remus::Ok) {
      REMUS_FATAL("error in mem node: {}", status.message.value());
    }
    remaining_conns_--;
  }

  /// This is not live code, we just have it here for reference.  If we wanted
  /// to handle dropped connections gracefully, we could do something like
  /// this (but not exactly this).
  void on_disconnect(rdma_cm_id *id) {
    // NB:  The event is already ack'ed by the caller, and the remote machine
    //      already disconnected, so we just clean up...
    //
    // TODO:  If this ever goes live, be sure to check the return value of
    //        rdma_disconnect()
    rdma_disconnect(id);
    // uint32_t node_id = IdContext::get_machine_id(id->context);
    auto *event_channel = id->channel;
    rdma_destroy_ep(id);
    rdma_destroy_event_channel(event_channel);
    // ... And don't forget to remove node_id from conns_
  }
  const uint64_t total_threads_;

 public:
  /// TODO: Need to make sure all fields get reclaimed/destructed properly
  ~MemoryNode() {
    auto cb_ptr =
        reinterpret_cast<internal::ControlBlock *>(segs_[0].seg_->raw());
    while (cb_ptr->control_flag_.load() != total_threads_) {
      std::this_thread::yield();
    }
    REMUS_INFO("MemoryNode shutdown");
    sleep(3);
  }

  /// Construct a MemoryNode
  ///
  /// TODO: The seg size shouldn't be hard-coded.  1MB is certainly more than
  ///       enough, so nothing is going to break right now, but all we really
  ///       need is ~16B more than the size of ris_
  explicit MemoryNode(const MachineInfo &self,
                      std::shared_ptr<remus::ArgMap> args)
      : self_(self),
        conns_(),
        send_seg_(1 << 20),
        total_threads_(args->uget(remus::CN_THREADS) *
                       (args->uget(remus::LAST_CN_ID) -
                        args->uget(remus::FIRST_CN_ID) + 1)) {
    int id = args->uget(remus::NODE_ID);
    uint64_t num_segs = args->uget(remus::SEGS_PER_MN);
    uint64_t seg_size_bits = args->uget(remus::SEG_SIZE);
    REMUS_INFO("Node {}: Configuring Memory Node ({} segments at 2^{}B each)",
               id, num_segs, seg_size_bits);

    // How many non-localhost compute nodes are there?
    int c0 = args->uget(remus::FIRST_CN_ID);
    int cn = args->uget(remus::LAST_CN_ID);
    uint32_t cns = cn - c0 + 1;
    if (id >= c0 && id <= cn) cns--;
    uint32_t qlw = args->uget(remus::QP_LANES);
    remaining_conns_ = cns * qlw;

    // Make the listening endpoint, and register the sending segment with it
    uint16_t port = args->uget(remus::MN_PORT);
    listen_id_ = remus::internal::make_listen_id(self.address, port);
    REMUS_ASSERT(listen_id_->pd != nullptr, "Error creating protection domain");
    mr_ = send_seg_.registerWithPd(listen_id_->pd);

    // Construct the memory pools, configure their control region, and register
    // them with the listening endpoint
    for (uint64_t i = 0; i < num_segs; ++i) {
      auto seg = std::make_unique<internal::Segment>(1ULL << seg_size_bits);
      new ((internal::ControlBlock *)(seg->raw()))
          internal::ControlBlock(1ULL << seg_size_bits);
      auto mr = seg->registerWithPd(listen_id_->pd);
      segs_.emplace_back(SegInfo{std::move(seg), std::move(mr)});
    }

    // Prepare the RegionInfo for this node
    //
    // NB:  We could be more efficient if we constructed the ris in a segment,
    //      but we're not concerned about the performance of Send(), since we
    //      only use it for network set-up.
    for (uint64_t i = 0; i < num_segs; ++i) {
      ris_.emplace_back((uintptr_t)segs_[i].seg_->raw(), segs_[i].mr_->rkey);
    }
    REMUS_INFO("Shared Segments:");
    for (auto ri : ris_) {
      REMUS_INFO("  0x{:x} (rk=0x{:x})", ri.raddr, ri.rkey);
    }

    // Initialize a listening endpoint, and then park a thread on it, so the
    // thread can receive new connection requests
    //
    // NB: Since we are using an async socket, we call `listen()` here, but it
    //     won't block or run any code for handling a listening request.  We'll
    //     do that later, in a separate thread.
    REMUS_INFO("Listener thread awaiting {} connections", remaining_conns_);
    listen_channel_ = rdma_create_event_channel();
    if (rdma_migrate_id(listen_id_, listen_channel_) != 0) {
      REMUS_FATAL("rdma_migrate_id(): {}", strerror(errno));
    }
    internal::make_nonblocking(listen_id_->channel->fd);
    if (rdma_listen(listen_id_, 0) != 0) {
      REMUS_FATAL("rdma_listen(): {}", strerror(errno));
    }
    address_ = std::string(inet_ntoa(
        reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(listen_id_))
            ->sin_addr));
    port_ = rdma_get_src_port(listen_id_);
    runner_ = std::thread([&]() { handle_connections(); });
  }

  /// Return a vector with the RegionInfo for this MemoryNode.  This is a
  /// special-purpose function, used only for the situation where a ComputeNode
  /// is also a MemoryNode.  In that case, the ComputeNode can't its own rkeys
  /// for use over the loopback in the standard way (i.e., it can't connect to
  /// itself and do a Send/Recv).
  std::vector<internal::RegionInfo> get_local_rkeys() {
    std::vector<internal::RegionInfo> res;
    for (auto &p : segs_)
      res.push_back(
          internal::RegionInfo{(uintptr_t)p.seg_->raw(), p.mr_->rkey});
    return res;
  }

  /// Stop listening for new connections, and terminate the listening thread
  ///
  /// NB: This blocks the caller until the listening thread has been joined
  void init_done() {
    REMUS_INFO("Stopping listening thread...");
    runner_.join();
    rdma_destroy_ep(listen_id_);
    // TODO: if not sleep, other compute threads can't write to it, need figure
    // out why
    sleep(1);
    // TODO:  If the barrier also had a "half barrier", where threads could
    //        increment but not wait, then we could use it to ACK when this can
    //        finally stop
  }
};
}  // namespace remus