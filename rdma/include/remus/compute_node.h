#pragma once

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <functional>
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cfg.h"
#include "cli.h"
#include "connection.h"
#include "logging.h"
#include "rdma_ops.h"
#include "ring.h"
#include "util.h"

namespace remus::internal {

/// Minimum microseconds for exponential backoff
constexpr uint32_t connect_backoff_min_us = 100;

/// Maximum microseconds for exponential backoff
constexpr uint32_t connect_backoff_max_us = 5000000;

/// Common code for creating and initializing an endpoint
///
/// TODO: This documentation should give an intuition about the way we configure
///       endpoints (i.e., any non-hard-coded config stuff)
///
/// @param address  The address that will be connected to
/// @param port     The port to connect to
///
/// @return A connection/id that has been configured properly
inline rdma_cm_id *initialize_ep(std::string_view address, uint16_t port) {
  // Compute the info for the node we're connecting to
  auto port_str = std::to_string(htons(port));
  rdma_addrinfo hints, *resolved = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  struct sockaddr_in src;
  std::memset(&src, 0, sizeof(src));
  hints.ai_port_space = RDMA_PS_TCP;
  hints.ai_qp_type = IBV_QPT_RC;
  hints.ai_family = AF_IB;
  hints.ai_src_len = sizeof(src);
  src.sin_family = AF_INET;
  // inet_aton(address_.data(), &src.sin_addr);
  hints.ai_src_addr = reinterpret_cast<sockaddr *>(&src);
  if (int err =
          rdma_getaddrinfo(address.data(), port_str.data(), &hints, &resolved);
      err != 0) {
    REMUS_FATAL("rdma_getaddrinfo(): {}", gai_strerror(err));
  }

  // Start making a connection
  ibv_qp_init_attr init_attr = make_default_qp_init_attrs();
  rdma_cm_id *id = nullptr;
  auto err = rdma_create_ep(&id, resolved, nullptr, &init_attr);
  rdma_freeaddrinfo(resolved);
  if (err) {
    REMUS_FATAL("compute node rdma_create_ep(): {}", strerror(errno));
  }
  return id;
}

/// Connect to a remote memory node.  It is an error to use this to create a
/// Loopback connection.  Terminates the program on any error.
///
/// @param my_id    The id of this (compute) node
/// @param mn_id    The id of the memory node
/// @param mn_addr  The address of the memory node
/// @param port     The port to connect to
/// @param seg      TODO: Document this
/// @param mrs      TODO: Document this
///
/// @return A connection object for the new connection
inline Connection *connect_remote(uint32_t my_id, uint32_t mn_id,
                                  std::string_view mn_addr, uint16_t port,
                                  internal::Segment &seg,
                                  std::vector<internal::ibv_mr_ptr> &mrs) {
  uint32_t backoff_us_ = 0;
  while (true) {
    // TODO: Need a one-line comment here explaining this block of code
    rdma_cm_id *id = initialize_ep(mn_addr, port);
    auto mr = seg.registerWithPd(id->pd);
    RDMA_CM_ASSERT(rdma_post_recv, id, nullptr, seg.raw(), seg.capacity(),
                   mr.get());
    mrs.push_back(std::move(mr));

    // Migrate the new endpoint to a nonblocking event channel and do more
    // config
    auto *event_channel = rdma_create_event_channel();
    make_nonblocking(event_channel->fd);
    if (rdma_migrate_id(id, event_channel) != 0) {
      REMUS_FATAL("rdma_migrate_id(): {}", strerror(errno));
    }

    // Set the QP ACK timeout
    uint8_t timeout = 12; // Example timeout value
    if (rdma_set_option(id, RDMA_OPTION_ID, RDMA_OPTION_ID_ACK_TIMEOUT,
                        &timeout, sizeof(timeout)) != 0) {
      REMUS_FATAL("rdma_set_option(): {}", strerror(errno));
    }

    // TODO: These values feel like they need more documentation
    rdma_conn_param conn_param;
    std::memset(&conn_param, 0, sizeof(conn_param));
    conn_param.private_data = &my_id;
    conn_param.private_data_len = sizeof(my_id);
    conn_param.retry_count = 255;
    conn_param.rnr_retry_count = 7;
    conn_param.responder_resources = 8;
    conn_param.initiator_depth = 8;
    if (rdma_connect(id, &conn_param) != 0) {
      REMUS_FATAL("rdma_connect(): {}", strerror(errno));
    }

    // It takes a few events before the channel is ready to use, so we need an
    // event loop here, to receive events and move through a set of transitions
    while (true) {
      // Poll until we get an event
      rdma_cm_event *event;
      auto result = rdma_get_cm_event(id->channel, &event);
      while (result < 0 && errno == EAGAIN) {
        result = rdma_get_cm_event(id->channel, &event);
      }

      // Save the event and ack it
      auto cm_event = event->event;
      if (rdma_ack_cm_event(event) != 0) {
        REMUS_FATAL("rdma_ack_cm_event(): {}", strerror(errno));
      }

      // On an "established" event, we can make and save the connection
      if (cm_event == RDMA_CM_EVENT_ESTABLISHED) {
        make_sync(event_channel->fd);
        make_nonblocking(id->recv_cq->channel->fd);
        make_nonblocking(id->send_cq->channel->fd);

        // Make and return the connection
        return new Connection(my_id, mn_id, id);
      }

      // On an ADDR_RESOLVED, we just ack (which we did above)
      else if (cm_event == RDMA_CM_EVENT_ADDR_RESOLVED) {
      }

      // If we get a REJECTED, cleanup, backoff, try it all again
      else if (cm_event == RDMA_CM_EVENT_REJECTED) {
        rdma_destroy_ep(id);
        rdma_destroy_event_channel(event_channel);
        backoff_us_ = backoff_us_ > 0
                          ? std::min((backoff_us_ + (100 * my_id)) * 2,
                                     connect_backoff_max_us)
                          : connect_backoff_min_us;
        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us_));
        break;
      }

      // Otherwise fail
      else {
        REMUS_FATAL("Got unexpected event: {}", rdma_event_str(cm_event));
      }
    }
  }
}

/// Create a connection to the local device.  It is an error to use this to
/// create a Remote connection.  Terminates the program on any error.
///
/// @param my_id    The id of this (compute) node
/// @param address  The address of this (also memory) node
/// @param port     The port to connect to
///
/// @return A connection object for the new connection
inline Connection *connect_loopback(uint32_t my_id, std::string_view address,
                                    uint16_t port) {
  // Do the initial endpoint configuration
  rdma_cm_id *id = initialize_ep(address, port);

  // Query the ports to find one that is available and appropriate
  ibv_device_attr dev_attr;
  if (ibv_query_device(id->verbs, &dev_attr) != 0) {
    REMUS_FATAL("ibv_query_device(): {}", strerror(errno));
  }

  // There are a whole bunch of funny rules and configuration requirements when
  // connecting via loopback, especially if it's not an IB device.
  //
  // TODO:  Do we want to drop the non-IB code paths here, since they're
  //        untested?  Or maybe factor them out into a helper method that can be
  //        documented with a big "UNTESTED" warning?
  ibv_port_attr port_attr;
  uint32_t LOOPBACK_PORT_NUM = 1;

  // use first port that is active for loopback
  for (int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
    if (ibv_query_port(id->verbs, i, &port_attr) != 0) {
      REMUS_FATAL("ibv_query_port(): {}", strerror(errno));
    }
    if (port_attr.state == IBV_PORT_ACTIVE) {
      LOOPBACK_PORT_NUM = i;
      // REMUS_DEBUG("Using physical port {} for loopback", i);
      break;
    }
  }

  // TODO:  These values feel like they need more documentation.  Also, we're
  //        configuring loopback differently (via ibv_qp_atttr) than we
  //        configure remote (via rdma_conn_param).  Need to verify that the two
  //        techniques result in the same behaviors and performance.  I'm
  //        especially confused because we don't do the RoCE or GRH stuff for
  //        Remote connections... why not?
  ibv_qp_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  attr.max_dest_rd_atomic = 8;
  attr.path_mtu = IBV_MTU_4096;
  attr.min_rnr_timer = 12;
  attr.rq_psn = 0;
  attr.sq_psn = 0;
  attr.timeout = 12;
  attr.retry_cnt = 255;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 8;
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = LOOPBACK_PORT_NUM;
  int attr_mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
    REMUS_FATAL("ibv_modify_qp(): {}", strerror(errno));
  }
  attr.ah_attr.dlid = port_attr.lid;
  attr.ah_attr.port_num = LOOPBACK_PORT_NUM;

  // TODO: This documentation isn't clear to a non-expert.  What are LID, GRH?
  //
  // This LID is invalid and likely RoCE, so this is a hack to get around that
  // or the GRH is required regardless
  if (port_attr.lid == 0x0 || (port_attr.flags & IBV_QPF_GRH_REQUIRED) != 0) {
    // Our address handle has a global route
    attr.ah_attr.is_global = 1;

    // We query the first GID, which should always exist There may be others,
    // but that shouldn't impact anything for us
    //
    // We can go from gid = 0 to gid = port_attr.gid_table_len - 1
    REMUS_ASSERT(port_attr.gid_tbl_len >= 1,
                 "Need a gid table that has at least one entry");
    ibv_gid gid;
    if (ibv_query_gid(id->verbs, LOOPBACK_PORT_NUM, 0, &gid)) {
      REMUS_FATAL("Fail on query gid");
    }

    // Set our gid to the gid index we queried
    attr.ah_attr.grh.dgid = gid;
    attr.ah_attr.grh.sgid_index = 0;
    // allow for the max number of hops
    attr.ah_attr.grh.hop_limit = 0xFF;
    attr.ah_attr.grh.traffic_class = 0; // some traffic class
    // non-zero is support to give a hint to switches
    // but we dont care; this is loopback
    attr.ah_attr.grh.flow_label = 0;
  }

  // TODO: Document this chunk of config
  attr.qp_state = IBV_QPS_RTR;
  attr.dest_qp_num = id->qp->qp_num;
  attr_mask =
      (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
       IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
    REMUS_FATAL("ibv_modify_qp(): {}", strerror(errno));
  }
  attr.qp_state = IBV_QPS_RTS;
  attr_mask = (IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
               IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
    REMUS_FATAL("ibv_modify_qp(): {}", strerror(errno));
  }
  make_nonblocking(id->recv_cq->channel->fd);
  make_nonblocking(id->send_cq->channel->fd);

  // Make and return the connection
  return new Connection(my_id, my_id, id);
}
} // namespace remus::internal

namespace remus {

/// @brief A ComputeNode is a machine that can run ComputeThread (s)
/// @details
/// Everything necessary for a machine to serve in the ComputeNode role
///
/// A ComputeNode has connections to all of the MemoryNodes in the system.  It
/// also knows about all of the Segments at each node.
///
/// Configuring the ComputeNode entails reaching out and connecting to every
/// MemoryNode, and getting all of the needed information from it.  There's a
/// difficulty though: Eventually, there will be ComputeThreads, and those
/// ComputeThreads are going to need to have Segments for their one-sided
/// operations.  To avoid registering lots of segments, ComputeNode makes a big
/// one, then chops it up for its ComputeThreads.  This lets remus only make one
/// registration per connection, while still supporting many threads.
class ComputeNode {
  /// A connection to a remote node, and an lkey that enables using the local
  /// Segment to do one-sided operations across that connection.
  struct conn_info {
    std::unique_ptr<internal::Connection> conn_; // The open connection
    uint32_t lkey_;                              // The lkey
  };

  const MachineInfo self_;                // This node's id and address
  std::vector<internal::ibv_mr_ptr> mrs_; // MRs for seg_
  const uint64_t num_threads_;            // Number of threads to support
  const uint64_t thread_bufsz_;           // Segment size for each thread
  internal::Segment seg_;                 // The segment shared by the threads
  std::atomic<uint64_t> threads_;         // Number of registered threads

  using conn_map = std::unordered_map<uint16_t, std::vector<conn_info>>;
  using rkey_map = std::unordered_map<uint64_t, uint32_t>;

  /// A map of all of the connections we have for each node
  ///
  /// TODO: Since connections are only to memory nodes, and memory node indices
  ///       start at 0 and are contiguous, this could just be a vector of
  ///       vectors.  Is that worth it, or should we leave "good enough" alone?
  conn_map node_connections_;

  /// A map for getting the rkey for any node/segment combination
  ///
  /// NB: The key for this map is a packed struct, with 16 bits for the node id
  ///     some number of low bits as the segment offset, and the rest of the
  ///     bits as the segment id.
  rkey_map segment_rkeys_;

  const uint64_t seg_mask_; // The bitmask for finding a segment id

  /// A description of a Segment, suitable for allocation
  struct seg_t {
    uint64_t start_;             // The (node id | base address) of the Segment
    std::atomic<uint64_t> hint_; // The observed last value of the bump counter

    /// Construct from parts
    ///
    /// @param start  TODO
    /// @param hint   TODO
    seg_t(uint64_t start, uint64_t hint) : start_(start), hint_(hint) {}

    /// Copy constructor, so we can put these in a vector
    ///
    /// @param rhs  The seg_t to copy
    seg_t(const seg_t &rhs) : start_(rhs.start_), hint_(rhs.hint_.load()) {}
  };

  /// A map from a MemoryNode id to all of the Segments at that MemoryNode. Each
  /// Segment is paired with this node's last observed value of the segment's
  /// allocated_ field, so that allocations have a higher likelihood of their
  /// FAA succeeding on the first try.
  ///
  /// TODO: key should be uint16_t?
  ///
  /// TODO: Why not a vector instead of a map?
  std::unordered_map<uint64_t, std::vector<seg_t>> segs_;

  std::shared_ptr<remus::ArgMap> args_; // The program's command-line args

  /// Save the connection to node_id, which has registered all ComputeThread
  /// Segments to use lkey.
  ///
  /// @param node_id  TODO
  /// @param conn     TODO
  /// @param lkey     TODO
  void save_conn(uint16_t node_id, internal::Connection *conn, uint32_t lkey) {
    if (node_connections_.find(node_id) == node_connections_.end())
      node_connections_.insert({node_id, std::vector<conn_info>()});
    node_connections_[node_id].emplace_back(
        conn_info{std::unique_ptr<internal::Connection>(conn), lkey});
  }

  /// Save the rkey for a given node/region pair
  ///
  /// @param node_id  TODO
  /// @param region   TODO
  /// @param rkey     TODO
  void save_region(uint16_t node_id, uint64_t region, uint32_t rkey) {
    constexpr uint64_t node_mask = 0xFFFFULL << 48;
    REMUS_ASSERT((region & node_mask) == 0, "Top bits of region must be 0");
    // TODO:  Should we also assert that the low bits are 0?  We expect
    //        alignment, after all...
    REMUS_ASSERT((region & seg_mask_) == 0,
                 "Region is not aligned to segment size");
    uint64_t n = node_id;
    n = n << 48;
    uint64_t key = n | region;
    // Only save it once, because it's per-node, not per-connection
    if (segment_rkeys_.find(key) == segment_rkeys_.end()) {
      REMUS_INFO("  Received segment 0x{:x} from node {} with rkey {}", region,
                 node_id, rkey);
      segment_rkeys_.insert({key, rkey});
      // save this region to the seg_ map, so we can allocate out of it
      //
      // NB:  Rather than do an rdma_read, we assume that nothing has been
      //      allocated yet, and thus the bump allocator FAA hint should be the
      //      tail of the segment's metadata.
      segs_[node_id].emplace_back(key, sizeof(internal::ControlBlock));
    }
  }

public:
  /// Op counters for each lane
  ///
  /// TODO: This documentation isn't descriptive.  What's this really for?
  std::vector<std::atomic<size_t>> lane_op_counters_;

  /// Return a connection and lkey for interacting with an rdma_ptr
  ///
  /// @param ptr_raw  TODO
  /// @param idx      TODO
  /// @return TODO
  conn_info &get_conn(uint64_t ptr_raw, uint64_t idx) {
    uint64_t node_id = ptr_raw >> 48 & 0xFFFF;
    return node_connections_[node_id].at(idx);
  }

  /// Return the rkey for a node/segment pair
  ///
  /// @param raw TODO
  /// @return TODO
  uint32_t get_rkey(uint64_t raw) {
    // TODO:  This is a non-constexpr const.  Can we make it a field, to avoid
    //        recomputing?  Or is the speed savings not worth it?
    auto mask_low = (-1ULL) ^ seg_mask_;
    return segment_rkeys_.find(raw & mask_low)->second;
  }

  /// Construct a ComputeNode
  ///
  /// @param self This process's machine Id and DNS address
  /// @param args The command-line arguments to the program
  ComputeNode(const MachineInfo &self, std::shared_ptr<remus::ArgMap> args)
      : self_(self), num_threads_(args->uget(remus::CN_THREADS)),
        thread_bufsz_(1ULL << args->uget(remus::CN_THREAD_BUFSZ)),
        seg_(
            (1ULL << (64 - __builtin_clzll(num_threads_ * thread_bufsz_ - 1)))),
        threads_(0), seg_mask_((1ULL << args->uget(remus::SEG_SIZE)) - 1),
        args_(args), lane_op_counters_(args->uget(remus::QP_LANES)) {
    REMUS_INFO("Node {}: Configuring Compute Node", args->uget(remus::NODE_ID));
    // Initialize the seg map
    uint64_t m0 = args->uget(remus::FIRST_MN_ID);
    uint64_t mn = args->uget(remus::LAST_MN_ID);
    for (uint64_t i = m0; i <= mn; ++i) {
      segs_[i] = std::vector<seg_t>();
    }
  }

  /// TODO: Do we need a proper dtor, or is connection map cleanup automatic?
  ~ComputeNode() = default;

  /// Create QPs to the localhost.  This should only be used when a ComputeNode
  /// is also a MemoryNode.  It requires the MemoryNode to provide its rkeys.
  ///
  /// @param memnodes     TODO
  /// @param local_rkeys  TODO
  void connect_local(std::vector<MachineInfo> &memnodes,
                     std::vector<internal::RegionInfo> local_rkeys) {
    uint64_t qp_lanes = args_->uget(remus::QP_LANES);
    uint32_t port = args_->uget(remus::MN_PORT);

    for (auto &p : memnodes) {
      if (p.id == self_.id) {
        for (uint64_t i = 0; i < qp_lanes; ++i) {
          // Connect, then register the big segment with that connection
          REMUS_INFO("Connecting to localhost {}:{} (id = {})", p.address, port,
                     p.id);
          auto conn = internal::connect_loopback(self_.id, self_.address, port);
          mrs_.push_back(seg_.registerWithPd(conn->pd()));

          // Save the connection and the regions
          auto lkey = mrs_.back()->lkey;
          save_conn(p.id, conn, lkey);
          for (auto &r : local_rkeys) {
            save_region(p.id, r.raddr, r.rkey);
          }
        }
      }
    }
  }

  /// Connect to all the remote memory nodes, save the QPs that are created, and
  /// get the memory regions and rkeys at each memory node
  ///
  /// @param memnodes TODO
  void connect_remote(std::vector<MachineInfo> &memnodes) {
    // Extract relevant information from Args map
    uint64_t qp_lanes = args_->uget(remus::QP_LANES);
    uint32_t port = args_->uget(remus::MN_PORT);

    for (const auto &p : memnodes) {
      if (p.id != self_.id) {
        for (uint64_t i = 0; i < qp_lanes; ++i) {
          // Connect, then register the big segment with that connection
          REMUS_INFO("Connecting to remote machine {}:{} (id = {}) from {}",
                     p.address, port, p.id, self_.id);
          auto conn = internal::connect_remote(self_.id, p.id, p.address, port,
                                               seg_, mrs_);

          // Get the RegionInfo vector
          auto got = conn->template DeliverVec<internal::RegionInfo>(seg_);
          if (got.status.t != remus::Ok) {
            REMUS_FATAL("{}", got.status.message.value());
          }

          // Save the connection and the regions
          auto lkey = mrs_.back()->lkey;
          save_conn(p.id, conn, lkey);
          for (auto &r : got.val.value())
            save_region(p.id, r.raddr, r.rkey);
        }
      }
    }
  }

  /// Register a thread by giving it a buffer and a unique, zero-based Id
  ///
  /// @return TODO
  std::pair<uint64_t, uint8_t *> register_thread() {
    uint64_t id = threads_++;
    if (id >= num_threads_) {
      REMUS_FATAL(
          "register_thread produced thread #{} when only {} are supported", id,
          num_threads_);
    }
    uint8_t *res = seg_.raw() + (id * thread_bufsz_);
    return {id, res};
  }

  /// Report the starting address of the requested Segment
  ///
  /// @param mn_id  TODO
  /// @param seg_id TODO
  /// @return TODO
  uint64_t get_seg_start(uint64_t mn_id, uint64_t seg_id) {
    return segs_[mn_id][seg_id].start_;
  }

  /// Report the most recently observed bump pointer value for the requested
  /// Segment
  ///
  /// TODO: The alloc_hint concept is probably coupling the allocator too
  ///       tightly to the ComputeNode?
  ///
  /// @param mn_id  TODO
  /// @param seg_id TODO
  /// @return TODO
  std::atomic<uint64_t> &get_alloc_hint(uint64_t mn_id, uint64_t seg_id) {
    return (segs_[mn_id][seg_id].hint_);
  }
};
} // namespace remus