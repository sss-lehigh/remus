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
#include "util.h"

namespace remus::internal {

/// Minimum microseconds for exponential backoff
constexpr uint32_t connect_backoff_min_us = 100;

/// Maximum microseconds for exponential backoff
constexpr uint32_t connect_backoff_max_us = 5000000;

/// Common code for creating and initializing an endpoint
///
/// @param address  The address that will be connected to
/// @param port     The port to connect to
///
/// @return A connection/id that has been configured properly
// rdma_cm_id *initialize_ep(std::string_view address, uint16_t port) {
//   // Compute the info for the node we're connecting to
//   auto port_str = std::to_string(htons(port));
//   rdma_addrinfo hints, *resolved = nullptr;
//   std::memset(&hints, 0, sizeof(hints));
//   struct sockaddr_in src;
//   std::memset(&src, 0, sizeof(src));
//   hints.ai_port_space = RDMA_PS_TCP;
//   hints.ai_qp_type = IBV_QPT_RC;
//   hints.ai_family = AF_IB;
//   hints.ai_src_len = sizeof(src);
//   src.sin_family = AF_INET;
//   // inet_aton(address_.data(), &src.sin_addr);
//   hints.ai_src_addr = reinterpret_cast<sockaddr *>(&src);
//   if (int err =
//           rdma_getaddrinfo(address.data(), port_str.data(), &hints, &resolved);
//       err != 0) {
//     REMUS_FATAL("rdma_getaddrinfo(): {}", gai_strerror(err));
//   }

//   // Start making a connection
//   ibv_qp_init_attr init_attr = make_default_qp_init_attrs();
//   rdma_cm_id *id = nullptr;
//   auto err = rdma_create_ep(&id, resolved, nullptr, &init_attr);
//   rdma_freeaddrinfo(resolved);
//   if (err) {
//     REMUS_FATAL("compute node rdma_create_ep(): {}", strerror(errno));
//   }
//   return id;
// }

rdma_cm_id *initialize_ep(std::string_view address, uint16_t port, 
                          const std::string &device_name) {
  // Open specific device
  int num_devices;
  auto **device_list = ibv_get_device_list(&num_devices);
  ibv_context *context = nullptr;
  
  for (int i = 0; i < num_devices; ++i) {
    if (strcmp(ibv_get_device_name(device_list[i]), device_name.c_str()) == 0) {
      context = ibv_open_device(device_list[i]);
      break;
    }
  }
  ibv_free_device_list(device_list);
  
  if (!context) {
    REMUS_FATAL("Could not open device {}", device_name);
  }

  REMUS_INFO("Connecting from device {} to {}:{}", device_name, address, port);

  // Create event channel and CM ID
  rdma_event_channel *channel = rdma_create_event_channel();
  rdma_cm_id *id = nullptr;
  rdma_create_id(channel, &id, nullptr, RDMA_PS_TCP);
  
  // KEY: Associate with our device BEFORE resolving
  id->verbs = context;
  
  // Resolve destination address
  struct sockaddr_in dst_addr;
  memset(&dst_addr, 0, sizeof(dst_addr));
  dst_addr.sin_family = AF_INET;
  dst_addr.sin_port = htons(port);
  inet_pton(AF_INET, address.data(), &dst_addr.sin_addr);
  
  // Resolve with NULL source (let device routing decide)
  if (rdma_resolve_addr(id, nullptr, (struct sockaddr*)&dst_addr, 2000) != 0) {
    REMUS_FATAL("rdma_resolve_addr failed: {}", strerror(errno));
  }
  
  // Wait for resolution
  rdma_cm_event *event;
  rdma_get_cm_event(channel, &event);
  
  if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
    REMUS_FATAL("Expected ADDR_RESOLVED, got {}", rdma_event_str(event->event));
  }
  rdma_ack_cm_event(event);
  
  // Continue with route resolution...
  if (rdma_resolve_route(id, 2000) != 0) {
    REMUS_FATAL("rdma_resolve_route failed: {}", strerror(errno));
  }
  
  rdma_get_cm_event(channel, &event);
  if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
    REMUS_FATAL("Expected ROUTE_RESOLVED, got {}", rdma_event_str(event->event));
  }
  rdma_ack_cm_event(event);
  
  // Create QP
  ibv_pd *pd = ibv_alloc_pd(context);
  ibv_cq *cq = ibv_create_cq(context, kMaxWr * 2, nullptr, nullptr, 0);
  
  ibv_qp_init_attr init_attr = make_default_qp_init_attrs();
  init_attr.send_cq = cq;
  init_attr.recv_cq = cq;
  
  ibv_qp *qp = ibv_create_qp(pd, &init_attr);
  
  id->qp = qp;
  id->pd = pd;
  id->send_cq = cq;
  id->recv_cq = cq;
  //Transition QP To INIT state
  ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;  // Use first active port
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | 
                            IBV_ACCESS_REMOTE_READ | 
                            IBV_ACCESS_REMOTE_ATOMIC |
                            IBV_ACCESS_LOCAL_WRITE;
  
  int attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | 
                  IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  
  if (ibv_modify_qp(qp, &qp_attr, attr_mask) != 0) {
    REMUS_FATAL("Failed to transition QP to INIT: {}", strerror(errno));
  }

  return id;
}

/// Connect to a remote machine.  It is an error to use this to create a
/// Loopback connection.  Terminates the program on any error.
///
/// @param my_id    The id of this (compute) node
/// @param mn_id    The id of the memory node
/// @param mn_addr  The address of the memory node
/// @param port     The port to connect to
///
/// @return A connection object for the new connection
Connection * connect_remote(uint32_t my_id, uint32_t mn_id,
                           std::string_view mn_addr, uint16_t port, 
                           internal::Segment& seg_, 
                           std::vector<internal::ibv_mr_ptr>& mrs_,
                           const std::string &device_name) {
  uint32_t backoff_us_ = 0;
  while (true) {
    rdma_cm_id *id = initialize_ep(mn_addr, port, device_name);
    auto mr = seg_.registerWithPd(id->pd);
    RDMA_CM_ASSERT(rdma_post_recv, id, nullptr, seg_.raw(), seg_.capacity(), mr.get());
    mrs_.push_back(std::move(mr));

    // Set ACK timeout BEFORE connecting
    uint8_t timeout = 12;
    if (rdma_set_option(id, RDMA_OPTION_ID, RDMA_OPTION_ID_ACK_TIMEOUT,
                        &timeout, sizeof(timeout)) != 0) {
        REMUS_FATAL("rdma_set_option(): {}", strerror(errno));    
    }

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

    // Wait for ESTABLISHED on the ORIGINAL channel (from initialize_ep)
    while (true) {
      rdma_cm_event *event;
      auto result = rdma_get_cm_event(id->channel, &event);
      while (result < 0 && errno == EAGAIN) {
        result = rdma_get_cm_event(id->channel, &event);
      }

      auto cm_event = event->event;
      if (rdma_ack_cm_event(event) != 0) {
        REMUS_FATAL("rdma_ack_cm_event(): {}", strerror(errno));
      }

      if (cm_event == RDMA_CM_EVENT_ESTABLISHED) {
        // ✅ NOW migrate AFTER established
        auto *event_channel = rdma_create_event_channel();
        make_nonblocking(event_channel->fd);
        if (rdma_migrate_id(id, event_channel) != 0) {
          REMUS_FATAL("rdma_migrate_id(): {}", strerror(errno));
        }
        make_sync(event_channel->fd);
        make_nonblocking(id->recv_cq->channel->fd);
        make_nonblocking(id->send_cq->channel->fd);

        return new Connection(my_id, mn_id, id);
      }
      else if (cm_event == RDMA_CM_EVENT_REJECTED) {
        rdma_destroy_ep(id);
        backoff_us_ = backoff_us_ > 0 ? std::min((backoff_us_ + (100 * my_id)) * 2, connect_backoff_max_us) : connect_backoff_min_us;
        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us_));
        break;
      }
      else if (cm_event == RDMA_CM_EVENT_ADDR_RESOLVED) {
        // Just ack and continue
      }
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
Connection *connect_loopback(uint32_t my_id, std::string_view address,
                             uint16_t port, const std::string &device_name) { 
  // Do the initial endpoint configuration
  rdma_cm_id *id = initialize_ep(address, port, device_name);
  // REMUS_DEBUG("[Connect] (Node {}) Trying to connect loopback", my_id);

  // Created a connection with the device now lets query the ports to find
  // one that works for us and use that
  ibv_device_attr dev_attr;
  if (ibv_query_device(id->verbs, &dev_attr) != 0) {
    REMUS_FATAL("ibv_query_device(): {}", strerror(errno));
  }

  // REMUS_DEBUG("Found device has {} ports", dev_attr.phys_port_cnt);

  // NB: There are a whole bunch of funny rules and configuration requirements
  // when connecting via loopback, especially if it's not an IB device.
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

  // [mfs] These values feel like they need more documentation
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

  // This LID is invalid and likely RoCE, so this is a hack to
  // get around that or the GRH is required regardless
  if (port_attr.lid == 0x0 || (port_attr.flags & IBV_QPF_GRH_REQUIRED) != 0) {
    // REMUS_DEBUG("Creating a GRH is necessary");

    // Our address handle has a global route
    attr.ah_attr.is_global = 1;

    // We query the first GID, which should always exist
    // There may be others, but I don't think that should impact
    // anything for us
    // We can go from gid = 0 to gid = port_attr.gid_table_len - 1
    REMUS_ASSERT(port_attr.gid_tbl_len >= 1,
                 "Need a gid table that has at least one entry");
    ibv_gid gid;
    if (ibv_query_gid(id->verbs, LOOPBACK_PORT_NUM, 0, &gid)) {
      REMUS_FATAL("Fail on query gid");
    }

    // Set our gid
    attr.ah_attr.grh.dgid = gid;
    // we set our gid to the gid index we queried
    attr.ah_attr.grh.sgid_index = 0;
    // allow for the max number of hops
    attr.ah_attr.grh.hop_limit = 0xFF;
    attr.ah_attr.grh.traffic_class = 0; // some traffic class
    // non-zero is support to give a hint to switches
    // but we dont care; this is loopback
    attr.ah_attr.grh.flow_label = 0;
  }

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
/// one, then chops it up for its ComputeThreads.  This lets us only make one
/// registration per connection, while still supporting many threads.
class ComputeNode {
  /// A connection to a remote node, and an lkey that enables using the local
  /// Segment to do one-sided operations across that connection.
  struct conn_info {
    std::unique_ptr<internal::Connection> conn_; // The open connection
    uint32_t lkey;                               // The lkey
  };

  const MachineInfo self_;                // This node's id and address
  std::vector<internal::ibv_mr_ptr> mrs_; // MRs for seg_
  const uint64_t num_threads_;            // Number of threads to support
  const uint64_t thread_bufsz_;           // Segment size for each thread
  internal::Segment seg_;                 // Use this for staging receives
  std::atomic<uint64_t> threads_;         // Number of registered threads

  using conn_map = std::unordered_map<uint16_t, std::vector<conn_info>>;
  using rkey_map = std::unordered_map<uint64_t, uint32_t>;

  /// A map of all of the connections we have for each node
  ///
  /// [mfs] Since connections are only to memory nodes, and memory node indices
  ///       start at 0 and are contiguous, this could just be a vector of
  ///       vectors.
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
    seg_t(uint64_t start, uint64_t hint) : start_(start), hint_(hint) {}
    /// Copy constructor, so we can put these in a vector
    seg_t(const seg_t &rhs) : start_(rhs.start_), hint_(rhs.hint_.load()) {}
  };

  /// A map from MemoryNode id to all of the Segments at that MemoryNode.  Each
  /// Segment is paired with this node's last observed value the segment's
  /// allocated_ field, so that we ahve a higher likelihood of FAA succeeding on
  /// the first try.
  ///
  /// [mfs] key should be uint16_t?
  ///
  /// [mfs] Why not a vector instead of a map?
  std::unordered_map<uint64_t, std::vector<seg_t>> segs_;

  std::shared_ptr<remus::ArgMap> args_; // The program's command-line args

  /// Save the connection to node_id, which has registered all ComputeThread
  /// Segments to use lkey.
  void save_conn(uint16_t node_id, internal::Connection *conn, uint32_t lkey) {
    if (node_connections_.find(node_id) == node_connections_.end())
      node_connections_.insert({node_id, std::vector<conn_info>()});
    node_connections_[node_id].emplace_back(
        conn_info{std::unique_ptr<internal::Connection>(conn), lkey});
  }

  /// Save the rkey for a given node/region pair
  void save_region(uint16_t node_id, uint64_t region, uint32_t rkey) {
    constexpr uint64_t node_mask = 0xFFFFULL << 48;
    REMUS_ASSERT((region & node_mask) == 0, "Top bits of region must be 0");
    // [mfs]  Should we also assert that the low bits are 0?  We expect
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
      // NB:  Rather than do an rdma_read, we guess that nothing has been
      //      allocated yet, and thus the bump allocator FAA hint should be the
      //      tail of the segment's metadata.
      segs_[node_id].emplace_back(key, sizeof(internal::ControlBlock));
    }
  }

public:
  /// Return a connection and lkey for interacting with an rdma_ptr
  conn_info &get_conn(uint64_t ptr_raw, uint64_t idx) {
    uint64_t node_id = ptr_raw >> 48 & 0xFFFF;
    return node_connections_[node_id].at(idx);
  }

  // // Return a connection and lket for interacting with a given node and lane index
  // conn_info &get_conn_by_index(uint16_t node_id, uint64_t lane_idx) {
  //   auto it = node_connections_.find(node_id);
  //   if (it == node_connections_.end()) {
  //       REMUS_FATAL("No connections found for node_id {}", node_id);
  //   }
  //   return it->second.at(lane_idx);
  // }

  /// Return the rkey for a node/segment pair
  uint32_t get_rkey(uint64_t raw) {
    // [mfs]  This is a non-constexpr const.  Can we make it a field, to avoid
    //        recomputing?
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
        seg_((1ULL << (64 - __builtin_clzll(num_threads_ * thread_bufsz_ - 1)))), threads_(0),
        seg_mask_((1ULL << args->uget(remus::SEG_SIZE)) - 1), args_(args) {
    REMUS_INFO("Node {}: Configuring Compute Node", args->uget(remus::NODE_ID));
    // Initialize the seg map
    uint64_t m0 = args->uget(remus::FIRST_MN_ID);
    uint64_t mn = args->uget(remus::LAST_MN_ID);
    for (uint64_t i = m0; i <= mn; ++i) {
      segs_[i] = std::vector<seg_t>();
    }
  }

  /// [mfs] Do we need a proper dtor?  IDK.  Connection map cleanup should be
  ///       automatic.
  ~ComputeNode() = default;

  /// Create QPs to the localhost.  This should only be used when a ComputeNode
  /// is also a MemoryNode.  It requires the MemoryNode to provide its rkeys.
  void connect_local(std::vector<MachineInfo> &memnodes,
                     std::vector<internal::RegionInfo> local_rkeys) {
    uint64_t qp_lanes = args_->uget(remus::QP_LANES);
    uint32_t port = args_->uget(remus::MN_PORT);

    // AUTO-DETECT
    std::string dev_name = internal::find_experimental_device(self_.address);
    if (dev_name.empty()) {
      REMUS_FATAL("Could not find RDMA device on experimental network");
    }
    REMUS_INFO("Auto-detected RDMA device: {}", dev_name);

    for (auto &p : memnodes) {
      if (p.id == self_.id) {
        for (uint64_t i = 0; i < qp_lanes; ++i) {
          // Connect, then register the big segment with that connection
          REMUS_INFO("Connecting to localhost {}:{} (id = {}) on device {}", 
                   p.address, port, p.id, dev_name);
          auto conn = internal::connect_loopback(self_.id, self_.address, port, dev_name);
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
  /// Connect to all the remote memory nodes, save the QPs that are created, and
/// get the memory regions and rkeys at each memory node
void connect_remote(std::vector<MachineInfo> &memnodes) {
  // Extract relevant information from Args map
  uint64_t qp_lanes = args_->uget(remus::QP_LANES);
  uint32_t port = args_->uget(remus::MN_PORT);
  REMUS_INFO("🚀 Node {}: NEW connect_remote() starting with {} memnodes, {} qp_lanes", 
             self_.id, memnodes.size(), qp_lanes);

  // AUTO-DETECT
  std::string dev_name = internal::find_experimental_device(self_.address);
  if (dev_name.empty()) {
    REMUS_FATAL("Could not find RDMA device on experimental network");
  }
  REMUS_INFO("Auto-detected RDMA device: {}", dev_name);

  // Store pending connections with their metadata
  struct PendingConn {
    internal::Connection* conn;
    uint16_t node_id;
    uint32_t lkey;
  };
  std::vector<PendingConn> pending;

  // PHASE 1: Establish ALL connections first (non-blocking)
  for (const auto &p : memnodes) {
    if (p.id != self_.id) {
      for (uint64_t i = 0; i < qp_lanes; ++i) {
        REMUS_INFO("Connecting to remote machine {}:{} (id = {}) from {} on device {}",
                 p.address, port, p.id, self_.id, dev_name);
        
        // This establishes the connection but doesn't wait for data
        auto conn = internal::connect_remote(self_.id, p.id, p.address, port, seg_, mrs_, dev_name);
        auto lkey = mrs_.back()->lkey;
        
        pending.push_back({conn, (uint16_t)p.id, lkey});
      }
    }
  }

  REMUS_INFO("Node {}: All {} connections established, now receiving region info", 
             self_.id, pending.size());

  // PHASE 2: Now receive region info from all connections
  // At this point, all memory nodes have received all their expected connections
  // and can start sending region info
  for (auto &pc : pending) {
    auto got = pc.conn->template DeliverVec<internal::RegionInfo>(seg_);
    if (got.status.t != remus::Ok) {
      REMUS_FATAL("Failed to receive region info from node {}: {}", 
                  pc.node_id, got.status.message.value());
    }

    // Save the connection and the regions
    save_conn(pc.node_id, pc.conn, pc.lkey);
    for (auto &r : got.val.value()) {
      save_region(pc.node_id, r.raddr, r.rkey);
    }
  }
  
  REMUS_INFO("Node {}: Successfully received all region info", self_.id);
}

  /// Register a thread by giving it a buffer and a unique, zero-based Id
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

  /// Set the control flag to 1 in each MemoryNode's first segment's control
  /// block, so that the MemoryNodes know to shut down
  void send_shutdown() {
    // NB: segs_.size() is the number of MemoryNodes
    for (uint64_t i = 0; i < segs_.size(); i++) {
      // segment0 is where we'll write the control message
      auto ptr = remus::rdma_ptr<uint64_t>(
          segs_[i][0].start_ +
          offsetof(remus::internal::ControlBlock, control_flag_));

      // NB:  The code is sequential, so we can use the start of seg_ and the
      //      0th connection without worrying about races
      auto &ci = get_conn(ptr.raw(), 0);
      uint32_t rkey = get_rkey(ptr.raw());
      std::atomic<int> counter;
      internal::Write(ptr, (uint64_t)1, seg_.raw(), rkey, ci.lkey,
                      ci.conn_.get(), &counter);
    }
  }

  /// Report the starting address of the requested Segment
  uint64_t get_seg_start(uint64_t mn_id, uint64_t seg_id) {
    return segs_[mn_id][seg_id].start_;
  }

  /// Report the most recently observed bump pointer value for the requested
  /// Segment
  ///
  /// [mfs] The alloc_hint concept is probably coupling the allocator too
  ///       tightly to the ComputeNode?
  std::atomic<uint64_t> &get_alloc_hint(uint64_t mn_id, uint64_t seg_id) {
    return (segs_[mn_id][seg_id].hint_);
  }
};
} // namespace remus