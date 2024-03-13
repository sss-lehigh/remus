#pragma once

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <functional>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <string>
#include <thread>

#include "remus/logging/logging.h"
#include "connection.h"
#include "connection_utils.h"

namespace remus::rdma::internal {



/// Connector is a utility class for establishing connections among nodes
class Connector {

  /// Minimum value, in us, for exponential backoff
  static constexpr uint32_t kMinBackoffUs = 100;

  /// Maximum value, in us, for exponential backoff
  ///
  /// TODO: This seems too high?
  static constexpr uint32_t kMaxBackoffUs = 5000000;

  /// A function type; used for saving a connection to the Connection Map
  using saver_t = std::function<void(uint32_t, Connection *)>;

  uint32_t my_id_;                // This node's Id
  std::string address_;           // This node's IP address
  ibv_pd *pd_;                    // This node's protection domain
  const saver_t connection_saver; // Saves a connection to the Connection Map

  /// Configure attributes for a QP
  static ibv_qp_attr DefaultQpAttr() {
    ibv_qp_attr attr; // NB: cannot use = {0}
    std::memset(&attr, 0, sizeof(attr));

    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    attr.max_dest_rd_atomic = 8;
    attr.path_mtu = IBV_MTU_4096;
    attr.min_rnr_timer = 12;
    attr.rq_psn = 0;
    attr.sq_psn = 0;
    attr.timeout = 12;
    attr.retry_cnt = 7;
    attr.rnr_retry = 1;
    attr.max_rd_atomic = 8;
    return attr;
  }

public:
  /// Construct a connector by passing in the node's id/pd/address and a
  /// function for saving connections that this object will create
  Connector(uint32_t my_id, ibv_pd *pd, std::string address,
            std::function<void(uint32_t, Connection *)> saver)
      : my_id_(my_id), pd_(pd), address_(address), connection_saver(saver) {}

  /// Connect to a remote peer.  It is an error to use this to create a Loopback
  /// connection.  Terminates the program on any error.
  Connection *ConnectRemote(uint32_t peer_id, std::string_view server,
                            uint16_t port) {
    using namespace std::string_literals;
    if (peer_id == my_id_)
      ROME_FATAL("Cannot connect to localhost via ConnectRemote");

    while (true) {
      // Compute the info for the node we're connecting to
      auto port_str = std::to_string(htons(port));
      rdma_addrinfo hints = {0}, *resolved = nullptr;
      hints.ai_port_space = RDMA_PS_TCP;
      hints.ai_qp_type = IBV_QPT_RC;
      hints.ai_family = AF_IB;
      sockaddr_in src = {0};
      hints.ai_src_len = sizeof(src);
      src.sin_family = AF_INET;
      inet_aton(address_.data(), &src.sin_addr);
      hints.ai_src_addr = reinterpret_cast<sockaddr *>(&src);
      if (int gai_ret = rdma_getaddrinfo(server.data(), port_str.data(), &hints,
                                         &resolved);
          gai_ret != 0) {
        ROME_FATAL("rdma_getaddrinfo(): "s + gai_strerror(gai_ret));
      }

      // Start connecting to the remote node
      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      rdma_cm_id *id = nullptr;
      auto err = rdma_create_ep(&id, resolved, pd_, &init_attr);
      rdma_freeaddrinfo(resolved);
      if (err) {
        ROME_FATAL("rdma_create_ep(): "s + strerror(errno));
      }
      ROME_TRACE("[Connect] (Node {}) Trying to connect to: {} (id={})", my_id_,
                 peer_id, fmt::ptr(id));

      // Migrate the new endpoint to a nonblocking event channel and do more
      // config
      auto *event_channel = rdma_create_event_channel();
      make_nonblocking(event_channel->fd);
      if (rdma_migrate_id(id, event_channel) != 0) {
        ROME_FATAL("rdma_migrate_id(): "s + strerror(errno));
      }
      rdma_conn_param conn_param = {0};
      conn_param.private_data = &my_id_;
      conn_param.private_data_len = sizeof(my_id_);
      conn_param.retry_count = 7;
      conn_param.rnr_retry_count = 1;
      conn_param.responder_resources = 8;
      conn_param.initiator_depth = 8;
      if (rdma_connect(id, &conn_param) != 0) {
        ROME_FATAL("rdma_connect(): "s + strerror(errno));
      }

      // It takes a few events before the channel is ready to use
      uint32_t backoff_us_{0}; // for backoff
      bool do_inner = true;
      while (do_inner) {
        // Poll until we get an event
        rdma_cm_event *event;
        auto result = rdma_get_cm_event(id->channel, &event);
        while (result < 0 && errno == EAGAIN) {
          result = rdma_get_cm_event(id->channel, &event);
        }
        ROME_TRACE("[Connect] (Node {}) Got event: {} (id={})", my_id_,
                   rdma_event_str(event->event), fmt::ptr(id));

        switch (event->event) {
        case RDMA_CM_EVENT_ESTABLISHED: {
          // On an "established" event, we can make and save the connection
          if (rdma_ack_cm_event(event) != 0) {
            ROME_FATAL("rdma_ack_cm_event(): "s + strerror(errno));
          }
          make_sync(event_channel->fd); // TODO: Why sync instead of nonblocking
          make_nonblocking(id->recv_cq->channel->fd);
          make_nonblocking(id->send_cq->channel->fd);

          // Make, save, and return the connection
          //
          // TODO: Does the caller ever use the return value?
          auto new_conn = new Connection(my_id_, peer_id, id);
          connection_saver(peer_id, new_conn);
          ROME_TRACE(
              "Connected: dev={}, addr={}, port={}", id->verbs->device->name,
              inet_ntoa(reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(id))
                            ->sin_addr),
              rdma_get_src_port(id));
          return new_conn;
        }

        case RDMA_CM_EVENT_ADDR_RESOLVED:
          // On an ADDR_RESOLVED, we just ack
          if (rdma_ack_cm_event(event) != 0) {
            ROME_FATAL("rdma_ack_cm_event(): "s + strerror(errno));
          }
          break;

        default: {
          // If we get a REJECTED, we can wait and try again.  Otherwise fail
          auto cm_event = event->event;
          if (rdma_ack_cm_event(event) != 0) {
            ROME_FATAL("rdma_ack_cm_event(): "s + strerror(errno));
          }
          // Destruct intermediate state...
          rdma_destroy_ep(id);
          rdma_destroy_event_channel(event_channel);
          if (cm_event == RDMA_CM_EVENT_REJECTED) {
            ROME_WARN("ROME_CM_EVENT_REJECTED... backing off"s +
                      std::to_string(peer_id));
            backoff_us_ = backoff_us_ > 0
                              ? std::min((backoff_us_ + (100 * my_id_)) * 2,
                                         kMaxBackoffUs)
                              : kMinBackoffUs;
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us_));
            do_inner = false;
          } else {
            ROME_FATAL("Got unexpected event: "s + rdma_event_str(cm_event));
          }
        }
        }
      }
    }
  }

  /// Create a connection to the local device.  It is an error to use this to
  /// create a Remote connection.  Terminates the program on any error.
  Connection *ConnectLoopback(uint32_t peer_id, std::string_view server,
                              uint16_t port) {
    using namespace std::string_literals;

    if (peer_id != my_id_) {
      ROME_FATAL("Cannot connect to remote machine via ConnectLoopback");
    }

    while (true) {
      // TODO:  There is a lot of shared code between ConnectLoopback and
      //        ConnectRemote.  We should factor it out.  Some might be common
      //        to OnConnectRequest, too.

      // Compute the info for the node we're connecting to
      auto port_str = std::to_string(htons(port));
      rdma_addrinfo hints = {0}, *resolved = nullptr;
      hints.ai_port_space = RDMA_PS_TCP;
      hints.ai_qp_type = IBV_QPT_RC;
      hints.ai_family = AF_IB;
      struct sockaddr_in src = {0};
      hints.ai_src_len = sizeof(src);
      src.sin_family = AF_INET;
      inet_aton(address_.data(), &src.sin_addr);
      hints.ai_src_addr = reinterpret_cast<sockaddr *>(&src);
      if (int gai_ret = rdma_getaddrinfo(server.data(), port_str.data(), &hints,
                                         &resolved);
          gai_ret != 0) {
        ROME_FATAL("rdma_getaddrinfo(): "s + gai_strerror(gai_ret));
      }

      // Start making a connection
      ibv_qp_init_attr init_attr = DefaultQpInitAttr();
      rdma_cm_id *id = nullptr;
      auto err = rdma_create_ep(&id, resolved, pd_, &init_attr);
      rdma_freeaddrinfo(resolved);
      if (err) {
        ROME_FATAL("rdma_create_ep(): "s + strerror(errno));
      }
      ROME_ASSERT_DEBUG(id->qp != nullptr, "No QP associated with endpoint");
      ROME_TRACE("[Connect] (Node {}) Trying to connect to: {} (id={})", my_id_,
                 peer_id, fmt::ptr(id));

      //
      // Created a connection with the device now lets query the ports to find
      // one that works for us and use that
      //

      ibv_device_attr dev_attr;
      if (ibv_query_device(id->verbs, &dev_attr) != 0) {
        ROME_FATAL("ibv_query_device(): "s + strerror(errno));
      }

      ROME_TRACE("Found device has "s + std::to_string(dev_attr.phys_port_cnt) + " ports"s); 

      ibv_port_attr port_attr;
      uint32_t LOOPBACK_PORT_NUM = 1;

      // use first port that is active for loopback
      for(int i = 1; i <= dev_attr.phys_port_cnt; ++i) {
        if (ibv_query_port(id->verbs, i, &port_attr) != 0) {
          ROME_FATAL("ibv_query_port(): "s + strerror(errno));
        }
        if (port_attr.state == IBV_PORT_ACTIVE) {
          LOOPBACK_PORT_NUM = i;
          ROME_DEBUG("Using physical port "s + std::to_string(i) 
                     + " for loopback"s);
          break;
        }
      }


      //
      //

      ibv_qp_attr attr = DefaultQpAttr();
      attr.qp_state = IBV_QPS_INIT;
      attr.port_num = LOOPBACK_PORT_NUM;
      int attr_mask =
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
      if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
        ROME_FATAL("ibv_modify_qp(): "s + strerror(errno));
      }

      attr.ah_attr.dlid = port_attr.lid;
      attr.ah_attr.port_num = LOOPBACK_PORT_NUM;

      if(port_attr.lid == 0x0 or
         (port_attr.flags & IBV_QPF_GRH_REQUIRED) != 0) {

        ROME_DEBUG("Creating a GRH is necessary");

        // This LID is invalid and likely RoCE, so this is a hack to
        // get around that or the GRH is required regardless
        
        // Our address handle has a global route
        attr.ah_attr.is_global = 1;

        // We query the first GID, which should always exist
        // There may be others, but I don't think that should impact
        // anything for us
        // We can go from gid = 0 to gid = port_attr.gid_table_len - 1
        ROME_ASSERT(port_attr.gid_tbl_len >= 1, 
                    "Need a gid table that has at least one entry");
        ibv_gid gid;
        if (ibv_query_gid(id->verbs, LOOPBACK_PORT_NUM, 0, &gid)) {
          ROME_FATAL("Fail on query gid"s);
        }

        // Set our gid
        attr.ah_attr.grh.dgid = gid;
        // we set our gid to the gid index we queried 
        attr.ah_attr.grh.sgid_index = 0;
        // allow for the max number of hops 
        attr.ah_attr.grh.hop_limit = 0xFF;
        attr.ah_attr.grh.traffic_class = 0; // some trafic class
        // non-zero is support to give a hint to switches
        // but we dont care; this is loopback
        attr.ah_attr.grh.flow_label = 0;
      }

      //ROME_ASSERT_DEBUG(port_attr.lid != 0x0, "LID of port uses reserved number");

      attr.qp_state = IBV_QPS_RTR;
      attr.dest_qp_num = id->qp->qp_num;
      attr_mask =
          (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
           IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
      if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
        ROME_FATAL("ibv_modify_qp(): "s + strerror(errno));
      }
      attr.qp_state = IBV_QPS_RTS;
      attr_mask =
          (IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
           IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
      ROME_TRACE("Loopback: IBV_QPS_RTS");
      if (ibv_modify_qp(id->qp, &attr, attr_mask) != 0) {
        ROME_FATAL("ibv_modify_qp(): "s + strerror(errno));
      }
      make_nonblocking(id->recv_cq->channel->fd);
      make_nonblocking(id->send_cq->channel->fd);

      // Make, save, and return the Connection
      //
      // TODO: Does the caller ever use the return value?
      auto res = new Connection(my_id_, my_id_, id);
      connection_saver(my_id_, res);
      ROME_TRACE(
          "Connected Loopback: dev={}, addr={}, port={}",
          id->verbs->device->name,
          inet_ntoa(reinterpret_cast<sockaddr_in *>(rdma_get_local_addr(id))
                        ->sin_addr),
          rdma_get_src_port(id));
      return res;
    }
  }
};

} // namespace remus::rdma::internal
