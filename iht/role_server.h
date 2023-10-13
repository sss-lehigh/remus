#include <algorithm>
#include <atomic>

#include "../rdma/connection_manager/connection_manager.h"
#include "../rdma/memory_pool/memory_pool.h"
#include "common.h"
#include "structures/iht_ds.h"
#include "protos/experiment.pb.h"

namespace ExperimentManager {
/// @brief Sleep (to avoid taking up resources) and then try to sync a exit with the clients
/// @param socket_handle the socket manager resource for communicating with remote clients
/// @param runtime_s how long to wait before listening for finishing messages
/// @param cleanup a cleanup script to run every 100ms 
/// [esl] IMP: cleanup was removed because it is used in the other hashmap and not the IHT
///            and is not necessary for a minimal IHT. I left it in the documentation to answer your comments
/// @return ok status
void ClientStopBarrier(tcp::SocketManager& socket_handle, int runtime_s) {
  // Sleep while clients are running if there is a set runtime.
  if (runtime_s > 0) {
    ROME_INFO("SERVER :: Sleeping for {}", runtime_s);

    // Sleep for runtime seconds while the clients are running
    std::this_thread::sleep_for(std::chrono::seconds(runtime_s));
  }

  // [esl] IMP: The purpose of the tcp module is not to be efficient, but rather to be able to serve as a simple barrier 
  // at least until an efficient RDMA-based one can be created
  // It also serves the function of sending the remote_ptr, which is why the API is not a barrier but more of a client-server (N:1 relationship)

  // Receive a message from all clients to sync
  tcp::message recv_buffer[socket_handle.num_clients()];
  socket_handle.recv_from_all(recv_buffer);
  ROME_DEBUG("SERVER :: received ack");
  
  // Once we receive a message from everyone, everyone is done
  // So we now send an OK to exit message
  tcp::message send_buffer;
  socket_handle.send_to_all(&send_buffer);
  ROME_DEBUG("SERVER :: sent ack");
}
};
