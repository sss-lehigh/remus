#pragma once

#include <memory>
#include <protos/experiment.pb.h>
#include <vector>

#include "../rdma/rdma.h"

// [mfs]  This needs documentation.  It looks like it is more of an "experiment
//        manager" than a "server", but I'm not seeing how it manages remote
//        workers.  It also seems like overkill, since there is only one method
//        that is called once... why is it a class?
class Server {
public:
  ~Server() = default;

  static std::unique_ptr<Server> Create(rome::rdma::Peer server,
                                        std::vector<rome::rdma::Peer> clients,
                                        ExperimentParams params,
                                        rome::rdma::rdma_capability *pool) {
    return std::unique_ptr<Server>(new Server(server, clients, params, pool));
  }

  /// @brief Start the server
  ///
  /// @param done       a bool for inter-thread communication
  /// @param runtime_s  how long to wait before listening for finishing messages
  /// @param cleanup    [mfs] ???
  ///
  /// @return the status
  sss::Status Launch(volatile bool *done, int runtime_s,
                     std::function<void()> cleanup) {
    // Sleep while clients are running if there is a set runtime.
    if (runtime_s > 0) {
      // [mfs] We should always worry if there are explicit sleeps...
      ROME_INFO("SERVER :: Sleeping for {}", runtime_s);

      for (int it = 0; it < runtime_s * 10; it++) {
        // We sleep for 1 second, runtime times, and do intermittent cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check for server related cleanup
        //
        // [mfs]  I do not understand this cleanup parameter... why wouldn't the
        //        IHT be self-rehashing?  If it were to take a significant
        //        amount of time, then the total duration would be more than
        //        expected.
        cleanup();
      }
    }

    // [mfs]  I think more documentation of "done" is needed.  Who is setting
    //        it? Why isn't the thread who just did all that sleeping the one
    //        who sets it? Is it because there is an option for sleeping for a
    //        number of operations, instead of a number of seconds?

    // Sync with the clients
    while (!(*done)) {
      // Sleep for 1/10 second while not done
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Check for server related cleanup
      //
      // [mfs] See above... I don't understand this cleanup...
      cleanup();
    }

    // [mfs]  As in role_client, it seems that exposing pool_->GetConnection()
    //        is just to let the application implement a barrier using RPC.  If
    //        that's the case, why not just have a real barrier using one-sided
    //        primitives?

    // Wait for all the other nodes to send a message
    //
    // [mfs] This is remote clients, not local clients
    for (auto &p : peers_) {
      if (p.id == self_.id)
        continue; // ignore self since joining threads will force client and
                  // server to end at the same time
      ROME_INFO("SERVER :: receiving ack from {}", p.id);
      auto conn_or = pool_->GetConnection(p);
      // [mfs] If this exits early, do other things ever get cleaned up?
      RETURN_STATUSVAL_ON_ERROR(conn_or);

      auto *conn = conn_or.val.value();
      // [mfs]  This is confusing.  Does "Deliver" mean "Receive?"
      //
      // [mfs]  Since this is blocking, why not use Deliver instead of
      //        TryDeliver?
      auto msg = conn->channel()->Deliver<AckProto>();
      // [mfs] The ACK might not be sss::Ok... is that acceptable?
      ROME_INFO("SERVER :: received ack");
    }

    // Let all clients know that we are done
    for (auto &p : peers_) {
      if (p.id == self_.id)
        continue; // ignore self since joining threads will force client and
                  // server to end at the same time
      ROME_INFO("SERVER :: sending ack to {}", p.id);
      auto conn_or = pool_->GetConnection(p);
      // [mfs] If this exits early, do other things ever get cleaned up?
      RETURN_STATUSVAL_ON_ERROR(conn_or);
      auto *conn = conn_or.val.value();
      AckProto e;
      // Send back an ack proto let the client know that all the other clients
      // are done
      //
      // [mfs]  The AckProto is empty... we could get by with something simpler,
      //        like an int, to reduce the dependence on protobufs.
      auto sent = conn->channel()->Send(e);
      ROME_INFO("SERVER :: sent ack");
    }

    // [mfs]  This all feels quite fragile.  If anything is an error, this won't
    //        return OK.  Is that acceptable?

    // [mfs]  *this* is the point where the clock should be read to get the end
    //        time.
    return sss::Status::Ok();
  }

private:
  // [mfs] This all needs documentation
  //
  // [mfs] Why a factory pattern here?
  Server(rome::rdma::Peer self, std::vector<rome::rdma::Peer> peers,
         ExperimentParams params, rome::rdma::rdma_capability *pool)
      : self_(self), peers_(peers), params_(params), pool_(pool) {}

  const rome::rdma::Peer self_;
  std::vector<rome::rdma::Peer> peers_;
  const ExperimentParams params_;
  rome::rdma::rdma_capability *pool_;
};