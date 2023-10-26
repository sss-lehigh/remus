#pragma once

#include <barrier>
#include <chrono>
#include <protos/experiment.pb.h>

#include "../colosseum/workload_driver.h"
#include "../logging/logging.h"
#include "../rdma/rdma.h"

#include "common.h"

// [mfs] Rework to avoid needing these two lines... template the client?
#include "structures/iht_ds.h"
typedef RdmaIHT<int, int, 16, 1024> IHT;

using SystemClock = std::chrono::system_clock;
using SteadyClock = std::chrono::steady_clock;

std::string fromStateValue(state_value value) {
  if (FALSE_STATE == value) {
    return "FALSE";
  } else if (TRUE_STATE == value) {
    return "TRUE";
  } else if (REHASH_DELETED == value) {
    return "REHASH DELETED";
  } else {
    return std::string("UNKNOWN - ") + std::to_string(value);
  }
}

// Function to run a test case
void test_output(bool show_passing, HT_Res<int> actual, HT_Res<int> expected,
                 std::string message) {
  if (actual.status != expected.status && actual.result != expected.result) {
    ROME_INFO("[-] {} func():({},{}) != expected:({},{})", message,
              fromStateValue(actual.status), actual.result,
              fromStateValue(expected.status), expected.result);
  } else if (show_passing) {
    ROME_INFO("[+] Test Case {} Passed!", message);
  }
}

template <class Operation> class Client {
  // static_assert(::rome::IsClientAdapter<Client, Operation>);

public:
  // [mfs]  Here and in Server, I don't understand the factory pattern.  It's
  //        not really adding any value.
  static std::unique_ptr<Client>
  Create(const rome::rdma::Peer &self, const rome::rdma::Peer &server,
         const std::vector<rome::rdma::Peer> &peers, ExperimentParams &params,
         std::barrier<> *barrier, IHT *iht, bool master_client) {
    return std::unique_ptr<Client>(
        new Client(self, server, peers, params, barrier, iht, master_client));
  }

  /// @brief Run the client
  /// @param client the client instance to run with
  /// @param done a volatile bool for inter-thread communication
  /// @param frac if 0, won't populate. Otherwise, will do this fraction of the
  /// population
  /// @return the resultproto
  static sss::StatusVal<rome::WorkloadDriverProto>
  Run(std::unique_ptr<Client> client, volatile bool *done, double frac) {
    // [mfs]  I was hopeful that this code was going to actually populate the
    //        data structure from *multiple nodes* simultaneously.  It should,
    //        or else all of the initial elists and plists are going to be on
    //        the same machine, which probably means all of the elists and
    //        plists will always be on the same machine.
    //
    // [mfs]  This should be in another function
    if (client->master_client_) {
      int key_lb = client->params_.key_lb(), key_ub = client->params_.key_ub();
      int op_count = (key_ub - key_lb) * frac;
      ROME_INFO("CLIENT :: Data structure ({}%) is being populated ({} items "
                "inserted) by this client",
                frac * 100, op_count);
      client->iht_->populate(op_count, key_lb, key_ub,
                             [=](int key) { return key; });
      ROME_INFO("CLIENT :: Done with populate!");
      // TODO: Sleeping for 1 second to account for difference between remote
      // client start times. Must fix this in the future to a better solution
      // The idea is even though remote nodes won't being workload at the same
      // time, at least the data structure is guaranteed to be populated
      //
      // [mfs] Indeed, this indicates the need for a distributed barrier
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // TODO: Signal Handler
    // signal(SIGINT, signal_handler);

    // auto *client_ptr = client.get();
    std::vector<Operation> operations = std::vector<Operation>();

    // initialize random number generator and key_range
    int key_range = client->params_.key_ub() - client->params_.key_lb();

    // Create a random operation generator that is
    // - evenly distributed among the key range
    // - within the specified ratios for operations
    //
    // [mfs]  Since we're working with ints, why not use int distributions? They
    //        should be faster.
    std::uniform_real_distribution<double> dist =
        std::uniform_real_distribution<double>(0.0, 1.0);
    // [mfs]  Just to be sure: will every thread, on every node, have a
    //        different seed?  Also, is it really necessary to have a different
    //        seed for each trial, or should the seed be a function of the node
    //        / thread, for repeatability?
    std::default_random_engine gen((unsigned)std::time(NULL));
    int lb = client->params_.key_lb();
    int contains = client->params_.contains();
    int insert = client->params_.insert();
    std::function<Operation(void)> generator = [&]() {
      double rng = dist(gen) * 100;
      int k = dist(gen) * key_range + lb;
      if (rng < contains) { // between 0 and CONTAINS
        return Operation(CONTAINS, k, 0);
      } else if (rng <
                 contains + insert) { // between CONTAINS and CONTAINS + INSERT
        return Operation(INSERT, k, k);
      } else {
        return Operation(REMOVE, k, 0);
      }
    };

    // Generate two streams based on what the user wants (operation count or
    // timed stream)
    std::unique_ptr<rome::Stream<Operation>> workload_stream;
    if (client->params_.unlimited_stream()) {
      workload_stream =
          std::make_unique<rome::EndlessStream<Operation>>(generator);
    } else {
      // Deliver a workload
      //
      // [mfs]  This seems problematic.  Making the whole stream ahead of time
      //        is going to increase the variance between when threads start,
      //        and it's going to lead to bad cache behavior.  Why isn't there a
      //        rome::FixedLengthStream?
      int WORKLOAD_AMOUNT = client->params_.op_count();
      for (int j = 0; j < WORKLOAD_AMOUNT; j++) {
        operations.push_back(generator());
      }
      workload_stream =
          std::make_unique<rome::TestStream<Operation>>(operations);
    }

    // Create and start the workload driver (also starts client and lets it
    // run).
    int32_t runtime = client->params_.runtime();
    int32_t qps_sample_rate = client->params_.qps_sample_rate();
    std::barrier<> *barr = client->barrier_;
    bool master_client = client->master_client_;
    // [mfs] Again, it looks like Create() is an unnecessary factory
    auto driver = rome::WorkloadDriver<Client, Operation>::Create(
        std::move(client), std::move(workload_stream),
        std::chrono::milliseconds(qps_sample_rate));
    // [mfs]  This is quite odd.  The current thread is invoking an async thread
    //        to actually do the work, which means we have lots of extra thread
    //        creation and joining.
    OK_OR_FAIL(driver->Start());
    // [mfs]
    std::this_thread::sleep_for(std::chrono::seconds(runtime));
    ROME_INFO("Done here, stop sequence");
    // Wait for all the clients to stop. Then set the done to true to release
    // the server
    if (master_client) {
      if (barr != nullptr)
        barr->arrive_and_wait();
    }

    // [mfs]  These writes to done are racy... lots of threads writing here...
    //        Also, it should be a std::atomic, no?  But who is being signalled?
    //        Just the server thread (if there is one)?
    *done = true;

    // [mfs] If we didn't use WorkloadDriver, then we wouldn't need Stop()
    OK_OR_FAIL(driver->Stop());
    ROME_INFO("CLIENT :: Driver generated {}", driver->ToString());
    // [mfs]  It seems like these protos aren't being sent across machines.  Are
    //        they really needed?
    return {sss::Status::Ok(), driver->ToProto()};
  }

  // Start the client
  sss::Status Start() {
    ROME_INFO("CLIENT :: Starting client...");
    // Conditional to allow us to bypass the barrier for certain client types
    // We want to start at the same time
    //
    // [mfs]  The entire barrier infrastructure is odd.  Nobody is using it to
    //        know when to get time, and it's completely per-node.
    if (barrier_ != nullptr)
      barrier_->arrive_and_wait();
    return sss::Status::Ok();
  }

  // Runs the next operation
  sss::Status Apply(const Operation &op) {
    count++;
    HT_Res<int> res = HT_Res<int>(FALSE_STATE, 0);
    switch (op.op_type) {
    case (CONTAINS):
      // [mfs]  I don't understand the purpose of "progression".  Is it just for
      //        getting periodic output?  If so, it's going to hurt the
      //        experiment's latency, so it's probably a bad idea.
      if (count % progression == 0)
        ROME_INFO("Running Operation {}: contains({})", count, op.key);
      res = iht_->contains(op.key);
      if (res.status == TRUE_STATE)
        ROME_ASSERT(res.result == op.key,
                    "Invalid result of ({}) contains operation {}!={}",
                    res.status, res.result, op.key);
      break;
    case (INSERT):
      if (count % progression == 0)
        ROME_INFO("Running Operation {}: insert({}, {})", count, op.key,
                  op.value);
      res = iht_->insert(op.key, op.value);
      break;
    case (REMOVE):
      if (count % progression == 0)
        ROME_INFO("Running Operation {}: remove({})", count, op.key);
      res = iht_->remove(op.key);
      if (res.status == TRUE_STATE)
        ROME_ASSERT(res.result == op.key,
                    "Invalid result of ({}) remove operation {}!={}",
                    res.status, res.result, op.key);
      break;
    default:
      ROME_INFO("Expected CONTAINS, INSERT, or REMOVE operation.");
      break;
    }
    // Think in between operations for simulation purposes.
    //
    // [mfs]  This doesn't actually work.  The "simulation" doesn't impact the
    //        cache or TLB, so it's not really representative of real work... it
    //        just throttles throughput (or throttles contention).
    if (params_.has_think_time() && params_.think_time() != 0) {
      auto start = SystemClock::now();
      while (SystemClock::now() - start <
             std::chrono::nanoseconds(params_.think_time()))
        ;
    }
    return sss::Status::Ok();
  }

  /// @brief Runs single-client silent-server test cases on the iht
  /// @param at_scale is true for testing at scale (+10,000 operations)
  /// @return OkStatus if everything worked. Otherwise will shutdown the client.
  //
  // [mfs] Why not split into two functions, instead of using "at_scale"?
  sss::Status Operations(bool at_scale) {
    if (at_scale) {
      int scale_size = (CNF_PLIST_SIZE * CNF_ELIST_SIZE) * 8;
      bool show_passing = false;
      // [mfs]  All of this string creation and concatenation is going to take a
      //        long time.  Is it part of the timed portion of the experiment?
      for (int i = 0; i < scale_size; i++) {
        test_output(show_passing, iht_->contains(i),
                    HT_Res<int>(FALSE_STATE, 0),
                    std::string("Contains ") + std::to_string(i) +
                        std::string(" false"));
        test_output(show_passing, iht_->insert(i, i),
                    HT_Res<int>(TRUE_STATE, 0),
                    std::string("Insert ") + std::to_string(i));
        test_output(show_passing, iht_->contains(i), HT_Res<int>(TRUE_STATE, i),
                    std::string("Contains ") + std::to_string(i) +
                        std::string(" true"));
      }
      ROME_INFO(" = 25% Finished = ");
      for (int i = 0; i < scale_size; i++) {
        test_output(show_passing, iht_->contains(i), HT_Res<int>(TRUE_STATE, i),
                    std::string("Contains ") + std::to_string(i) +
                        std::string(" maintains true"));
      }
      ROME_INFO(" = 50% Finished = ");
      for (int i = 0; i < scale_size; i++) {
        test_output(show_passing, iht_->remove(i), HT_Res<int>(TRUE_STATE, i),
                    std::string("Removes ") + std::to_string(i));
        test_output(show_passing, iht_->contains(i),
                    HT_Res<int>(FALSE_STATE, 0),
                    std::string("Contains ") + std::to_string(i) +
                        std::string(" false"));
      }
      ROME_INFO(" = 75% Finished = ");
      for (int i = 0; i < scale_size; i++) {
        test_output(show_passing, iht_->contains(i),
                    HT_Res<int>(FALSE_STATE, 0),
                    std::string("Contains ") + std::to_string(i) +
                        std::string(" maintains false"));
      }
      ROME_INFO("All test cases finished");
    } else {
      ROME_INFO("Starting test cases.");
      test_output(true, iht_->contains(5), HT_Res<int>(FALSE_STATE, 0),
                  "Contains 5");
      test_output(true, iht_->contains(4), HT_Res<int>(FALSE_STATE, 0),
                  "Contains 4");
      test_output(true, iht_->insert(5, 10), HT_Res<int>(FALSE_STATE, 0),
                  "Insert 5");
      test_output(true, iht_->insert(5, 11), HT_Res<int>(FALSE_STATE, 10),
                  "Insert 5 again should fail");
      test_output(true, iht_->contains(5), HT_Res<int>(TRUE_STATE, 10),
                  "Contains 5");
      test_output(true, iht_->contains(4), HT_Res<int>(FALSE_STATE, 0),
                  "Contains 4");
      test_output(true, iht_->remove(5), HT_Res<int>(TRUE_STATE, 10),
                  "Remove 5");
      test_output(true, iht_->remove(4), HT_Res<int>(FALSE_STATE, 0),
                  "Remove 4");
      test_output(true, iht_->contains(5), HT_Res<int>(FALSE_STATE, 0),
                  "Contains 5");
      test_output(true, iht_->contains(4), HT_Res<int>(FALSE_STATE, 0),
                  "Contains 4");
      ROME_INFO("All cases finished");
    }
    auto stop_status = Stop();
    OK_OR_FAIL(stop_status);
    return sss::Status::Ok();
  }

  // A function for communicating with the server that we are done. Will wait
  // until server says it is ok to shut down
  //
  // [mfs]  This is really just trying to create a Barrier over RPC.  There's
  //        nothing wrong with that, in principle, but if all we really need is
  //        a barrier, then why not just make a barrier?
  sss::Status Stop() {
    ROME_INFO("CLIENT :: Stopping client...");
    if (!master_client_) {
      // if we aren't the master client we don't need to do the stop sequence.
      // Just arrive at the barrier
      //
      // [mfs]  Why doesn't the master client also arrive at the barrier, before
      //        acking to the lead node?
      if (barrier_ != nullptr)
        barrier_->arrive_and_wait();
      return sss::Status::Ok();
    }
    if (host_.id == self_.id)
      return sss::Status::Ok(); // if we are the host, we don't need to do the
                                // stop sequence
    // send the ack to let the server know that we are done
    AckProto e;
    auto sent = iht_->pool_->Send(host_, e);
    ROME_INFO("CLIENT :: Sent Ack");

    // Wait to receive an ack back. Letting us know that the other clients are
    // done.
    auto msg = iht_->pool_->Recv<AckProto>(host_);
    // [mfs] The ACK might not be sss::Ok... is that acceptable?
    ROME_INFO("CLIENT :: Received Ack");

    // Return ok status
    return sss::Status::Ok();
  }

private:
  // [mfs]  This all needs to be documented

  Client(const rome::rdma::Peer &self, const rome::rdma::Peer &host,
         const std::vector<rome::rdma::Peer> &peers, ExperimentParams &params,
         std::barrier<> *barrier, IHT *iht, bool master_client)
      : self_(self), host_(host), peers_(peers), params_(params),
        barrier_(barrier), iht_(iht), master_client_(master_client) {
    if (params.unlimited_stream())
      progression = 10000;
    else
      progression = params_.op_count() * 0.001;
  }

  int count = 0;

  const rome::rdma::Peer self_;
  const rome::rdma::Peer host_;
  std::vector<rome::rdma::Peer> peers_;
  const ExperimentParams params_;
  std::barrier<> *barrier_;
  IHT *iht_;
  bool master_client_;

  // [mfs]  This one in particular needs more explanation.  It looks like it is
  //        for giving some "heartbeat" output during the experiment, which
  //        could be costly.
  int progression;
};