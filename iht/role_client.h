#pragma once

#include <barrier>
#include <chrono>
#include <cstdlib>

#include "../colosseum/client_adaptor.h"
#include "../colosseum/qps_controller.h"
#include "../colosseum/streams/streams.h"
#include "../colosseum/workload_driver.h"
#include "../rdma/connection_manager/connection_manager.h"
#include "../rdma/memory_pool/memory_pool.h"
#include "../util/clocks.h"

#include "common.h"
#include "protos/experiment.pb.h"
// #include "structures/hashtable.h"
#include "structures/iht_ds.h"
// #include "structures/test_map.h"

using ::rome::ClientAdaptor;
using ::rome::WorkloadDriver;
using ::rome::WorkloadDriverProto;
using ::rome::rdma::MemoryPool;

// [mfs] This should really be defined somewhere else
typedef RdmaIHT<int, int, 16, 1024> IHT;

std::string fromStateValue(state_value value) {
  if (FALSE_STATE == value) {
    return std::string("FALSE");
  } else if (TRUE_STATE == value) {
    return std::string("TRUE");
  } else if (REHASH_DELETED == value) {
    return std::string("REHASH DELETED");
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

// [mfs] This is declared at the wrong scope?
typedef IHT_Op<int, int> Operation;

class Client : public ClientAdaptor<Operation> {
public:
  // [mfs]  Here and in Server, I don't understand the factory pattern.  It's
  //        not really adding any value.
  // [esl]  To be honest, it was mainly a monkey-see monkey-do situation
  //        I felt like since the Client inherited from the ClientAdaptor, 
  //        I had to follow the pattern I saw in other examples.
  /// @brief Force the creation of a unique ptr to a client instance
  /// @param server the "server"-peer that is responsible for coordination among clients
  /// @param endpoint a EndpointManager instance that can be owned by the client. TODO: replace for unique ptr?
  /// @param params the experiment parameters
  /// @param barrier a barrier to synchonize local clients
  /// @param iht a pointer to an IHT
  /// @return a unique ptr
  static std::unique_ptr<Client>
  Create(const MemoryPool::Peer &server, tcp::EndpointManager &endpoint, ExperimentParams& params, std::barrier<> *barrier, IHT* iht) {
    return std::unique_ptr<Client>(new Client(server, endpoint, params, barrier, iht));
  }

  /// @brief Run the client
  /// @param client the client instance to run with
  /// @param thread_id a thread index to use for seeding the random number generation
  /// @param frac if 0, won't populate. Otherwise, will do this fraction of the
  /// population
  /// @return the resultproto
  static sss::StatusVal<WorkloadDriverProto>
  Run(std::unique_ptr<Client> client, int thread_id, double frac) {
    // [mfs]  I was hopeful that this code was going to actually populate the
    //        data structure from *multiple nodes* simultaneously.  It should,
    //        or else all of the initial elists and plists are going to be on
    //        the same machine, which probably means all of the elists and
    //        plists will always be on the same machine.
    //
    // [mfs]  This should be in another function
    int key_lb = client->params_.key_lb(), key_ub = client->params_.key_ub();
    int op_count = (key_ub - key_lb) * frac;
    ROME_INFO("CLIENT :: Data structure ({}%) is being populated ({} items inserted) by this client", frac * 100, op_count);
    client->iht_->pool_->RegisterThread();
    client->iht_->populate(op_count, key_lb, key_ub, [=](int key){ return key; });
    ROME_DEBUG("CLIENT :: Done with populate!");
    // TODO: Sleeping for 1 second to account for difference between remote
    // client start times. Must fix this in the future to a better solution
    // The idea is even though remote nodes won't be starting a workload at the same
    // time, at least the data structure is roughly guaranteed to be populated
    //
    // [mfs] Indeed, this indicates the need for a distributed barrier
    // [esl] I'm not sure what the design for a distributed barrier over RDMA would look like
    // But I would be interested in creating one so everyone can use it
    std::this_thread::sleep_for(std::chrono::seconds(1));
    

    // TODO: Signal Handler
    // signal(SIGINT, signal_handler);

    // Setup qps_controller.
    std::unique_ptr<rome::LeakyTokenBucketQpsController<util::SystemClock>>
        qps_controller =
            rome::LeakyTokenBucketQpsController<util::SystemClock>::Create(
                client->params_.max_qps_second()); // what is the value here

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
    // different
    //        seed?  Also, is it really necessary to have a different seed for
    //        each trial, or should the seed be a function of the node / thread,
    //        for repeatability?
    // [esl]  TODO: Creating a random number generator using the node and thread id
    std::default_random_engine gen(client->params_.node_id() * client->params_.thread_count() + thread_id);
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
      // [esl]  TODO: Implement a FixedLengthStream
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

    // [mfs] Again, it looks like Create() is an unnecessary factory
    // [esl] It is, I wish the WorkloadDriver was a bit more simple to use
    auto driver = rome::WorkloadDriver<Operation>::Create(
        std::move(client), std::move(workload_stream), qps_controller.get(),
        std::chrono::milliseconds(qps_sample_rate));
    // [mfs]  This is quite odd.  The current thread is invoking an async thread
    //        to actually do the work, which means we have lots of extra thread
    //        creation and joining.
    // [esl]  I am not a huge fan of this WorkloadDriver. The concept is cool 
    //        and useful, but feels wrong in implementation
    OK_OR_FAIL(driver->Start());
    // [mfs]
    std::this_thread::sleep_for(std::chrono::seconds(runtime));
    ROME_DEBUG("Done here, stop sequence");
    // Wait for all the clients to stop. Then set the done to true to release
    // the server
    if (barr != nullptr) 
        barr->arrive_and_wait();
    // [mfs] If we didn't use WorkloadDriver, then we wouldn't need Stop()
    OK_OR_FAIL(driver->Stop());
    ROME_INFO("CLIENT :: Driver generated {}", driver->ToString());
    // [mfs]  It seems like these protos aren't being sent across machines.  Are
    //        they really needed?
    // [esl]  TODO: They are used by the workload driver. It was easier to live with
    //        then to spend the time to refactor. There probably needs to be a class for storing the result of an experiment
    return {sss::Status::Ok(), driver->ToProto()};
  }

  // Start the client
  sss::Status Start() override {
    ROME_INFO("CLIENT :: Starting client...");
    this->iht_->pool_->RegisterThread();
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
  sss::Status Apply(const Operation &op) override {
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
    // [esl]  TODO: Again another case of "monkey-see monkey-do"... when copying the client adaptor
    if (params_.has_think_time() && params_.think_time() != 0) {
      auto start = util::SystemClock::now();
      while (util::SystemClock::now() - start <
             std::chrono::nanoseconds(params_.think_time()))
        ;
    }
    return sss::Status::Ok();
  }

  // A function for communicating with the server that we are done. Will wait
  // until server says it is ok to shut down
  sss::Status Stop() override {
    ROME_DEBUG("CLIENT :: Stopping client...");

    // send the ack to let the server know that we are done
    tcp::message send_buffer;
    endpoint_.send_server(&send_buffer);
    ROME_DEBUG("CLIENT :: Sent Ack");

    // Wait to receive an ack back. Letting us know that the other clients are done.
    tcp::message recv_buffer;
    endpoint_.recv_server(&recv_buffer);
    ROME_DEBUG("CLIENT :: Received Ack");
    return sss::Status::Ok();
  }

private:
  /// @brief Private constructor of client
  /// @param server the "server"-peer that is responsible for coordination among clients
  /// @param endpoint a EndpointManager instance that can be owned by the client. TODO: replace for unique ptr?
  /// @param params the experiment parameters
  /// @param barrier a barrier to synchonize local clients
  /// @param iht a pointer to an IHT
  /// @return a unique ptr
  Client(const MemoryPool::Peer &host, tcp::EndpointManager &endpoint, ExperimentParams &params, std::barrier<> *barrier, IHT* iht)
    : host_(host), endpoint_(endpoint), params_(params), barrier_(barrier), iht_(iht) {
      if (params.unlimited_stream()) progression = 10000;
      else progression = params_.op_count() * 0.001;
    }

  int count = 0;

  /// @brief Represents the host peer
  const MemoryPool::Peer host_;
  /// @brief Represents an endpoint to be used for communication with the host peer
  tcp::EndpointManager endpoint_;
  /// @brief Experimental parameters
  const ExperimentParams params_;
  /// @brief a barrier for syncing amount clients locally
  std::barrier<> *barrier_;
  /// @brief an IHT instance to use
  IHT* iht_;

  /// @brief The number of operations to do before debug-printing the number of completed operations
  /// This is useful in debugging since I can see around how many operations have been done (if at all) before crashing
  int progression;
};