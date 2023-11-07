#include <barrier>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <random>

#include "../colosseum/workload_driver.h"
#include "../logging/logging.h"
#include "../rdma/rdma.h"

#include "common.h"
#include "protos/experiment.pb.h"
// #include "structures/hashtable.h"
#include "iht_ds.h"
#include "common.h"
// #include "structures/test_map.h"

using rome::WorkloadDriver;
using rome::WorkloadDriverProto;
using namespace rome::rdma;
using namespace std;

// [mfs] This should really be defined somewhere else
typedef RdmaIHT<int, int, CNF_ELIST_SIZE, CNF_PLIST_SIZE> IHT;

// Function to run a test case (will return a success code)
inline bool test_output(bool show_passing, optional<int> actual, optional<int> expected, string message) {
  if (actual.has_value() != expected.has_value() && actual.value_or(0) != expected.value_or(0)) {
    ROME_INFO("[-] {} func():(Has Value {}=>{}) != expected:(Has Value {}=>{})", message, actual.has_value(), actual.value_or(0),
              expected.has_value(), expected.value_or(0));
    return false;
  } else if (show_passing) {
    ROME_INFO("[+] Test Case {} Passed!", message);
  }
  return true;
}

template <class Operation> class Client {
  // static_assert(::rome::IsClientAdapter<Client, Operation>);

public:
  // [mfs]  Here and in Server, I don't understand the factory pattern.  It's
  //        not really adding any value.
  // [esl]  I think Jacob was trying to force the users of Client to make a unique_ptr? 
  //        It was the pattern I observed, so it felt safer to just follow it haha
  /// @brief Force the creation of a unique ptr to a client instance
  /// @param server the "server"-peer that is responsible for coordination among clients
  /// @param ep a EndpointManager instance that can be owned by the client.
  /// @param params the experiment parameters
  /// @param barr a barrier to synchonize local clients
  /// @param iht a pointer to an IHT
  /// @return a unique ptr
  static unique_ptr<Client>
  Create(const Peer &server, tcp::EndpointManager &ep, ExperimentParams& params, barrier<> *barr, unique_ptr<IHT> iht, shared_ptr<rdma_capability> pool) {
    return unique_ptr<Client>(new Client(server, ep, params, barr, std::move(iht), pool));
  }

  /// @brief Run the client
  /// @param client the client instance to run with
  /// @param thread_id a thread index to use for seeding the random number generation
  /// @param frac if 0, won't populate. Otherwise, will do this fraction of the
  /// population
  /// @return the resultproto
  static sss::StatusVal<rome::WorkloadDriverProto>
  Run(unique_ptr<Client> client, int thread_id, double frac) {
    // [mfs]  I was hopeful that this code was going to actually populate the
    //        data structure from *multiple nodes* simultaneously.  It should,
    //        or else all of the initial elists and plists are going to be on
    //        the same machine, which probably means all of the elists and
    //        plists will always be on the same machine.
    // [esl]  A remote barrier is defintely needed to make sure this all happens at the same time...
    int key_lb = client->params_.key_lb(), key_ub = client->params_.key_ub();
    int op_count = (key_ub - key_lb) * frac;
    ROME_INFO("CLIENT :: Data structure ({}%) is being populated ({} items inserted) by this client", frac * 100, op_count);
    client->pool_->RegisterThread();
    // arrive at the barrier so we are populating in sync with local clients -- TODO: replace for remote barrier
    client->barrier_->arrive_and_wait();
    client->iht_->populate(client->pool_, op_count, key_lb, key_ub, [=](int key){ return key; });
    ROME_DEBUG("CLIENT :: Done with populate!");
    // TODO: Sleeping for 1 second to account for difference between remote
    // client start times. Must fix this in the future to a better solution
    // The idea is even though remote nodes won't be starting a workload at the same
    // time, at least the data structure is roughly guaranteed to be populated
    //
    // [mfs] Indeed, this indicates the need for a distributed barrier
    // [esl] I'm not sure what the design for a distributed barrier over RDMA would look like
    //       But I would be interested in creating one so everyone can use it.
    this_thread::sleep_for(chrono::seconds(1));
    
    // TODO: Signal Handler
    // signal(SIGINT, signal_handler);

    // auto *client_ptr = client.get();
    std::vector<Operation> operations = std::vector<Operation>();

    // initialize random number generator and key_range
    int key_range = client->params_.key_ub() - client->params_.key_lb();

    // Create a random operation generator that is
    // - evenly distributed among the key range
    // - within the specified ratios for operations
    uniform_int_distribution<int> op_dist = uniform_int_distribution<int>(1, 100);
    uniform_int_distribution<int> k_dist = uniform_int_distribution<int>(client->params_.key_lb(), client->params_.key_ub());

    // Ensuring each node has a different seed value
    default_random_engine gen(client->params_.node_id() * client->params_.thread_count() + thread_id);
    int lb = client->params_.key_lb();
    int contains = client->params_.contains();
    int insert = client->params_.insert();
    function<Operation(void)> generator = [&]() {
      int rng = op_dist(gen);
      int k = k_dist(gen);
      if (rng <= contains) { 
        // between 0 and CONTAINS
        return Operation(CONTAINS, k, 0);
      } else if (rng <= contains + insert) { 
        // between CONTAINS and CONTAINS + INSERT
        return Operation(INSERT, k, k);
      } else {
        return Operation(REMOVE, k, 0);
      }
    };

    // Generate two streams based on what the user wants (operation count or
    // timed stream)
    unique_ptr<rome::Stream<Operation>> workload_stream;
    if (client->params_.unlimited_stream()) {
      workload_stream = make_unique<rome::EndlessStream<Operation>>(generator);
    } else {
      // Deliver a workload
      workload_stream = make_unique<rome::FixedLengthStream<Operation>>(generator, client->params_.op_count());
    }

    // Create and start the workload driver (also starts client and lets it
    // run).
    int32_t runtime = client->params_.runtime();
    int32_t qps_sample_rate = client->params_.qps_sample_rate();
    barrier<> *barr = client->barrier_;
    bool unlimited_stream = client->params_.unlimited_stream();

    // [mfs] Again, it looks like Create() is an unnecessary factory
    auto driver = rome::WorkloadDriver<Client, Operation>::Create(
        std::move(client), std::move(workload_stream),
        std::chrono::milliseconds(qps_sample_rate));
    // [mfs]  This is quite odd.  The current thread is invoking an async thread
    //        to actually do the work, which means we have lots of extra thread
    //        creation and joining.
    // [esl]  I am not a huge fan of this WorkloadDriver. The concept is cool 
    //        and useful, but feels wrong in implementation
    OK_OR_FAIL(driver->Start());
    // TODO: The workload driver has no way for me to sleep on a condition variable and wait on the result
    // TODO: It would be nice to redesign the dependency anyways, so maybe this won't need to be fixed
    this_thread::sleep_for(chrono::seconds(runtime));
    ROME_DEBUG("Done here, stop sequence");
    // Wait for all the clients to stop. Then set the done to true to release
    // the server
    if (barr != nullptr) {
      barr->arrive_and_wait();
    }
    // [mfs] If we didn't use WorkloadDriver, then we wouldn't need Stop()
    OK_OR_FAIL(driver->Stop());
    ROME_INFO("CLIENT :: Driver generated {}", driver->ToString());
    // [mfs]  It seems like these protos aren't being sent across machines.  Are
    //        they really needed?
    // [esl]  TODO: They are used by the workload driver. It was easier to live with
    //        then to spend the time to refactor, which is why they haven't been changed yet. 
    //        There probably needs to be a class for storing the result of an experiment.
    return {sss::Status::Ok(), driver->ToProto()};
  }

  // Start the client
  sss::Status Start() {
    ROME_INFO("CLIENT :: Starting client...");
    pool_->RegisterThread(); // TODO? REMOVE?
    // [mfs]  The entire barrier infrastructure is odd.  Nobody is using it to
    //        know when to get time, and it's completely per-node.
    // [esl]  I think the Workload driver gets time, which is why I think its a good idea to synchronize the threads
    //        You make a good point, synchronizing among nodes would be good 
    if (barrier_ != nullptr)
      barrier_->arrive_and_wait();
    return sss::Status::Ok();
  }

  // Runs the next operation
  sss::Status Apply(const Operation &op) {
    count++;
    optional<int> res;
    switch (op.op_type) {
    case (CONTAINS):
      // [mfs]  I don't understand the purpose of "progression".  Is it just for
      //        getting periodic output?  If so, it's going to hurt the
      //        experiment's latency, so it's probably a bad idea.
      // [esl]  Periodic output helps me determine faster if my code is still running or if I've deadlocked
      //        Changing it to ROME_DEBUG to try and avoid hurting latency...
      if (count % progression == 0) {
        ROME_DEBUG("Running Operation {}: contains({})", count, op.key);
      }
      res = iht_->contains(pool_, op.key);
      if (res.has_value()) {
        ROME_ASSERT(res.value() == op.key, "Invalid result of contains operation {}!={}", res.value(), op.key);
      }
      break;
    case (INSERT):
      if (count % progression == 0){
        ROME_DEBUG("Running Operation {}: insert({}, {})", count, op.key, op.value);
      }
      res = iht_->insert(pool_, op.key, op.value);
      if (res.has_value()){
        ROME_ASSERT(res.value() == op.key, "Invalid result of insert operation {}!={}", res.value(), op.key);
      }
      break;
    case (REMOVE):
      if (count % progression == 0){
        ROME_DEBUG("Running Operation {}: remove({})", count, op.key);
      }
      res = iht_->remove(pool_, op.key);
      if (res.has_value()){
        ROME_ASSERT(res.value() == op.key, "Invalid result of remove operation {}!={}", res.value(), op.key);
      }
      break;
    default:
      // if we get something other than a contains, insert, or remove, the program probably should die
      ROME_FATAL("Expected CONTAINS, INSERT, or REMOVE operation.");
      break;
    }
    return sss::Status::Ok();
  }

  // A function for communicating with the server that we are done. Will wait
  // until server says it is ok to shut down
  //
  // [mfs]  This is really just trying to create a Barrier over RPC.  There's
  //        nothing wrong with that, in principle, but if all we really need is
  //        a barrier, then why not just make a barrier?
  sss::Status Stop() {
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
  /// @param endpoint a EndpointManager instance that can be owned by the client.
  /// @param params the experiment parameters
  /// @param barrier a barrier to synchonize local clients
  /// @param iht a pointer to an IHT
  /// @param pool the memory pool capability for the IHT
  /// @return a unique ptr
  Client(const Peer &host, tcp::EndpointManager &ep, ExperimentParams &params, barrier<> *barr, unique_ptr<IHT> iht, shared_ptr<rdma_capability> pool)
    : host_(host), endpoint_(ep), params_(params), barrier_(barr), iht_(std::move(iht)), pool_(pool) {
      if (params.unlimited_stream()) progression = 10000;
      else progression = max(20.0, params_.op_count() * params_.thread_count() * 0.001);
    }

  int count = 0;

  /// @brief Represents the host peer
  const Peer host_;
  /// @brief Represents an endpoint to be used for communication with the host peer
  tcp::EndpointManager endpoint_;
  /// @brief Experimental parameters
  const ExperimentParams params_;
  /// @brief a barrier for syncing amount clients locally
  barrier<> *barrier_;
  /// @brief an IHT instance to use
  unique_ptr<IHT> iht_;
  /// @brief the memorypool to attribute to the IHT
  shared_ptr<rdma_capability> pool_;

  /// @brief The number of operations to do before debug-printing the number of completed operations
  /// This is useful in debugging since I can see around how many operations have been done (if at all) before crashing
  int progression;
};