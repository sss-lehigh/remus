#include "../rdma/memory_pool.h"

#include <google/protobuf/text_format.h>
#include <protos/experiment.pb.h>
#include <protos/workloaddriver.pb.h>
#include <vector>

#include "../logging/logging.h"
#include "../vendor/sss/cli.h"

#include "../rdma/rdma.h"

#include "role_client.h"
#include "role_server.h"
#include "structures/iht_ds.h"

// [mfs] TODO: Avoid needing this?
using IHT = RdmaIHT<int, int, 16, 1024>;

/// Declare the command-line arguments for this program
const auto ARGS = {
    sss::STR_ARG_OPT("--experiment_params", "Experimental parameters", ""),
    sss::BOOL_ARG_OPT("--send_bulk",
                      "If to run bulk operations. (More for benchmarking)"),
    sss::BOOL_ARG_OPT("--send_test",
                      "If to test the functionality of the methods."),
    sss::BOOL_ARG_OPT("--send_exp", "If to run an experiment"),
};

// [mfs] Why are these hard-coded?
constexpr char iphost[] = "node0";
constexpr uint16_t portNum = 18000;

int main(int argc, char **argv) {
  ROME_INIT_LOG();

  sss::ArgMap args;
  // import_args will validate that the newly added args don't conflict with
  // those already added.
  auto res = args.import_args(ARGS);
  if (res) {
    ROME_ERROR(res.value());
    exit(1);
  }
  // NB: Only call parse_args once.  If it fails, a mandatory arg was skipped
  res = args.parse_args(argc, argv);
  if (res) {
    args.usage();
    ROME_ERROR(res.value());
    exit(1);
  }

  // Extract the args to variables
  bool bulk_operations = args.bget("--send_bulk");
  bool test_operations = args.bget("--send_test");
  bool do_exp = args.bget("--send_exp");
  std::string experiment_parms = args.sget("--experiment_params");

  // Use the args to set up the experiment
  //
  // [mfs]  There is no documentation for ExperimentParams.  This means that the
  //        only way to run the code is via a script that knows the internals of
  //        the ExperimentParams proto.  This needs to be refactored.  It is
  //        especially strange because protobufs are usually meant for
  //        communication, not for convenient merging of structs.
  ExperimentParams params = ExperimentParams();
  bool success =
      google::protobuf::TextFormat::MergeFromString(experiment_parms, &params);
  ROME_ASSERT(success, "Couldn't parse protobuf");
  ROME_ASSERT(params.node_count() > 0,
              "Cannot start experiment. Node count must be > 0");

  // Get hostname of this process
  char hostname[4096];
  gethostname(hostname, 4096);

  // Start initializing a vector of peers
  //
  // [mfs]  This is pushing the hard-coded info about the host into the vector.
  //        It should really not be hard-coded.
  rome::rdma::Peer host{0, std::string(iphost), portNum};
  std::vector<rome::rdma::Peer> peers;
  peers.push_back(host);

  // Set values if we are host machine as well
  //
  // [mfs]  This is a hack.  The host machine should be part of the config
  //
  // [mfs]  The use of "outside_exp" doesn't really make sense.  It seems like
  //        it is validating that the node is part of the experiment, but it's
  //        not clear what conditions would lead to it *not* being part of the
  //        experiment (unless the configuration was wrong).
  //
  // [mfs]  If this check really is needed, then consider using
  //        optional<MemoryPool>, to avoid the unnecessary extra variable.
  rome::rdma::Peer self;
  bool outside_exp = true;
  if (hostname[4] == '0') {
    self = host;
    outside_exp = false;
  }

  // Make the peer list by iterating through the node count
  //
  // [mfs]  This use of do_exp is strange.  Why not have different functions for
  //        each of the execution modes?
  for (int n = 1; n < params.node_count() && do_exp; n++) {
    // Create the ip_peer (really just node name)
    //
    // [mfs] Again, the hard-coded names and ports are concerning
    std::string ippeer = "node";
    std::string node_id = std::to_string(n);
    ippeer.append(node_id);
    // Create the peer and add it to the list
    rome::rdma::Peer next{static_cast<uint16_t>(n), ippeer, portNum};
    peers.push_back(next);
    // Compare after 4th character to node_id
    if (strncmp(hostname + 4, node_id.c_str(), node_id.length()) == 0) {
      // If matching, next is self
      self = next;
      outside_exp = false;
    }
  }

  // Test for an invalid configuration at this node
  //
  // [mfs]  See above... how is this possible without the whole experiment being
  //        invalid?
  if (outside_exp) {
    // [mfs] Sleeping is a bad way to synchronize
    std::this_thread::sleep_for(
        std::chrono::milliseconds(200)); // So we get this printed last
    ROME_INFO("Not in experiment. Shutting down");
    return 0;
  }

  // Print the IPs of the peers of this node
  //
  // [mfs]  I would prefer some kind of explicit handshaking, instead of using
  //        the node names.
  for (int i = 0; i < peers.size(); i++) {
    ROME_INFO("Peer list {}:{}@{}", i, peers.at(i).id, peers.at(i).address);
  }

  // Initialize our capability for interacting with ROME
  uint32_t block_size = 1 << params.region_size();
  rome::rdma::rdma_capability ROME(self);
  ROME.init_pool(block_size, peers);

  // Put an IHT into the memory pool
  //
  // [mfs]  This should be done differently.  `iht` should be a remote pointer,
  //        distributed by the lead node.
  //
  // [mfs]  Right now, this is TestMap?  Switch to IHT?  Or define it in this
  //        file...
  IHT iht = IHT(self, &ROME);
  auto status_iht = iht.Init(host, peers);
  OK_OR_FAIL(status_iht);

  std::vector<std::thread> threads;
  volatile bool done = false;
  if (hostname[4] == '0') {
    // If dedicated server-node, we must start the server
    //
    // [mfs]  I don't understand this.  What is the job of the server, relative
    //        to the other nodes?  Why one thread?
    threads.emplace_back(std::thread([&]() {
      // We are the server
      std::unique_ptr<Server> server =
          Server::Create(host, peers, params, &ROME);
      ROME_INFO("Server Created");
      // [mfs] What are these params to Launch?
      auto run_status = server->Launch(&done, params.runtime(),
                                       [&iht]() { iht.try_rehash(); });
      OK_OR_FAIL(run_status);
      ROME_INFO("[SERVER THREAD] -- End of execution; -- ");
    }));
  }

  using Operation = IHT_Op<int, int>;

  // [mfs] This is a bit odd.  Why wouldn't this be a separate function?
  if (!do_exp) {
    // Not doing experiment, so just create some test clients
    std::unique_ptr<Client<Operation>> client = Client<Operation>::Create(
        self, host, peers, params, nullptr, &iht, true);
    if (bulk_operations) {
      auto status = client->Operations(true);
      OK_OR_FAIL(status);
    } else if (test_operations) {
      auto status = client->Operations(false);
      OK_OR_FAIL(status);
    }
    // Wait for server to complete
    done = true;
    threads[0].join();
    ROME_INFO("[TEST] -- End of execution; -- ");
    exit(0);
  }

  // Barrier to start all the clients at the same time
  //
  // [mfs]  This starts all the clients *on this thread*, but that's not really
  //        a sufficient barrier.  A tree barrier is needed, to coordinate
  //        across nodes.
  std::barrier client_sync = std::barrier(params.thread_count());
  // [mfs]  This seems like a misuse of protobufs: why would the local threads
  //        communicate via protobufs?
  rome::WorkloadDriverProto results[params.thread_count()];
  for (int n = 0; n < params.thread_count(); n++) {
    // Add the thread
    threads.emplace_back(std::thread(
        [&](int index) {
          // Create and run a client in a thread
          std::unique_ptr<Client<Operation>> client = Client<Operation>::Create(
              self, host, peers, params, &client_sync, &iht, index == 0);
          // [mfs]  I would have thought that the `done` flag would be set by
          //        the main thread, as a way to tell all clients to stop?
          auto output = Client<Operation>::Run(
              std::move(client), &done, 0.5 / (double)params.node_count());
          // [mfs]  It would be good to document how a client can fail, because
          //        it seems like if even one client fails, on any machine, the
          //        whole experiment should be invalidated.
          if (output.status.t == sss::Ok) {
            results[index] = output.val.value();
          } else {
            ROME_ERROR("Client run failed");
          }
          ROME_INFO("[CLIENT THREAD] -- End of execution; -- ");
        },
        n));
  }

  // Join all threads
  int i = 0;
  for (auto it = threads.begin(); it != threads.end(); it++) {
    // [mfs]  This seems like a bad message.  It doesn't say which node.  Why
    //        not have one at the end?  And where's the error checking on the
    //        returns from the clients?
    ROME_INFO("Syncing {}", ++i);
    auto t = it;
    t->join();
  }

  // [mfs]  Again, odd use of protobufs for relatively straightforward combining
  //        of results.  Or am I missing something, and each node is sending its
  //        results, so they are all accumulated at the main node?
  ResultProto result_proto = ResultProto();
  *result_proto.mutable_params() = params;
  for (int i = 0; i < params.thread_count(); i++) {
    rome::WorkloadDriverProto *r = result_proto.add_driver();
    std::string output;
    results[i].SerializeToString(&output);
    r->MergeFromString(output);
  }

  // [mfs]  Does this produce one file per node?
  ROME_INFO("Compiled Proto Results ### {}", result_proto.DebugString());
  std::ofstream filestream("iht_result.pbtxt");
  filestream << result_proto.DebugString();
  filestream.flush(); // [mfs] I thought streams auto-flushed on close?
  filestream.close();

  ROME_INFO("[EXPERIMENT] -- End of execution; -- ");
  return 0;
}
