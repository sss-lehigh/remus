#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <google/protobuf/text_format.h>

#include "../logging/logging.h"
#include "../rdma/connection_manager/connection_manager.h"
#include "../rdma/memory_pool/memory_pool.h"
// #include "../util/proto_util.h"

#include "protos/experiment.pb.h"

#include "role_client.h"
#include "role_server.h"

ABSL_FLAG(std::string, experiment_params, "", "Experimental parameters");
ABSL_FLAG(bool, send_bulk, false,
          "If to run bulk operations. (More for benchmarking)");
ABSL_FLAG(bool, send_test, false,
          "If to test the functionality of the methods.");
ABSL_FLAG(bool, send_exp, false, "If to run an experiment");

#define PATH_MAX 4096

using ::rome::rdma::ConnectionManager;
using ::rome::rdma::MemoryPool;

constexpr char iphost[] = "node0";
constexpr uint16_t portNum = 18000;

using cm_type = MemoryPool::cm_type;

int main(int argc, char **argv) {
  ROME_INIT_LOG();

  absl::ParseCommandLine(argc, argv);
  bool bulk_operations = absl::GetFlag(FLAGS_send_bulk);
  bool test_operations = absl::GetFlag(FLAGS_send_test);
  bool do_exp = absl::GetFlag(FLAGS_send_exp);
  ExperimentParams params = ExperimentParams();
  ResultProto result_proto = ResultProto();
  std::string experiment_parms = absl::GetFlag(FLAGS_experiment_params);
  bool success =
      google::protobuf::TextFormat::MergeFromString(experiment_parms, &params);
  ROME_ASSERT(success, "Couldn't parse protobuf");

  // Get hostname to determine who we are
  char hostname[4096];
  gethostname(hostname, 4096);

  // Start initializing a vector of peers
  volatile bool done = false;
  MemoryPool::Peer host{0, std::string(iphost), portNum};
  MemoryPool::Peer self;
  bool outside_exp = true;
  std::vector<MemoryPool::Peer> peers;
  peers.push_back(host);

  if (params.node_count() == 0) {
    ROME_INFO("Cannot start experiment. Node count was found to be 0");
    exit(1);
  }

  // Set values if we are host machine as well
  if (hostname[4] == '0') {
    self = host;
    outside_exp = false;
  }

  // Make the peer list by iterating through the node count
  for (int n = 1; n < params.node_count() && do_exp; n++) {
    // Create the ip_peer (really just node name)
    std::string ippeer = "node";
    std::string node_id = std::to_string(n);
    ippeer.append(node_id);
    // Create the peer and add it to the list
    MemoryPool::Peer next{static_cast<uint16_t>(n), ippeer, portNum};
    peers.push_back(next);
    // Compare after 4th character to node_id
    if (strncmp(hostname + 4, node_id.c_str(), node_id.length()) == 0) {
      // If matching, next is self
      self = next;
      outside_exp = false;
    }
  }

  if (outside_exp) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(200)); // So we get this printed last
    ROME_INFO("Not in experiment. Shutting down");
    return 0;
  }

  // Make a memory pool for the node to share among all client instances
  uint32_t block_size = 1 << params.region_size();
  MemoryPool pool =
      MemoryPool(self, std::make_unique<MemoryPool::cm_type>(self.id));

  for (int i = 0; i < peers.size(); i++) {
    ROME_INFO("Peer list {}:{}@{}", i, peers.at(i).id, peers.at(i).address);
  }

  auto status_pool = pool.Init(block_size, peers);
  ROME_ASSERT_OK(status_pool);
  ROME_INFO("Created memory pool");

  IHT iht = IHT(self, &pool);
  auto status_iht = iht.Init(host, peers);
  ROME_ASSERT_OK(status_iht);

  std::vector<std::thread> threads;
  if (hostname[4] == '0') {
    // If dedicated server-node, we must start the server
    threads.emplace_back(std::thread([&]() {
      // We are the server
      std::unique_ptr<Server> server =
          Server::Create(host, peers, params, &pool);
      ROME_INFO("Server Created");
      auto run_status = server->Launch(&done, params.runtime(),
                                       [&iht]() { iht.try_rehash(); });
      ROME_ASSERT_OK(run_status);
      ROME_INFO("[SERVER THREAD] -- End of execution; -- ");
    }));
  }

  if (!do_exp) {
    // Not doing experiment, so just create some test clients
    std::unique_ptr<Client> client =
        Client::Create(self, host, peers, params, nullptr, &iht, true);
    if (bulk_operations) {
      auto status = client->Operations(true);
      ROME_ASSERT_OK(status);
    } else if (test_operations) {
      auto status = client->Operations(false);
      ROME_ASSERT_OK(status);
    }
    // Wait for server to complete
    done = true;
    threads[0].join();
    ROME_INFO("[TEST] -- End of execution; -- ");
    exit(0);
  }

  // Barrier to start all the clients at the same time
  std::barrier client_sync = std::barrier(params.thread_count());
  WorkloadDriverProto results[params.thread_count()];
  for (int n = 0; n < params.thread_count(); n++) {
    // Add the thread
    threads.emplace_back(std::thread(
        [&](int index) {
          // Create and run a client in a thread
          std::unique_ptr<Client> client = Client::Create(
              self, host, peers, params, &client_sync, &iht, index == 0);
          auto output = Client::Run(std::move(client), &done,
                                    0.5 / (double)params.node_count());
          if (output.ok()) {
            results[index] = output.value();
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
    ROME_INFO("Syncing {}", ++i);
    auto t = it;
    t->join();
  }

  *result_proto.mutable_params() = params;

  for (int i = 0; i < params.thread_count(); i++) {
    IHTWorkloadDriverProto *r = result_proto.add_driver();
    std::string output;
    results[i].SerializeToString(&output);
    r->MergeFromString(output);
  }

  ROME_INFO("Compiled Proto Results ### {}", result_proto.DebugString());

  std::ofstream filestream("iht_result.pbtxt");
  filestream << result_proto.DebugString();
  filestream.flush();
  filestream.close();

  ROME_INFO("[EXPERIMENT] -- End of execution; -- ");
  return 0;
}
