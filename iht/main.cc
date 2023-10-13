#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>

#include <google/protobuf/text_format.h>

#include "../logging/logging.h"
#include "../rdma/connection_manager/connection_manager.h"
#include "../rdma/memory_pool/memory_pool.h"
#include "../util/tcp/tcp.h"

#include "protos/colosseum.pb.h"

#include "role_client.h"
#include "role_server.h"
#include "context_manager.h"

#include "../vendor/sss/cli.h"

auto ARGS = {
    cli::STR_ARG_OPT("--experiment_params", "Experimental parameters", ""),
    cli::BOOL_ARG_OPT("--send_bulk",
                      "If to run bulk operations. (More for benchmarking)"),
    cli::BOOL_ARG_OPT("--send_test",
                      "If to test the functionality of the methods."),
    cli::BOOL_ARG_OPT("--send_exp", "If to run an experiment"),
};

#define PATH_MAX 4096
#define PORT_NUM 18000

using rome::rdma::ConnectionManager;
using rome::rdma::MemoryPool;
using cm_type = MemoryPool::cm_type;

// The optimial number of memory pools is mp=min(t, MAX_QP/n) where n is the number of nodes and t is the number of threads
// To distribute mp (memory pools) across t threads, it is best for t/mp to be a whole number


int main(int argc, char **argv) {
  ROME_INIT_LOG();

  cli::ArgMap args;
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
  // [esl]  TODO: replace for json?
  ExperimentParams params = ExperimentParams();
  bool success =
      google::protobuf::TextFormat::MergeFromString(experiment_parms, &params);
  ROME_ASSERT(success, "Couldn't parse protobuf");
  // Check node count
  if (params.node_count() <= 0 || params.thread_count() <= 0){
      ROME_INFO("Cannot start experiment. Node/thread count was found to be 0");
      exit(1);
  }
  // Check we are in this experiment
  if (params.node_id() >= params.node_count()){
      ROME_INFO("Not in this experiment. Exiting");
      exit(0);
  }

  // Determine the number of memory pools to use in the experiment
  // Each memory pool represents 
  int mp = std::min(params.thread_count(), (int) std::floor(params.qp_max() / params.node_count()));
  if (mp == 0) mp = 1; // Make sure if node_count > qp_max, we don't end up with 0 memory pools
    
  ROME_INFO("Distributing {} MemoryPools across {} threads", mp, params.thread_count());

  // Start initializing a vector of peers
  volatile bool done = false; // (TODO: Should be atomic?)
  std::vector<MemoryPool::Peer> peers;
  for(uint16_t n = 0; n < mp * params.node_count(); n++){
      // Create the ip_peer (really just node name)
      std::string ippeer = "node";
      std::string node_id = std::to_string((int) floor(n / mp));
      ippeer.append(node_id);
      // Create the peer and add it to the list
      MemoryPool::Peer next = MemoryPool::Peer(n, ippeer, PORT_NUM + n + 1);
      peers.push_back(next);
  }
  // Print the peers included in this experiment
  // This is just for debugging to ensure they are what you expect
  for(int i = 0; i < peers.size(); i++){
      ROME_DEBUG("Peer list {}:{}@{}", i, peers.at(i).id, peers.at(i).address);
  }
  MemoryPool::Peer host = peers.at(0);
  // Initialize memory pools into an array
  std::vector<std::thread> mempool_threads;
  MemoryPool* pools[mp];
  // Create multiple memory pools to be shared (have to use threads since Init is blocking)
  uint32_t block_size = 1 << params.region_size();
  for(int i = 0; i < mp; i++){
      mempool_threads.emplace_back(std::thread([&](int mp_index, int self_index){
          MemoryPool::Peer self = peers.at(self_index);
          ROME_INFO(mp != params.thread_count() ? "Is shared" : "Is not shared");
          MemoryPool* pool = new MemoryPool(self, std::make_unique<MemoryPool::cm_type>(self.id), mp != params.thread_count());
          sss::Status status_pool = pool->Init(block_size, peers);
          OK_OR_FAIL(status_pool);
          pools[mp_index] = pool;
      }, i, (params.node_id() * mp) + i));
  }
  // Let the init finish
  for(int i = 0; i < mp; i++){
      mempool_threads[i].join();
  }
  // Assign a context to free the memory allocated by the new memorypool calls
  ContextManger manager = ContextManger([&](){
      for(int i = 0; i < mp; i++){
          delete pools[i];
      }
  });

  // Create a list of client and server  threads
  std::vector<std::thread> threads;
  if (params.node_id() == 0){
      // If dedicated server-node, we must send IHT pointer and wait for clients to finish
      threads.emplace_back(std::thread([&](){
          // Initialize X connections
          tcp::SocketManager socket_handle = tcp::SocketManager(PORT_NUM);
          for(int i = 0; i < params.thread_count() * params.node_count(); i++){
              // TODO: Can we have a per-node connection?
              // I haven't gotten around to coming up with a clean way to reduce the number of sockets connected to the server
              socket_handle.accept_conn();
          }
          // Create a root ptr to the IHT
          IHT iht = IHT(host, pools[0]);
          remote_ptr<anon_ptr> root_ptr = iht.InitAsFirst();
          // Send the root pointer over
          tcp::message ptr_message = tcp::message(root_ptr.raw());
          socket_handle.send_to_all(&ptr_message);
          // We are the server
          ExperimentManager::ClientStopBarrier(socket_handle, params.runtime());
          for(int i = 0; i < mp; i++){
              // Stop the worker thread of the memory pool
              pools[i]->KillWorkerThread();
          }
          ROME_INFO("[SERVER THREAD] -- End of execution; -- ");
      }));
    }

  // Initialize T endpoints, one for each thread
  tcp::EndpointManager endpoint_managers[params.thread_count()];
  for(uint16_t i = 0; i < params.thread_count(); i++){
      endpoint_managers[i] = tcp::EndpointManager(PORT_NUM, host.address.c_str());
  }


  // Barrier to start all the clients at the same time
  //
  // [mfs]  This starts all the clients *on this thread*, but that's not really
  //        a sufficient barrier.  A tree barrier is needed, to coordinate
  //        across nodes.
  // [esl]  Is this something that would need to be implemented using rome?
  // I'm not exactly sure what the implementation of an RDMA barrier would look like. If you have one in mind, lmk and I can start working on it.
  std::barrier client_sync = std::barrier(params.thread_count());
  // [mfs]  This seems like a misuse of protobufs: why would the local threads
  //        communicate via protobufs?
  // [esl]  TODO: In the refactoring of the client adaptor, remove dependency on protobufs for a workload object
  WorkloadDriverProto results[params.thread_count()];
  for(int i = 0; i < params.thread_count(); i++){
      threads.emplace_back(std::thread([&](int thread_index){
          int mempool_index = thread_index % mp;
          MemoryPool* pool = pools[mempool_index];
          MemoryPool::Peer self = peers.at((params.node_id() * mp) + mempool_index);
          IHT iht = IHT(self, pool);
          // sleep for a short while to ensure the receiving end is up and running
          // NB: We have no way to ensure the server is running before connecting to it
          // In the future, having some kind of remote_barrier data structure would be much better
          std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
          // Get the data from the server to init the IHT
          tcp::message ptr_message;
          endpoint_managers[thread_index].recv_server(&ptr_message);
          iht.InitFromPointer(remote_ptr<anon_ptr>(ptr_message.get_first()));
          
          ROME_DEBUG("Creating client");
          // Create and run a client in a thread
          // [mfs]  I would have thought that the `done` flag would be set by
          // the main thread, as a way to tell all clients to stop?
          std::unique_ptr<Client> client = Client::Create(host, endpoint_managers[thread_index], params, &client_sync, &iht);
          sss::StatusVal<WorkloadDriverProto> output = Client::Run(std::move(client), 
                                                                   thread_index, 
                                                                   0.5 / (double) (params.node_count() * params.thread_count()));
          // [mfs]  It would be good to document how a client can fail, because
          // it seems like if even one client fails, on any machine, the
          //  whole experiment should be invalidated.
          if (output.status.t == sss::StatusType::Ok && output.val.has_value()){
              results[thread_index] = output.val.value();
          } else {
              ROME_ERROR("Client run failed");
          }
          ROME_INFO("[CLIENT THREAD] -- End of execution; -- ");
      }, i));
  }

  // Join all threads
  int i = 0;
  for (auto it = threads.begin(); it != threads.end(); it++){
      // For debug purposes, sometimes it helps to see which threads haven't deadlocked
      ROME_DEBUG("Syncing {}", ++i);
      auto t = it;
      t->join();
  }
  // [mfs]  Again, odd use of protobufs for relatively straightforward combining
  //        of results.  Or am I missing something, and each node is sending its
  //        results, so they are all accumulated at the main node?
  ResultProto result_proto = ResultProto();
  *result_proto.mutable_params() = params;
  for (int i = 0; i < params.thread_count(); i++) {
    WorkloadDriverProto *r = result_proto.add_driver();
    std::string output;
    results[i].SerializeToString(&output);
    r->MergeFromString(output);
  }

  // [mfs]  Does this produce one file per node?
  // [esl] Yes, this produces one file per node, 
  // The launch.py script will scp this file and use the protobuf to interpret it
  // TODO: Replace this file out for a simple CSV --> much easier to parse on the developer side and a lot more code friendly
  ROME_DEBUG("Compiled Proto Results ### {}", result_proto.DebugString());
  std::ofstream filestream("iht_result.pbtxt");
  filestream << result_proto.DebugString();
  filestream.close();

  ROME_INFO("[EXPERIMENT] -- End of execution; -- ");
  return 0;
}
