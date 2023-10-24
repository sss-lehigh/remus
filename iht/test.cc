#include <fstream>
#include <google/protobuf/message.h>
#include <iostream>
#include <memory>
#include <optional>
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
#include "protos/experiment.pb.h"

#include "role_client.h"
#include "role_server.h"
#include "context_manager.h"

#include "../vendor/sss/cli.h"

auto ARGS = {
    cli::BOOL_ARG_OPT("--send_bulk",
                      "If to run bulk operations. (More for benchmarking)"),
    cli::BOOL_ARG_OPT("--send_test",
                      "If to test the functionality of the methods."),
};

#define PATH_MAX 4096
#define PORT_NUM 18000

using rome::rdma::MemoryPool;
using cm_type = MemoryPool::cm_type;

// The optimial number of memory pools is mp=min(t, MAX_QP/n) where n is the number of nodes and t is the number of threads
// To distribute mp (memory pools) across t threads, it is best for t/mp to be a whole number

int main(int argc, char** argv){
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
    ROME_ASSERT((bulk_operations ^ test_operations) == 1, "Assert one flag (bulk or test) is used"); // assert one or the other

    // Create a single peer
    volatile bool done = false; // (Should be atomic?)
    MemoryPool::Peer host = MemoryPool::Peer(0, "node0", PORT_NUM + 1);
    std::vector<MemoryPool::Peer> peer_list = std::vector<MemoryPool::Peer>(0);
    peer_list.push_back(host);
    // Initialize a memory pool
    std::vector<std::thread> mempool_threads;
    MemoryPool* pool = new MemoryPool(host, std::make_unique<MemoryPool::cm_type>(host.id), false);
    uint32_t block_size = 1 << 24;
    sss::Status status_pool = pool->Init(block_size, peer_list);
    OK_OR_FAIL(status_pool);
    pool->RegisterThread();

    // Create an iht
    IHT* iht_ = new IHT(host, pool);
    ContextManger manager = ContextManger([&](){
        delete pool;
        delete iht_;
    });
    iht_->InitAsFirst();
    if (bulk_operations){
      int scale_size = (CNF_PLIST_SIZE * CNF_ELIST_SIZE) * 2;
      ROME_INFO("Scale is {}", scale_size);
      bool show_passing = false;
      for(int i = 0; i < scale_size; i++){
        test_output(show_passing, iht_->contains(i), std::nullopt, std::string("Contains ") + std::to_string(i) + std::string(" false"));
        test_output(show_passing, iht_->insert(i, i), std::nullopt, std::string("Insert ") + std::to_string(i));
        test_output(show_passing, iht_->contains(i), std::make_optional(i), std::string("Contains ") + std::to_string(i) + std::string(" true"));
      }
      ROME_INFO(" = 25% Finished = ");
      for(int i = 0; i < scale_size; i++){
        test_output(show_passing, iht_->contains(i), std::make_optional(i), std::string("Contains ") + std::to_string(i) + std::string(" maintains true"));
      }
      ROME_INFO(" = 50% Finished = ");
      for(int i = 0; i < scale_size; i++){
        test_output(show_passing, iht_->remove(i), std::make_optional(i), std::string("Removes ") + std::to_string(i));
        test_output(show_passing, iht_->contains(i), std::nullopt, std::string("Contains ") + std::to_string(i) + std::string(" false"));
      }
      ROME_INFO(" = 75% Finished = ");
      for(int i = 0; i < scale_size; i++){
        test_output(show_passing, iht_->contains(i), std::nullopt, std::string("Contains ") + std::to_string(i) + std::string(" maintains false"));
      }
      ROME_INFO("All test cases finished");
    } else if (test_operations) {
      ROME_INFO("Starting test cases.");
      test_output(true, iht_->contains(5), std::nullopt, "Contains 5");
      test_output(true, iht_->contains(4), std::nullopt, "Contains 4");
      test_output(true, iht_->insert(5, 10), std::nullopt, "Insert 5");
      test_output(true, iht_->insert(5, 11),  std::make_optional(10), "Insert 5 again should fail");
      test_output(true, iht_->contains(5),  std::make_optional(10), "Contains 5");
      test_output(true, iht_->contains(4), std::nullopt, "Contains 4");
      test_output(true, iht_->remove(5),  std::make_optional(10), "Remove 5");
      test_output(true, iht_->remove(4), std::nullopt, "Remove 4");
      test_output(true, iht_->contains(5), std::nullopt, "Contains 5");
      test_output(true, iht_->contains(4), std::nullopt, "Contains 4");
      ROME_INFO("All cases finished");
    } else {
      ROME_INFO("Use main executable not test");
    }
    pool->KillWorkerThread();

    ROME_INFO("[EXPERIMENT] -- End of execution; -- ");
    return 0;
}
