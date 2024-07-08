/* Manually incorporated */
#include <remus/rdma/rdma_obj.h>
using namespace remus::rdma;
using namespace remus::util;
using namespace string_literals;

/* constructor */
RDMA_obj::RDMA_obj(int port, std::string &nodes_str, int id, int threads): 
    port(port), nodes_str(nodes_str), node_id(id), num_threads(threads) {

    REMUS_INIT_LOG();

    /* Split nodes by comma */
    std::vector<std::string> nodes;

    size_t offset = 0;
    while ((offset = nodes_str.find(",")) != std::string::npos) {
        nodes.push_back(nodes_str.substr(0, offset));
        nodes_str.erase(0, offset + 1);
    }
    /* Add the final node str */
    nodes.push_back(nodes_str);

    for(auto n : nodes) {
        REMUS_DEBUG("Have node: {}", n);
    }

    std::vector<Peer> peers;
    /* #node * #threads = #peers */
    /* One thread for every peer */
    for(auto n : nodes) {
        for(int tid = 0; tid < num_threads; ++tid) {
            Peer next(node_id, n, port + node_id + 1);
            peers.push_back(next);
            REMUS_DEBUG("Peer list {}:{}@{}", node_id, peers.at(node_id).id, peers.at(node_id).address);
            node_id++;
        }
    }

    std::vector<std::thread> mempool_threads;
    std::shared_ptr<rdma_capability> *pools = new std::shared_ptr<rdma_capability>[num_threads];

    // Create multiple memory pools to be shared (have to use threads since Init is blocking)
    uint32_t block_size = 1 << 10;
    for (int i = 0; i < num_threads; i++) {
        mempool_threads.emplace_back(std::thread(
        [&](int mp_index, int self_index) {
            Peer self = peers.at(self_index);
            REMUS_DEBUG("Creating pool for {}:{}@{}", self_index, self.id, self.address);
            /* Create a rdma capability with 2 sets of connections */
            std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(peers.at(id), 2);
            pool->init_pool(block_size, peers);
            REMUS_DEBUG("Created pool for {}:{}@{}", self_index, self.id, self.address);
            pools[mp_index] = pool;
        },
        i, i + id * num_threads));
    }

    // Let the init finish
    for (int i = 0; i < num_threads; i++) {
        mempool_threads[i].join();
    }
    
    /* Establish rdma capabilities */
    for(int t = 0; t < num_threads; t++){
        rdma_capabilities.push_back(pools[t]->RegisterThread());
    }
}

/* destructor */
RDMA_obj::~RDMA_obj(){
    REMUS_DEBUG("Deleting pools now");
    delete[] pools;
}

std::vector<rdma_capability_thread *> RDMA_obj::get_rdma_capabilities(){ return rdma_capabilities; }


