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
    for(auto n : nodes) {
        Peer next(node_id, n, port + node_id + 1);
        peers.push_back(next);
        REMUS_DEBUG("Peer list {}:{}@{}", node_id, peers.at(node_id).id, peers.at(node_id).address);
        node_id++;

    }

    uint32_t block_size = 1 << 10;
    // Create a rdma capability with num_thread sets of connections
    std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(peers.at(id), num_threads);
    pool->init_pool(block_size, peers);

}

size_t RDMA_obj::read(){
    
}

size_t RDMA_obj::write(){

}

/* destructor */
RDMA_obj::~RDMA_obj(){
    REMUS_DEBUG("Deleting pools now");
    delete[] pools;
}

std::vector<rdma_capability_thread *> RDMA_obj::get_rdma_capabilities(){ return rdma_capabilities; }


