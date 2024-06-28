/* Manually incorporated */
#include "rdma_obj.h"

class RDMA_obj{
    private:
        int port;
        std::string nodes_str;
        int id;
        int num_threads;

        /* Memory regions we operate on */
        std::shared_ptr<rdma_capability> *pools;

    public:
        /* constructor */
        RDMA_obj(int port, const std::string &nodes_str, int id, int threads)
        : port(port), nodes_str(nodes_str), id(id), num_threads(threads) {
            using namespace remus::rdma;
            using namespace std::string_literals;
            REMUS_INIT_LOG();

            /* Split nodes by comma */
            std::vector<std::string> nodes;

            size_t offset = 0;
            while ((offset = nodes_str.find(",")) != std::string::npos) {
                nodes.push_back(nodes_str.substr(0, offset));
                nodes_str.erase(0, offset + 1);
            }
            nodes.push_back(nodes_str);

            /* Logging the nodes */
            for(auto n : nodes) {
                REMUS_DEBUG("Have node: {}", n);
            }

            /* Defining the peer memory regions */
            std::vector<Peer> peers;
            int node_id = 0;
            for(auto n : nodes) {
                for(int tid = 0; tid < threads; ++tid) { // 1:1 peer to thread, #peers = #nodes * #threads
                Peer next(node_id, n, port_num + node_id + 1);
                peers.push_back(next);
                REMUS_DEBUG("Peer list {}:{}@{}", node_id, peers.at(node_id).id, peers.at(node_id).address);
                node_id++;
                }
            }

            std::vector<std::thread> mempool_threads;
            std::shared_ptr<rdma_capability>* pools = new std::shared_ptr<rdma_capability>[threads];

            // Create multiple memory pools to be shared (have to use threads since Init
            // is blocking)
            uint32_t block_size = 1 << 10;
            for (int i = 0; i < threads; i++) {
                mempool_threads.emplace_back(std::thread(
                [&](int mp_index, int self_index) {
                    Peer self = peers.at(self_index);
                    REMUS_DEBUG("Creating pool for {}:{}@{}", self_index, self.id, self.address);
                    std::shared_ptr<rdma_capability> pool = std::make_shared<rdma_capability>(self);
                    pool->init_pool(block_size, peers);
                    REMUS_DEBUG("Created pool for {}:{}@{}", self_index, self.id, self.address);
                    pools[mp_index] = pool;
                },
                i, i + id * threads));
            }

            /* Allow initialization to complete */
            for (int i = 0; i < threads; i++) {
                mempool_threads[i].join();
            }
        }

        /* destructor */
        ~RDMA_obj(){
            REMUS_DEBUG("Deleting pools now");
            delete[] pools;
        }

        size_t read(){
            
        }

        size_t write(){

        }
};

