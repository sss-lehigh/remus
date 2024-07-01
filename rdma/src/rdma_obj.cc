/* Manually incorporated */
#include "rdma_obj.h"

class RDMA_obj{
    private:
        int port;
        std::string nodes_str;
        int num_threads;

        /* node id */
        int id;

        /* Memory we can operate on */
        std::vector<rdma_capability_thread *> rdma_capabilities;

    public:
        /* constructor */
        RDMA_obj(int port, const std::string &nodes_str, int threads)
        : port(port), nodes_str(nodes_str), id(id), num_threads(threads) {
            using namespace remus::rdma;
            using namespace std::string_literals;
            REMUS_INIT_LOG();

            /* Grab the node id from the local environment */
            id = stoi(std::getenv("NODE_ID"));

            /* Split nodes by comma */
            std::vector<std::string> nodes;

            size_t offset = 0;
            while ((offset = nodes_str.find(",")) != std::string::npos) {
                nodes.push_back(nodes_str.substr(0, offset));
                nodes_str.erase(0, offset + 1);
            }
            /* Vector of node strings */
            nodes.push_back(nodes_str);

            for(auto n : nodes) {
                REMUS_DEBUG("Have node: {}", n);
            }

            std::vector<Peer> peers;
            int node_id = 0;
            /* #node * #threads = #peers */
            /* One thread for every peer */
            for(auto n : nodes) {
                for(int tid = 0; tid < num_threads; ++tid) {
                    Peer next(node_id, n, port_num + node_id + 1);
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
            for(int t = 0; t < num_threads; i++){
                rdma_capabilities.push_back(pool[t]->RegisterThread());
            }
        }
           

        /* destructor */
        ~RDMA_obj(){
            REMUS_DEBUG("Deleting pools now");
            delete[] pools;
        }

        void establish_pool

        size_t read(uint32_t thread_id, unsigned char *bytes, size_t len){
            rdma_capabilities[thread_id]->Allocate<unsigned char>(len);
            
        }

        size_t write(uint32_t thread_id, unsigned char *bytes, size_t len){

        }

        bool is_local(uint32_t thread_id){
            return rdma_capabilities[thread_id]
        }
};

