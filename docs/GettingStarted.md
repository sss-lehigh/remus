# Getting Started

- Install remus using `tools/install.sh` on your ubuntu machine

To use RDMA:

- Initialize logging using `REMUS_INIT_LOG()`
- Create `remus::rdma::Peer` objects, with node ids, the node, and the port number
- Create a `remus::rdma::rdma_capability` per each thread (Peer) on the node 
- Initialize the pool with `init_pool` for the memory size and a vector of peers
    - Note initialization must be done in parallel for each thread to not block
- Call `RegisterThread` on each thread that should participate in RDMA
- Use the APIs
- See `rdma/test/multinode.cc` for a reference

Common Errors:
- Make sure your user limits are set high enough with ulimit. You will need to create many files to use remus. The launch script does this automatically.
- If you receive errors about creating connections, you are likely choosing ports that are unavailable. Try different ports instead.

