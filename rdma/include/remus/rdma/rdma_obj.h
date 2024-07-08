#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include <unistd.h>

#include <remus/logging/logging.h>
#include <remus/rdma/rdma.h>
#include <remus/util/cli.h>

class RDMA_obj {
private:
    int port;
    std::string nodes_str;
    int num_threads;

    /* node id */
    int node_id;

    /* Memory we can operate on */
    std::vector<remus::rdma::rdma_capability_thread*> rdma_capabilities;
    std::shared_ptr<remus::rdma::rdma_capability> *pools;

public:
    /* constructor */
    RDMA_obj(int port, std::string &nodes_str, int id, int threads);

    /* destructor */
    ~RDMA_obj();

    std::vector<remus::rdma::rdma_capability_thread *> get_rdma_capabilities();
};
