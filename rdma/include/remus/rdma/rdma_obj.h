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

    /* A subset of available QPs that enable one-sided communication */
    rdma_capability_thread *one_sided_obj;

public:
    /* constructor */
    RDMA_obj(int port, std::string &nodes_str, int id, int threads);

    /* destructor */
    ~RDMA_obj();

};
