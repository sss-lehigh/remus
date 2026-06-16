#pragma once

#include "cli.h"

namespace remus {
constexpr const char *NODE_ID = "--node-id";
constexpr const char *MN_PORT = "--mn-port";
constexpr const char *FIRST_MN_ID = "--first-mn-id";
constexpr const char *LAST_MN_ID = "--last-mn-id";
constexpr const char *SEG_SIZE = "--seg-size";
constexpr const char *SEGS_PER_MN = "--segs-per-mn";
constexpr const char *FIRST_CN_ID = "--first-cn-id";
constexpr const char *LAST_CN_ID = "--last-cn-id";
constexpr const char *QP_LANES = "--qp-lanes";
constexpr const char *QP_SCHED_POL = "--qp-sched-pol";
constexpr const char *ALLOC_POL = "--alloc-pol";
constexpr const char *CN_THREADS = "--cn-threads";
constexpr const char *CN_THREAD_MSGS = "--cn-thread-msgs";
constexpr const char *CN_THREAD_BUFSZ = "--cn-thread-bufsz";

/// Standard command-line options for Remus.  Note that every machine should
/// have identical args, except for NODE_ID.
auto ARGS = {
    U64_ARG(NODE_ID, "A numerical identifier for this node."),
    U64_ARG_OPT(SEG_SIZE,
                "The size of each remotely-accessible memory segment on each "
                "memory node will be 2^{seg-size}.",
                20),
    U64_ARG_OPT(SEGS_PER_MN,
                "The number of remotely-accessible memory segments on each "
                "memory node.",
                2),
    U64_ARG(FIRST_CN_ID,
            "The node-id of the first node that performs computations."),
    U64_ARG(LAST_CN_ID,
            "The node-id of the last node that performs computations."),
    U64_ARG(FIRST_MN_ID,
            "The node-id of the first node that hosts memory segments."),
    U64_ARG(LAST_MN_ID,
            "The node-id of the last node that hosts memory segments."),
    U64_ARG_OPT(QP_LANES,
                "Each compute node should have qp-lane-width connections to "
                "each memory node.",
                2),
    ENUM_ARG_OPT(QP_SCHED_POL,
                 "How to choose which qp to use: RAND, RR, or MOD", "RAND",
                 {"RAND", "RR", "MOD", "ONE_TO_ONE"}),
    U64_ARG(MN_PORT, "The port that memory nodes should use to wait for "
                     "connections during the initialization phase."),
    U64_ARG(CN_THREADS, "The number of threads to run on each compute node"),
    U64_ARG_OPT(CN_THREAD_MSGS,
                "The maximum number of concurrent messages that a thread can "
                "issue without waiting on a completion.",
                16),
    U64_ARG_OPT(CN_THREAD_BUFSZ,
                "The log_2 of the size of the buffer to allocate to each "
                "compute thread.",
                20),
    ENUM_ARG_OPT(ALLOC_POL,
                 "How should ComputeThreads pick Segments for allocation: "
                 "RAND, GLOBAL-RR, GLOBAL-MOD, LOCAL-RR, LOCAL-MOD",
                 "GLOBAL-RR",
                 {"RAND", "GLOBAL-RR", "GLOBAL-MOD", "LOCAL-RR", "LOCAL-MOD"}),
};
} // namespace remus
