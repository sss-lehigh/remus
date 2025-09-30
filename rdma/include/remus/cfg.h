#pragma once

#include "cli.h"

namespace remus {
/// Node ID for this physical machine.
constexpr const char *NODE_ID = "--node-id";
/// The port that memory nodes should use to wait for connections
constexpr const char *MN_PORT = "--mn-port";
/// The node-id of the first node that hosts memory segments.
constexpr const char *FIRST_MN_ID = "--first-mn-id";
/// The node-id of the last node that hosts memory segments.
constexpr const char *LAST_MN_ID = "--last-mn-id";
/// The size of each remotely-accessible memory segment on each
/// memory node will be 2^{seg-size}.
constexpr const char *SEG_SIZE = "--seg-size";
/// The number of remotely-accessible memory segments on each
constexpr const char *SEGS_PER_MN = "--segs-per-mn";
/// The node-id of the first node that performs computations.
constexpr const char *FIRST_CN_ID = "--first-cn-id";
/// The node-id of the last node that performs computations.
constexpr const char *LAST_CN_ID = "--last-cn-id";
/// Each compute node should have qp-lanes number of connections to
/// each memory node.
constexpr const char *QP_LANES = "--qp-lanes";
/// The QP scheduling policy to use for choosing which
/// connection to use for a given operation. Options are: 
/// MOD, ONE_TO_ONE, RAND, RR. 
constexpr const char *QP_SCHED_POL = "--qp-sched-pol";
/// The allocation policy for ComputeThreads to use when
/// choosing which Segment to allocate from. Options are:
/// RAND, GLOBAL-RR, GLOBAL-MOD, LOCAL-RR, LOCAL-MOD.
constexpr const char *ALLOC_POL = "--alloc-pol";
/// The number of threads to run on each compute node
constexpr const char *CN_THREADS = "--cn-threads";
/// The maximum number of concurrent messages that a thread can
/// issue without waiting on a completion.
constexpr const char *CN_OPS_PER_THREAD = "--cn-ops-per-thread";
/// The log_2 of the size of the buffer to allocate to each
/// compute thread. This is the size of the buffer in bytes.
constexpr const char *CN_THREAD_BUFSZ = "--cn-thread-bufsz";
/// The number of sequential operations that a thread can perform
/// concurrently. This is the number of write operations that can be
/// performed in a row before the thread must wait for a completion.
constexpr const char *CN_WRS_PER_SEQ = "--cn-wrs-per-seq";
/// The command-line option for requesting help
constexpr const char *HELP = "--help";

/// Standard command-line options for Remus.  Note that every machine should
/// have identical args, except for NODE_ID.
///
/// TODO: At one time, we had separate pieces for CloudLab args and for general
///       remus args.  Should we go back to that?  For example, this code uses
///       node ids, but doesn't have anything about how to make node ids from
///       DNS names. Is that sufficient, or is there some implicit tethering to
///       CloudLab hiding in this config?
inline auto ARGS = {
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
                "Each compute node should have qp-lanes connections to "
                "each memory node.",
                2),
    ENUM_ARG_OPT(QP_SCHED_POL,
                 "How to choose which qp to use: RAND, RR, or MOD", "RAND",
                 {"RAND", "RR", "MOD", "ONE_TO_ONE"}),
    U64_ARG(MN_PORT,
            "The port that memory nodes should use to wait for "
            "connections during the initialization phase."),
    U64_ARG(CN_THREADS, "The number of threads to run on each compute node"),

    U64_ARG_OPT(CN_THREAD_BUFSZ,
                "The log_2 of the size of the buffer to allocate to each "
                "compute thread.",
                20),
    ENUM_ARG_OPT(ALLOC_POL,
                 "How should ComputeThreads pick Segments for allocation: "
                 "RAND, GLOBAL-RR, GLOBAL-MOD, LOCAL-RR, LOCAL-MOD",
                 "GLOBAL-RR",
                 {"RAND", "GLOBAL-RR", "GLOBAL-MOD", "LOCAL-RR", "LOCAL-MOD"}),
    U64_ARG_OPT(CN_OPS_PER_THREAD,
                "The maximum number of concurrent messages that a thread can "
                "issue without waiting on a completion.",
                8),
    U64_ARG_OPT(CN_WRS_PER_SEQ,
                "The number of sequential operations that a thread can perform "
                "concurrently.",
                16),
    BOOL_ARG_OPT(HELP, "Print this help message")};
}  // namespace remus
