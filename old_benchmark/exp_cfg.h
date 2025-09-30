#pragma once

#include <remus/remus.h>

constexpr const char *EXP_NAME = "--exp-name";
constexpr const char *OPS = "--ops";
constexpr const char *EXP_OP = "--exp-op";
constexpr const char *ZERO_COPY = "--zero-copy";
constexpr const char *ELEMENTS = "--elements";
constexpr const char *OVERLAP = "--overlap";

// for datastructure benchmark
constexpr const char *TIME_MODE = "--time-mode";
constexpr const char *RUN_TIME = "--run-time";
constexpr const char *NUM_OPS = "--num-ops";
constexpr const char *PREFILL = "--prefill";
constexpr const char *INSERT = "--insert";
constexpr const char *REMOVE = "--remove";
constexpr const char *KEY_LB = "--key-lb";
constexpr const char *KEY_UB = "--key-ub";

auto EXP_ARGS = {
    remus::STR_ARG_OPT(EXP_NAME, "Name of the experiment", "perftest"),
    remus::U64_ARG_OPT(OPS, "Number of operations to perform by each thread",
                       0),
    remus::ENUM_ARG_OPT(EXP_OP, "Operation to perform", "Read",
                        {"Read", "Write", "CAS", "FAA"}),
    remus::U64_ARG_OPT(ZERO_COPY, "Use zero-copy for memory allocation", 1),
    remus::U64_ARG_OPT(ELEMENTS, "Number of elements to allocate", 1),
    remus::U64_ARG_OPT(OVERLAP, "Overlap the memory nodes with compute nodes",
                       1),
};

auto DS_EXP_ARGS = {
    remus::U64_ARG_OPT(TIME_MODE,
                       "Time mode (0: run given operations then stop, 1: run "
                       "RUN_TIME seconds)",
                       1),
    remus::U64_ARG_OPT(RUN_TIME, "Number of seconds to run the experiment", 10),
    remus::U64_ARG_OPT(NUM_OPS, "Number of operations to run", 65536),
    remus::U64_ARG_OPT(PREFILL,
                       "Percent of elements to prefill the data structure", 50),
    remus::U64_ARG_OPT(INSERT, "Percent of operations that should be inserts",
                       50),
    remus::U64_ARG_OPT(REMOVE, "Percent of operations that should be removes",
                       50),
    remus::U64_ARG_OPT(KEY_LB, "Lower bound of the key range", 1),
    remus::U64_ARG_OPT(KEY_UB, "Upper bound of the key range", 4096),
};
