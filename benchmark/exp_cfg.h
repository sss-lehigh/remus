#pragma once

#include <remus/cli.h>

constexpr const char *EXP_NAME = "--exp-name";
constexpr const char *OPS = "--ops";
constexpr const char *EXP_OP = "--exp-op";
constexpr const char *ZERO_COPY = "--zero-copy";
constexpr const char *ELEMENTS = "--elements";
constexpr const char *OVERLAP = "--overlap";
constexpr const char *THINK_TIME = "--think-time";
constexpr const char *READ_P = "--read-p";
auto EXP_ARGS = {
    remus::STR_ARG_OPT(EXP_NAME,
                       "Name of the experiment", "perftest"),
    remus::U64_ARG_OPT(OPS,
                       "Number of operations to perform by each thread", 0),
    remus::ENUM_ARG_OPT(EXP_OP,
                       "Operation to perform", "Read", {"Read", "Write", "CAS", "FAA"}),
    remus::U64_ARG_OPT(ZERO_COPY,
                       "Use zero-copy for memory allocation", 1),
    remus::U64_ARG_OPT(ELEMENTS,
                       "Number of elements to allocate", 1),
    remus::U64_ARG_OPT(OVERLAP,
                       "Overlap the memory nodes with compute nodes", 1),
    remus::U64_ARG_OPT(THINK_TIME,
                        "Think time between operations", 0),
    remus::U64_ARG_OPT(READ_P,
                            "Percentage of read operations to write ops.", 0)                                      
};