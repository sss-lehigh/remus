#pragma once

#include <remus/cli.h>

constexpr const char *NELEMS = "--nelems";

auto EXP_ARGS = {
    remus::U64_ARG_OPT(NELEMS,
                       "Number of elements to allocate in one-sided test", 0),

};
