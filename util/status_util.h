#pragma once

// [mfs] This isn't really enough to justify a file

#define STATUSVAL_OR_DIE(__s)                                                  \
  if (!(__s.status.t == sss::Ok)) {                                            \
    ROME_FATAL(__s.status.message.value());                                    \
  }