#pragma once

#include "absl/strings/str_cat.h"
#include "gmock/gmock-matchers.h"
#include "rome/colosseum/streams/stream.h"
#include "status_matcher.h"

namespace testutil {

MATCHER(IsStreamTerminatedStatus,
        absl::StrCat("is ", negation ? "not " : "", "terminated")) {
  return rome::IsStreamTerminated(GetStatus(arg));
}

}  // namespace testutil