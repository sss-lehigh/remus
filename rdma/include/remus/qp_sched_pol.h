#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>
#include <x86intrin.h>

#include "cfg.h"
#include "logging.h"
#include "util.h"

namespace remus::internal {

/// @brief A policy for scheduling QPs (Queue Pairs) to interact with MemoryNodes
/// @details
/// QpSchedPolicy encapsulates the state and decision making process regarding
/// how to pick a QP to use in order to interact with a MemoryNode.
class QpSchedPolicy {
  // TODO: Why class+public instead of struct?
public:
  /// An enum for tracking which policy was configured at start-up time
  ///
  /// TODO: Document each option
  enum Policy { NONE, MOD, RR, RAND, ONE_TO_ONE };

  /// Convert a string (such as what would be in an ArgMap) into a Policy
  ///
  /// @param policy TODO
  /// @return
  static Policy to_policy(std::string policy) {
    if (policy == "MOD") {
      return MOD;
    } else if (policy == "ONE_TO_ONE") {
      return ONE_TO_ONE;
    } else if (policy == "RAND") {
      return RAND;
    } else if (policy == "RR") {
      return RR;
    }
    REMUS_FATAL("Invalid QpSchedPolicy {}", policy);
  }

private:
  Policy policy_;                // The chosen policy
  rdtsc_rand_t prng_;            // A pseudorandom number generator
  const uint32_t num_lanes_;     // The number of QP lanes to each MemoryNode
  const uint32_t num_threads_;   // The number of ComputeThreads per ComputeNode
  uint32_t last_lane_;           // The last lane we took
  std::vector<uint32_t> per_mn_; // Independent tracking for each MemoryNode

public:
  /// Construct a QpSchedPolicy with the default ("none") policy, which always
  /// uses lane 0.
  ///
  /// @param args The arguments to the program
  QpSchedPolicy(std::shared_ptr<remus::ArgMap> args)
      : policy_(NONE), num_lanes_(args->uget(QP_LANES)),
        num_threads_(args->uget(CN_THREADS)), last_lane_(0) {
    // Initialize the per_mn_ counters, so we can switch freely among policies
    for (uint64_t i = 0; i <= args->uget(remus::LAST_MN_ID); ++i) {
      per_mn_.push_back(prng_.rand() % num_lanes_);
    }
  }

  /// Change the policy that will be used for picking a QP
  ///
  /// @param policy     The desired policy
  /// @param thread_id  The unique, zero-based identifier for this thread
  void set_policy(Policy policy, uint64_t thread_id) {
    policy_ = policy;
    if (policy_ == ONE_TO_ONE) {
      REMUS_ASSERT(num_lanes_ >= num_threads_,
                   "ONE_TO_ONE requested with {} threads and only {} lanes",
                   num_threads_, num_lanes_);
      last_lane_ = thread_id;
    } else if (policy_ == MOD) {
      last_lane_ = thread_id % num_lanes_;
    } else if (policy_ == NONE) {
      last_lane_ = 0;
    } else if (policy_ == RAND || policy == RR) {
      // No config needed
    } else {
      REMUS_FATAL("Unrecognized QpSchedPol {}", (uint32_t)policy_);
    }
  }

  /// Use the previously selected QP_SCHED_POL to decide on the index for the
  /// next Connection to use.
  ///
  /// @param mn TODO
  /// @return TODO
  uint32_t get_lane_idx(uint32_t mn) {
    if (policy_ == RR) {
      return (per_mn_[mn] = (++per_mn_[mn]) % num_lanes_);
    } else if (policy_ == RAND) {
      last_lane_ = prng_.rand() % num_lanes_;
    }
    return last_lane_;
  }
};
} // namespace remus::internal