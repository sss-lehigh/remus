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

/// @brief A policy for allocating Segments from MemoryNodes
/// @details
/// MnAllocPolicy encapsulates the state and decision making process regarding
/// how to pick a Segment to use to satisfy an allocation request
class MnAllocPolicy {
  // TODO: why class+public?  Make it a struct?
public:
  /// An enum for tracking which policy was configured at start-up time
  ///
  /// TODO: Document each point?
  enum Policy { NONE, GLOBAL_MOD, GLOBAL_RR, RAND, LOCAL_RR, LOCAL_MOD };

  /// Convert a string (such as what would be in an ArgMap) into a Policy
  ///
  /// @param policy
  /// @return
  static Policy to_policy(std::string policy) {
    if (policy == "RAND") {
      return RAND;
    } else if (policy == "GLOBAL-RR") {
      return GLOBAL_RR;
    } else if (policy == "GLOBAL-MOD") {
      return GLOBAL_MOD;
    } else if (policy == "LOCAL-RR") {
      return LOCAL_RR;
    } else if (policy == "LOCAL-MOD") {
      return LOCAL_MOD;
    }
    REMUS_FATAL("Invalid MnAllocPolicy {}", policy);
  }

private:
  Policy policy_;             // The chosen policy
  rdtsc_rand_t prng_;         // A pseudorandom number generator
  const uint32_t num_segs_;   // The number of Segments per MemoryNode
  const uint32_t num_mns_;    // The number of MemoryNodes
  const uint32_t total_segs_; // The total number of Segments
  uint32_t last_mn_;          // The last MemoryNode we tried
  uint32_t last_seg_;         // The last segment we tried

public:
  /// Construct a MnAllocPolicy with the default ("none") policy, which always
  /// uses segment 0.
  ///
  /// @param args The arguments to the program
  MnAllocPolicy(std::shared_ptr<remus::ArgMap> args)
      : policy_(NONE), num_segs_(args->uget(SEGS_PER_MN)),
        // TODO: This assumes MemoryNode Ids start at 0
        num_mns_(args->uget(LAST_MN_ID) + 1), total_segs_(num_segs_ * num_mns_),
        last_mn_(0), last_seg_(0) {}

  /// Change the policy that will be used for picking a QP
  ///
  /// @param policy     The desired policy
  /// @param args       The command-line arguments to the program
  /// @param thread_id  The unique, zero-based identifier for this thread
  void set_policy(Policy policy, std::shared_ptr<ArgMap> args,
                  uint64_t thread_id) {
    // Figure out the number of ComputeNodes and ComputeThreads
    uint64_t m0 = args->uget(FIRST_MN_ID);
    uint64_t mn = args->uget(LAST_MN_ID);
    uint64_t c0 = args->uget(FIRST_CN_ID);
    uint64_t cn = args->uget(LAST_CN_ID);
    uint64_t num_threads = args->uget(CN_THREADS);
    uint64_t thread_uid = (args->uget(NODE_ID) - c0) * num_threads + thread_id;

    policy_ = policy;
    if (policy_ == GLOBAL_MOD) {
      uint64_t seg_uid = thread_uid % total_segs_;
      last_mn_ = seg_uid / num_segs_;
      last_seg_ = seg_uid % num_segs_;
    } else if (policy_ == GLOBAL_RR) {
      // Randomize starting point
      last_mn_ = prng_.rand() % num_mns_;
      last_seg_ = prng_.rand() % num_segs_;
    } else if (policy_ == LOCAL_MOD) {
      REMUS_ASSERT(c0 == m0 && cn == mn,
                   "LOCAL_MOD requires every node to be Compute and Memory");
      last_mn_ = args->uget(NODE_ID);
      last_seg_ = thread_id % num_segs_;
    } else if (policy_ == LOCAL_RR) {
      REMUS_ASSERT(c0 == m0 && cn == mn,
                   "LOCAL_RR requires every node to be Compute and Memory");
      last_mn_ = args->uget(NODE_ID);
      last_seg_ = prng_.rand() % num_segs_;
    } else if (policy_ == RAND) {
      // No initialization needed
    } else if (policy_ == NONE) {
      last_mn_ = 0;
      last_seg_ = 0;
    } else {
      REMUS_FATAL("Unrecognized MnAllocPol {}", (uint32_t)policy_);
    }
  }

  /// Use the previously selected MN_ALLOC_POL to decide on the MemoryNode and
  /// Segment to use for the next allocation
  ///
  /// @return A pair consisting of the id of the memory node to allocate from,
  ///         and the id of the segment on that node to allocate from.
  std::pair<uint32_t, uint32_t> get_mn_seg() {
    if (policy_ == GLOBAL_MOD || policy_ == LOCAL_MOD || policy_ == NONE) {
      // Don't change last_mn_ or last_seg_
    } else if (policy_ == GLOBAL_RR) {
      // Go to next seg, if that causes overflow, go to next MemoryNode
      last_seg_ = (++last_seg_) % num_segs_;
      if (last_seg_ == 0) {
        last_mn_ = (++last_mn_) % num_mns_;
      }
    } else if (policy_ == LOCAL_RR) {
      // Don't change MemoryNodes on overflow, just start back at 0
      last_seg_ = (++last_seg_) % num_segs_;
    } else if (policy_ == RAND) {
      last_mn_ = prng_.rand() % num_mns_;
      last_seg_ = prng_.rand() % num_segs_;
    }
    return {last_mn_, last_seg_};
  }
};
} // namespace remus::internal