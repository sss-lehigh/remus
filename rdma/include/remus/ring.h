#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "remus/logging.h"

namespace remus {

/// @brief Tracks the availability of ring slots via a state machine
struct ring_counter_t {
  /// @brief An enum to track the state of a ring counter slot
  enum class State {
    AVAILABLE,
    IN_USE,
    TO_BE_FREED,
  };

  /// @brief Attempts to reserve the next available slot in the ring
  /// @param counter_end The index of the next slot to check for availability
  /// @param counter_assignments A vector of State enums that tracks the state
  /// of each slot
  /// @param counter_num The total number of slots in the ring
  /// @return Index or nullopt
  static inline std::optional<size_t> acquire(
      size_t &counter_end, std::vector<State> &counter_assignments,
      const size_t counter_num) {
    if (counter_assignments[counter_end] == State::AVAILABLE) {
      size_t idx = counter_end;
      counter_assignments[idx] = State::IN_USE;
      counter_end = (counter_end + 1) % counter_num;
      return idx;
    } else {
      return std::nullopt;
    }
  }

  /// @brief Releases a slot in the ring counter
  /// @param idx The index of the slot to release
  /// @param counter_start The index of the first slot in the ring
  /// @param counter_assignments A vector of State enums that tracks the state
  /// @param counter_num The total number of slots in the ring
  static inline void release(const size_t idx, size_t &counter_start,
                             std::vector<State> &counter_assignments,
                             const size_t counter_num) {
    REMUS_ASSERT(counter_assignments[idx] == State::IN_USE,
                 "ring_counter double free is not allowed");
    counter_assignments[idx] = State::TO_BE_FREED;

    while (counter_assignments[counter_start] == State::TO_BE_FREED) {
      counter_assignments[counter_start] = State::AVAILABLE;
      counter_start = (counter_start + 1) % counter_num;
    }
  }
};

/// @brief Manages allocation and deallocation of buffers in a ring buffer
///
/// TODO: This is an odd code pattern: all the methods are static?
struct ring_buf_t {
  /// A struct to record the advance of the buf
  struct buf_allocation_t {
    uint8_t
        *next_available_addr;  // The next available address in the ring buffer
    bool in_use;               // Bool to indicate if the buffer is in use
  };

  /// @brief Calculates the next aligned address based on the given address and
  /// alignment
  /// @param address The address to align
  /// @param align The alignment value, must be a power of two
  /// @return A pointer to the next aligned address
  static inline uint8_t *nextalign(const uint8_t *const address,
                                   const size_t align) {
    auto res = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(address) + (align - 1)) & ~(align - 1));
    return res;
  }

  /// @brief Keeps the alignment of the ring buffer end
  /// @param ring_buf_end A reference to the end of the ring buffer
  /// @param ring_buf_allocations A map that tracks buffer allocations
  /// @param align The alignment value, must be a power of two
  static inline void keep_align(
      uint8_t *&ring_buf_end,
      std::unordered_map<uint8_t *, buf_allocation_t> &ring_buf_allocations,
      const size_t align) {
    uint8_t *aligned = nextalign(ring_buf_end, align);
    if (aligned != ring_buf_end) {
      auto padding_start = ring_buf_end;
      ring_buf_end = aligned;
      ring_buf_allocations[padding_start] = {ring_buf_end, false};
    }
  }

  /// @brief Allocates an aligned memory region from the ring buffer if there is
  /// enough space
  /// @param ring_buf The start of the ring buffer
  /// @param ring_buf_end A reference to the end of the ring buffer
  /// @param ring_buf_start A reference to the start of the ring buffer
  /// @param ring_buf_size The size of the ring buffer
  /// @param ring_buf_allocations A map that tracks buffer allocations
  /// @param size The size of the buffer to allocate
  /// @param align The alignment value, must be a power of two
  /// @return A pointer to the allocated buffer, or nullptr if allocation fails
  static inline uint8_t *acquire(
      const uint8_t *const ring_buf, uint8_t *&ring_buf_end,
      uint8_t *&ring_buf_start, const size_t ring_buf_size,
      std::unordered_map<uint8_t *, buf_allocation_t> &ring_buf_allocations,
      const size_t size, const size_t align) {
    if (size + nextalign(ring_buf, align) > ring_buf + ring_buf_size) {
      return nullptr;
    }
    uint8_t *buf = nullptr;
    auto real_size = size + nextalign(ring_buf_end, align) - ring_buf_end;
    // case 1: ring_buf_start <= ring_buf_end and the remainning space
    // is enough
    if (ring_buf_start <= ring_buf_end &&
        ring_buf_end + real_size <= ring_buf + ring_buf_size) {
      // the remainning space is enough
      // first, keep alignment
      keep_align(ring_buf_end, ring_buf_allocations, align);
      buf = ring_buf_end;
      ring_buf_end += size;  // if the branch = condition met, ring_buf_end will
                             // be updated to ring_buf_ + ring_buf_size_
      ring_buf_allocations[buf] = {ring_buf_end, true};

    } else {
      // case 2: ring_buf_start <= ring_buf_end and the remainning
      // space is not enough, we convert case 2 to case 3 or case 4
      if (ring_buf_start <= ring_buf_end) {
        ring_buf_allocations[ring_buf_end] = {const_cast<uint8_t *>(ring_buf),
                                              false};
        ring_buf_end = const_cast<uint8_t *>(ring_buf);  // wrap to front
        // realign at the new position – the alignment gap is different now
        real_size = size + nextalign(ring_buf_end, align) - ring_buf_end;
      }
      // case 3: ring_buf_start == ring_buf_end, can only from case 2
      if (ring_buf_start == ring_buf_end) {
        if (ring_buf_allocations.size() == 1) {
          keep_align(ring_buf_end, ring_buf_allocations, align);
          buf = ring_buf_end;
          ring_buf_end += size;
          if (ring_buf_end == ring_buf + ring_buf_size) {
            ring_buf_allocations[buf] = {const_cast<uint8_t *>(ring_buf),
                                         true};  // Correct for wrap
          } else {
            ring_buf_allocations[buf] = {ring_buf_end, true};
          }
        } else {
          return nullptr;
        }
      } else {
        // case 4: ring_buf_start > ring_buf_end
        if (ring_buf_end + real_size <= ring_buf_start) {
          // if not overflowed, check if we can satisfy the request
          // first, keep alignment
          keep_align(ring_buf_end, ring_buf_allocations, align);
          buf = ring_buf_end;
          ring_buf_end += size;
          if (ring_buf_end == ring_buf + ring_buf_size) {
            ring_buf_allocations[buf] = {const_cast<uint8_t *>(ring_buf),
                                         true};  // Correct for wrap
          } else {
            ring_buf_allocations[buf] = {ring_buf_end, true};
          }
        } else {
          return nullptr;
        }
      }
    }
    return buf;
  }

  /// @brief Frees a buffer in the ring buffer
  /// @param buf The pointer to the buffer to free
  /// @param ring_buf_allocations A map that tracks buffer allocations
  /// @param ring_buf_start A reference to the start of the ring buffer
  /// @param ring_buf The start of the ring buffer
  /// @param ring_buf_size The size of the ring buffer
  static inline void release(
      uint8_t *const buf,
      std::unordered_map<uint8_t *, buf_allocation_t> &ring_buf_allocations,
      uint8_t *&ring_buf_start, uint8_t *ring_buf, const size_t ring_buf_size) {
    REMUS_ASSERT(ring_buf_allocations.find(buf) != ring_buf_allocations.end() &&
                     ring_buf_allocations[buf].in_use,
                 "ring buf not exists or not in use, can not release");
    ring_buf_allocations[buf].in_use = false;
    while (ring_buf_allocations.find(ring_buf_start) !=
               ring_buf_allocations.end() &&
           !ring_buf_allocations[ring_buf_start].in_use) {
      auto to_erase = ring_buf_start;
      ring_buf_start = ring_buf_allocations[ring_buf_start].next_available_addr;
      ring_buf_allocations.erase(to_erase);
      if (ring_buf_start == ring_buf) {
        REMUS_ASSERT(ring_buf + ring_buf_size - to_erase >= 0,
                     "ring_buf_start = ring_buf_ + ring buf size = "
                     "{} is not greater than to_erase = {}",
                     (uint64_t)(ring_buf + ring_buf_size), (uint64_t)to_erase);
      } else {
        REMUS_ASSERT(ring_buf_start - to_erase >= 0,
                     "ring buf start is not greater than to_erase");
      }
    }
  }
};
}  // namespace remus