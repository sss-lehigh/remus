#pragma once

#include <infiniband/verbs.h>
#include <sys/mman.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "util.h"

namespace rome::rdma {

// Remotely accessible memory backed by either raw memory or hugepages (if
// enabled). This is just a flat buffer that is preallocated. The user of this
// memory region can then use it directly, or wrap it around some more complex
// allocation mechanism.
class RdmaMemory {
public:
  static constexpr int kDefaultAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  ~RdmaMemory();
  RdmaMemory(uint64_t capacity, ibv_pd *const pd)
      : RdmaMemory(capacity, std::nullopt, pd) {}
  RdmaMemory(uint64_t capacity, std::optional<std::string_view> path,
             ibv_pd *const pd);

  RdmaMemory(const RdmaMemory &) = delete;
  RdmaMemory(RdmaMemory &&rm)
      : capacity_(rm.capacity_), raw_(std::move(rm.raw_)),
        memory_regions_(std::move(rm.memory_regions_)) {}

  // Getters.
  uint64_t capacity() const { return capacity_; }

  uint8_t *raw() const {
    return std::visit([](const auto &r) { return r.get(); }, raw_);
  }

  // Creates a new memory region associated with the given protection domain
  // `pd` at the provided offset and with the given length. If a region with the
  // same `id` already exists then it returns `AlreadyExists`.
  sss::Status RegisterMemoryRegion(std::string_view id, int offset, int length);
  sss::Status RegisterMemoryRegion(std::string_view id, ibv_pd *const pd,
                                   int offset, int length);

  ibv_mr *GetDefaultMemoryRegion() const;
  sss::StatusVal<ibv_mr *> GetMemoryRegion(std::string_view id) const;

private:
  static constexpr char kDefaultId[] = "default";

  // Handles deleting memory allocated using mmap (when using hugepages)
  struct mmap_deleter {
    void operator()(uint8_t raw[]) { munmap(raw, sizeof(*raw)); }
  };

  // Validates that the given offset and length are not ill formed w.r.t. to the
  // capacity of this memory.
  bool ValidateRegion(int offset, int length);

  // Preallocated size.
  const uint64_t capacity_;

  // Either points to an array of bytes allocated with the system allocator or
  // with `mmap`. At some point, this could be replaced with a custom allocator.
  std::variant<std::unique_ptr<uint8_t[]>,
               std::unique_ptr<uint8_t[], mmap_deleter>>
      raw_;

  // A map of memory regions registered with this particular memory.
  std::unordered_map<std::string, ibv_mr_unique_ptr> memory_regions_;
};

} // namespace rome::rdma