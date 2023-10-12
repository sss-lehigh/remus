#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <variant>

#include "../logging/logging.h"
#include "../util/status_util.h"
#include "../vendor/sss/status.h"

#include "util.h"

namespace {

// Tries to read the number of available hugepages from the system. This is only
// implemented for Linux-based operating systems.
inline sss::StatusVal<int> GetNumHugepages(std::string_view path) {
  // Try to open file.
  // [mfs] I had to explicitly convert to string?
  std::ifstream file(std::string(path).data());
  if (!file.is_open()) {
    sss::Status err = {sss::Unknown, "Failed to open file: "};
    err << path;
    return {err, {}};
  }

  // Read file.
  int nr_hugepages;
  file >> nr_hugepages;
  if (!file.fail()) {
    return {sss::Status::Ok(), nr_hugepages};
  } else {
    return {{sss::Unknown, "Failed to read nr_hugepages"}, {}};
  }
}

} // namespace

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

  ~RdmaMemory() { memory_regions_.clear(); }
  RdmaMemory(uint64_t capacity, ibv_pd *const pd)
      : RdmaMemory(capacity, std::nullopt, pd) {}
  RdmaMemory(uint64_t capacity, std::optional<std::string_view> path,
             ibv_pd *const pd)
      : capacity_(capacity) {
    bool use_hugepages = false;
    if (path.has_value()) {
      auto nr_hugepages = GetNumHugepages(path.value());
      if (nr_hugepages.status.t == sss::Ok && nr_hugepages.val.value() > 0) {
        use_hugepages = true;
      }
    }

    if (!use_hugepages) {
      ROME_TRACE("Not using hugepages; performance might suffer.");
      auto bytes = ((capacity >> 6) + 1) << 6; // Round up to nearest 64B
      raw_ = std::unique_ptr<uint8_t[]>(
          reinterpret_cast<uint8_t *>(std::aligned_alloc(64, bytes)));
      ROME_ASSERT(std::get<0>(raw_) != nullptr, "Allocation failed.");
    } else {
      ROME_INFO("Using hugepages");
      raw_ =
          std::unique_ptr<uint8_t[], mmap_deleter>(reinterpret_cast<uint8_t *>(
              mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)));
      ROME_ASSERT(reinterpret_cast<void *>(std::get<1>(raw_).get()) !=
                      MAP_FAILED,
                  "mmap failed.");
    }
    OK_OR_FAIL(RegisterMemoryRegion(kDefaultId, pd, 0, capacity_));
  }

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
  sss::Status RegisterMemoryRegion(std::string_view id, int offset,
                                   int length) {
    return RegisterMemoryRegion(id, GetDefaultMemoryRegion()->pd, offset,
                                length);
  }
  sss::Status RegisterMemoryRegion(std::string_view id, ibv_pd *const pd,
                                   int offset, int length) {
    if (!ValidateRegion(offset, length)) {
      sss::Status err = {sss::FailedPrecondition,
                         "Requested memory region invalid: "};
      err << id;
      return err;
    }

    auto iter = memory_regions_.find(std::string(id));
    if (iter != memory_regions_.end()) {
      sss::Status err = {sss::AlreadyExists, "Memory region exists: {}"};
      err << id;
      return err;
    }

    auto *base = reinterpret_cast<uint8_t *>(std::visit(
                     [](const auto &raw) { return raw.get(); }, raw_)) +
                 offset;
    auto mr = ibv_mr_unique_ptr(ibv_reg_mr(pd, base, length, kDefaultAccess));
    if (mr == nullptr) {
      return {sss::InternalError, "Failed to register memory region"};
    }
    memory_regions_.emplace(id, std::move(mr));
    ROME_DEBUG("Memory region registered: {} @ {} to {} (length={})", id,
               fmt::ptr(base), fmt::ptr(base + length), length);
    return sss::Status::Ok();
  }

  ibv_mr *GetDefaultMemoryRegion() const {
    return memory_regions_.find(kDefaultId)->second.get();
  }
  sss::StatusVal<ibv_mr *> GetMemoryRegion(std::string_view id) const {
    auto iter = memory_regions_.find(std::string(id));
    if (iter == memory_regions_.end()) {
      sss::Status err = {sss::NotFound, "Memory region not found: {}"};
      err << id;
      return {err, {}};
    }
    return {sss::Status::Ok(), iter->second.get()};
  }

private:
  static constexpr char kDefaultId[] = "default";

  // Handles deleting memory allocated using mmap (when using hugepages)
  struct mmap_deleter {
    void operator()(uint8_t raw[]) { munmap(raw, sizeof(*raw)); }
  };

  // Validates that the given offset and length are not ill formed w.r.t. to the
  // capacity of this memory.
  bool ValidateRegion(int offset, int length) {
    if (offset < 0 || length < 0)
      return false;
    if (offset + length > capacity_)
      return false;
    return true;
  }

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