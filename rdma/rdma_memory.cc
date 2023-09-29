#include "rdma_memory.h"

#include <infiniband/verbs.h>
#include <sys/mman.h>

#include <cstdlib>
#include <fstream>
#include <memory>

#include "../logging/logging.h"
#include "../util/status_util.h"
#include "absl/status/statusor.h"

namespace rome::rdma {

using ::util::AlreadyExistsErrorBuilder;
using ::util::FailedPreconditionErrorBuilder;
using ::util::NotFoundErrorBuilder;
using ::util::UnknownErrorBuilder;

namespace {

// Tries to read the number of available hugepages from the system. This is only
// implemented for Linux-based operating systems.
absl::StatusOr<int> GetNumHugepages(std::string_view path) {
  // Try to open file.
  // [mfs] I had to explicitly convert to string?
  std::ifstream file(std::string(path).data());
  if (!file.is_open()) {
    return UnknownErrorBuilder() << "Failed to open file: " << path;
  }

  // Read file.
  int nr_hugepages;
  file >> nr_hugepages;
  if (!file.fail()) {
    return nr_hugepages;
  } else {
    return absl::UnknownError("Failed to read nr_hugepages");
  }
}

} // namespace

RdmaMemory::~RdmaMemory() { memory_regions_.clear(); }

RdmaMemory::RdmaMemory(uint64_t capacity, std::optional<std::string_view> path,
                       ibv_pd *const pd)
    : capacity_(capacity) {
  bool use_hugepages = false;
  if (path.has_value()) {
    auto nr_hugepages = GetNumHugepages(path.value());
    if (nr_hugepages.ok() && nr_hugepages.value() > 0) {
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
    raw_ = std::unique_ptr<uint8_t[], mmap_deleter>(reinterpret_cast<uint8_t *>(
        mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)));
    ROME_ASSERT(reinterpret_cast<void *>(std::get<1>(raw_).get()) != MAP_FAILED,
                "mmap failed.");
  }
  ROME_ASSERT_OK(RegisterMemoryRegion(kDefaultId, pd, 0, capacity_));
}

inline bool RdmaMemory::ValidateRegion(int offset, int length) {
  if (offset < 0 || length < 0)
    return false;
  if (offset + length > capacity_)
    return false;
  return true;
}

absl::Status RdmaMemory::RegisterMemoryRegion(std::string_view id, int offset,
                                              int length) {
  return RegisterMemoryRegion(id, GetDefaultMemoryRegion()->pd, offset, length);
}

absl::Status RdmaMemory::RegisterMemoryRegion(std::string_view id,
                                              ibv_pd *const pd, int offset,
                                              int length) {
  if (!ValidateRegion(offset, length)) {
    return FailedPreconditionErrorBuilder()
           << "Requested memory region invalid: " << id;
  }

  auto iter = memory_regions_.find(std::string(id));
  if (iter != memory_regions_.end()) {
    return AlreadyExistsErrorBuilder() << "Memory region exists: {}" << id;
  }

  auto *base = reinterpret_cast<uint8_t *>(std::visit(
                   [](const auto &raw) { return raw.get(); }, raw_)) +
               offset;
  auto mr = ibv_mr_unique_ptr(ibv_reg_mr(pd, base, length, kDefaultAccess));
  ROME_CHECK_QUIET(
      ROME_RETURN(absl::InternalError("Failed to register memory region")),
      mr != nullptr);
  memory_regions_.emplace(id, std::move(mr));
  ROME_DEBUG("Memory region registered: {} @ {} to {} (length={})", id,
             fmt::ptr(base), fmt::ptr(base + length), length);
  return absl::OkStatus();
}

ibv_mr *RdmaMemory::GetDefaultMemoryRegion() const {
  return memory_regions_.find(kDefaultId)->second.get();
}

absl::StatusOr<ibv_mr *>
RdmaMemory::GetMemoryRegion(std::string_view id) const {
  auto iter = memory_regions_.find(std::string(id));
  if (iter == memory_regions_.end()) {
    return NotFoundErrorBuilder() << "Memory region not found: {}" << id;
  }
  return iter->second.get();
}

} // namespace rome::rdma