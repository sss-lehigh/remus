#pragma once

#include <cstdint>
#include <fstream>
#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <variant>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"

namespace rome::rdma::internal {

/// TODO: Update class-level documentation after this is fully
///       refactored/cleaned
///
/// The "remote memory partition map"
///
/// Each node in the system allocates a big region of memory, initializes an
/// allocator in it, and registers it with the RDMA device.  They then exchange
/// the names of these regions with each other.  This partition map keeps track
/// of all of those regions.
///
/// TODO: This isn't just the map... it is also responsible for making the
///       current process's big region of memory.  Can that be refactored?
///
/// TODO: Do we still want the following old documentation?
///
/// Remotely accessible memory backed by either raw memory or hugepages (if
/// enabled). This is just a flat buffer that is preallocated. The user of this
/// memory region can then use it directly, or wrap it around some more complex
/// allocation mechanism.
///
/// TODO: This is used by connection_manager.h... can we refactor?
class RdmaMemory {
  /// The name of the Linux proc file that contains the number of huge pages
  static constexpr char linux_hugepage_proc_path[] =
      "/proc/sys/vm/nr_hugepages";

  /// The default flags we use when registering a memory region with RDMA.  In
  /// our usage scenarios, we pretty much want everything turned on.
  static constexpr int kDefaultAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  /// TODO: Stop needing this
  static constexpr char kDefaultId[] = "default";

  // Preallocated size.
  const uint64_t capacity_;

  /// The mmap_deleter functor wraps a call to munmap.  This is used by the
  /// `rdma_unique_ptr` type, so that we can use a unique_ptr to hold raw memory
  /// regions returned by mmap or malloc, and have the region get reclaimed
  /// correctly on unique_ptr deallocation.
  struct mmap_deleter {
    void operator()(uint8_t raw[]) { munmap(raw, sizeof(*raw)); }
  };
  using mmap_unique_ptr = std::unique_ptr<uint8_t[], mmap_deleter>;
  using malloc_unique_ptr = std::unique_ptr<uint8_t[]>;
  using rdma_unique_ptr = std::variant<malloc_unique_ptr, mmap_unique_ptr>;

  /// The ibv_mr_deleter functor wraps a call to ibv_dereg_mr, so that we can
  /// use a unique_ptr to hold an RDMA memory region (ibv_mr*) and have the
  /// region get deregistered correctly on unique_ptr deallocation.
  struct ibv_mr_deleter {
    void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
  };
  using ibv_mr_unique_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

  /// A region of memory, pinned and registered with RDMA
  rdma_unique_ptr raw_;

  // A map of memory regions registered with this particular memory.
  //
  // TODO: It should be possible to get rid of this entirely
  std::unordered_map<std::string, ibv_mr_unique_ptr> memory_regions_;

  /// Try to read the number of available hugepages from the system.  Returns
  /// the number of huge pages on success, and a string error message on
  /// failure.
  ///
  /// NB: This is only implemented for Linux-based operating systems.
  ///
  /// TODO: We call this more than once, though it always returns the same
  ///       value.  It's not performance-critical, but still, it would be more
  ///       efficient to cache the result.
  ///
  /// TODO: We don't have any systems where this returns an integer > 0?
  static int GetNumHugepages() {
    using namespace std::string_literals;

    // NB: The file just contains a single int
    std::ifstream file(linux_hugepage_proc_path);
    if (!file.is_open()) {
      ROME_TRACE("Failed to open file: "s + linux_hugepage_proc_path);
      return 0;
    }
    int nr_hugepages;
    file >> nr_hugepages;
    if (file.fail()) {
      ROME_TRACE("Failed to read nr_hugepages");
      return 0;
    }
    return nr_hugepages;
  }

public:
  ~RdmaMemory() { memory_regions_.clear(); }

  /// Construct a slab of RDMA memory by allocating a region of memory (from
  /// hugepages if possible) and registering it with RDMA
  ///
  /// @param capacity The size (in bytes) of the region to allocate
  /// @param pd       The RDMA protection domain in which to register the memory
  RdmaMemory(uint64_t capacity, ibv_pd *const pd) : capacity_(capacity) {
    if (GetNumHugepages() <= 0) {
      ROME_TRACE("Not using hugepages; performance might suffer.");
      // [mfs] Does the next line turn 64B into 128B?  Looks like it's not just
      // rounding?
      auto bytes = ((capacity >> 6) + 1) << 6; // Round up to nearest 64B
      raw_ = malloc_unique_ptr((uint8_t *)(std::aligned_alloc(64, bytes)));
      ROME_ASSERT(std::get<0>(raw_) != nullptr, "Allocation failed.");
    } else {
      ROME_INFO("Using hugepages");
      raw_ = mmap_unique_ptr(reinterpret_cast<uint8_t *>(
          mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)));
      ROME_ASSERT((void *)(std::get<1>(raw_).get()) != MAP_FAILED,
                  "mmap failed.");
    }
    RegisterMemoryRegion(kDefaultId, pd, 0, capacity_);
  }

  RdmaMemory(const RdmaMemory &) = delete;
  RdmaMemory(RdmaMemory &&rm) = delete;
  RdmaMemory &operator=(const RdmaMemory &) = delete;

  // TODO: stop needing this
  uint8_t *raw() const {
    return std::visit([](const auto &r) { return r.get(); }, raw_);
  }

  /// Associate this RdmaMemory's `raw` region with protection domain `pd`.
  ///
  /// TODO: When we get rid of the map, this signature will change
  ///
  /// TODO: The offset/length stuff is rather annoying.  It's all there so that
  ///       we can allocate one slab, cut it in half, and use the halves for the
  ///       send and receive regions.  It would be much better if we just had
  ///       separate slabs for sending and receiving, even if that means wasting
  ///       a hugepage or two.
  ibv_mr *RegisterMemoryRegion(std::string id, ibv_pd *const pd, int offset,
                               int length) {
    using namespace std::string_literals;

    // TODO: This will go away if we get rid of splitting
    if (!ValidateRegion(offset, length)) {
      ROME_FATAL("RegisterMemoryRegion :: validation failed for "s + id)
      std::terminate();
    }

    // TODO: This will go away when we get rid of the map
    auto iter = memory_regions_.find(id);
    if (iter != memory_regions_.end()) {
      ROME_FATAL("RegisterMemoryRegion :: region already registered: "s + id);
      std::terminate();
    }

    // TODO: When the map goes away, I think this can just return the MR?
    auto *base = reinterpret_cast<uint8_t *>(std::visit(
                     [](const auto &raw) { return raw.get(); }, raw_)) +
                 offset;
    auto mr = ibv_mr_unique_ptr(ibv_reg_mr(pd, base, length, kDefaultAccess));
    if (mr == nullptr) {
      ROME_FATAL("RegisterMemoryRegion :: ibv_reg_mr failed")
      std::terminate();
    }
    auto res = mr.get();
    memory_regions_.emplace(id, std::move(mr));
    ROME_TRACE("Memory region registered: {} @ {} to {} (length={})", id,
               fmt::ptr(base), fmt::ptr(base + length), length);
    return res;
  }

  // TODO: Get rid of this?
  ibv_mr *GetDefaultMemoryRegion() const {
    return memory_regions_.find(kDefaultId)->second.get();
  }

private:
  // Validates that the given offset and length are not ill formed w.r.t. to the
  // capacity of this memory.
  //
  // [mfs] Why can't we use unsigned ints for offset and length?
  bool ValidateRegion(int offset, int length) {
    if (offset < 0) {
      ROME_DEBUG("ValidateRegion :: offset < 0");
      return false;
    }
    if (length < 0) {
      ROME_DEBUG("ValidateRegion :: length < 0");
      return false;
    }
    if (offset + length > capacity_) {
      ROME_DEBUG("ValidateRegion :: offset + length > capacity_");
      return false;
    }
    return true;
  }
};
}