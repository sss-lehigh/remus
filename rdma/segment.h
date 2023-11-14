#pragma once

#include <cstdint>
#include <fstream>
#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <variant>

#include "../logging/logging.h"
#include "../vendor/sss/status.h"

namespace rome::rdma::internal {

/// Remotely accessible memory backed by either raw memory or huge pages (if
/// enabled). This is just a flat buffer that is preallocated. The user of this
/// memory region can then use it directly, or wrap it around some more complex
/// allocation mechanism.  Note that the buffer gets pinned by the RDMA device.
class Segment {
  /// The name of the Linux proc file that contains the number of huge pages
  static constexpr char HUGE_PAGE_PATH[] = "/proc/sys/vm/nr_hugepages";

  /// The default flags we use when registering a memory region with RDMA.  In
  /// our usage scenarios, we pretty much want everything turned on.
  static constexpr int DEFAULT_ACCESS_MODE =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  /// The mmap_deleter functor wraps a call to munmap.  This is used by the
  /// `rdma_unique_ptr` type, so that we can use a unique_ptr to hold raw memory
  /// regions returned by mmap or malloc, and have the region get reclaimed
  /// correctly on unique_ptr deallocation (i.e., when this Segment goes out of
  /// scope).
  struct mmap_deleter {
    void operator()(uint8_t raw[]) { munmap(raw, sizeof(*raw)); }
  };
  using mmap_unique_ptr = std::unique_ptr<uint8_t[], mmap_deleter>;
  using malloc_unique_ptr = std::unique_ptr<uint8_t[]>;
  using rdma_unique_ptr = std::variant<malloc_unique_ptr, mmap_unique_ptr>;

  /// The ibv_mr_deleter functor wraps a call to ibv_dereg_mr, so that we can
  /// use a unique_ptr to hold an RDMA memory region (ibv_mr*) and have the
  /// region get deregistered correctly on unique_ptr deallocation (i.e., when
  /// this Segment goes out of scope).
  struct ibv_mr_deleter {
    void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
  };
  using ibv_mr_unique_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

  const uint64_t capacity_;         // Size of the memory segment
  rdma_unique_ptr raw_;             // Pointer to the raw memory segment
  ibv_mr_unique_ptr memory_region_; // RDMA object for accessing the segment

  /// Try to read the number of available huge pages from the system.  Returns
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
  static int GetNumHugePages() {
    using namespace std::string_literals;

    // NB: The file just contains a single int
    std::ifstream file(HUGE_PAGE_PATH);
    if (!file.is_open()) {
      ROME_TRACE("Failed to open file: "s + HUGE_PAGE_PATH);
      return 0;
    }
    int nr_huge_pages;
    file >> nr_huge_pages;
    if (file.fail()) {
      ROME_TRACE("Failed to read nr_huge_pages");
      return 0;
    }
    return nr_huge_pages;
  }

public:
  /// Construct a slab of RDMA memory by allocating a region of memory (from
  /// huge pages if possible) and registering it with RDMA
  ///
  /// @param capacity The size (in bytes) of the region to allocate
  /// @param pd       The RDMA protection domain in which to register the memory
  Segment(uint64_t capacity, ibv_pd *pd) : capacity_(capacity) {
    using namespace std::string_literals;

    // Allocate raw memory
    if (GetNumHugePages() <= 0) {
      ROME_TRACE("Not using huge pages; performance might suffer.");
      // [mfs] Does the next line turn 64B into 128B?  Looks like it's not just
      // rounding?
      auto bytes = ((capacity >> 6) + 1) << 6; // Round up to nearest 64B
      raw_ = malloc_unique_ptr((uint8_t *)(std::aligned_alloc(64, bytes)));
      ROME_ASSERT(std::get<0>(raw_) != nullptr, "Allocation failed.");
    } else {
      ROME_INFO("Using huge pages");
      raw_ = mmap_unique_ptr(reinterpret_cast<uint8_t *>(
          mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)));
      ROME_ASSERT((void *)(std::get<1>(raw_).get()) != MAP_FAILED,
                  "mmap failed.");
    }

    // Register the memory with RDMA
    auto *base = reinterpret_cast<uint8_t *>(
        std::visit([](const auto &raw) { return raw.get(); }, raw_));
    memory_region_ =
        ibv_mr_unique_ptr(ibv_reg_mr(pd, base, capacity_, DEFAULT_ACCESS_MODE));
    if (memory_region_ == nullptr) {
      ROME_FATAL("RegisterMemoryRegion :: ibv_reg_mr failed")
    }
    ROME_TRACE("Memory region registered: @ {} to {} (length={})",
               fmt::ptr(base), fmt::ptr(base + capacity_), capacity_);
  }

  Segment(const Segment &) = delete;
  Segment(Segment &&rm) = delete;
  Segment &operator=(const Segment &) = delete;

  /// Return the underlying raw memory region
  uint8_t *raw() const {
    return std::visit([](const auto &r) { return r.get(); }, raw_);
  }

  /// Return the underlying RDMA object
  ibv_mr *mr() { return memory_region_.get(); }
};
} // namespace rome::rdma::internal