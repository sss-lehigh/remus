#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <infiniband/verbs.h>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <sys/param.h>
#include <variant>

#include "logging.h"
#include "util.h"

// TODO: This is functional, but:
// 1. Should we be restricting sizes more?  Is 2^20 a reasonable minimum?
// 2. Review documentation and destructors
namespace remus::internal {
/// find_mmap_location Uses /proc/self/maps to find an aligned region of memory
/// that is not currently mapped into the address space, so that it can be used
/// for an RDMA segment.
///
/// The idea behind this is that we can use find_mmap_location to get an
/// address, and then pass that address to mmap (using MAP_FIXED_NOREPLACE) in
/// order to get an aligned memory chunk.
///
/// NB: There is a TOCTOU issue with this code.  After calling it, before
///     passing the result to mmap, you must not do any allocs, thread
///     creations, or mmaps that could lead to another call to sbrk or mmap.
///
/// TODO: Right now, we're calling this on each mmap.  We could scan for a
///       bigger contiguous region to avoid that overhead, but it's probably not
///       worth it.
///
/// TODO: Right now, we're using roundup(), which has a risk of overflow.
///       Eventually look into something a bit more robust.
///
/// @param min_addr The smallest address that we're willing to use.  It is
///                 probably a good idea to not use anything too small.
/// @param len      The size of the region to allocate.  This will also be used
///                 as the alignment.  It should be a nonzero power of 2.
///
/// @return An address that could be passed to MAP_FIXED_NOREPLACE, NONE on any
///         error
inline std::optional<uintptr_t> find_mmap_location(uintptr_t min_addr,
                                                   size_t len) {
  // Ensure non-zero power of two for min_addr and len
  assert(min_addr && !(min_addr & (min_addr - 1)));
  if (!(len && !(len & (len - 1)))) {
    REMUS_FATAL("len is not a power of 2");
  }

  // Round the initial guess up to the next multiple of len
  uintptr_t addr = roundup(min_addr, len);

  // Start going through the procfile.  We assume each line is <= 2048B
  //
  // The format is: `l-u p o d i f`
  // - l, u:  The lower and upper bound of the valid memory region (hex)
  // - p:     The permissions of that region
  // - o:     The offset of the file that is mapped (if it's a mmap'd file)
  // - d:     The (hex) device number if it's a mmap'd file
  // - i:     The inode / file number if it's a mmap'd file
  // - f:     The filename if it's a mmap'd file
  //
  // Note that lines are sorted by l, and that entries do not overlap.  Thus if
  // we pull in each line, then use scanf to get l-u, we have all the info we
  // need to know that a given region is mapped.  Everything else is available
  FILE *fp;
  if ((fp = fopen("/proc/self/maps", "r")) == NULL) {
    return {};
  }
  char line[2048];
  while (fgets(line, 2048, fp) != NULL) {
    uintptr_t l = 0, u = 0; // Lower and upper bounds of the ranges in the file
    if (sscanf(line, "%lx-%lx", &l, &u) == 2) {
      if ((addr + len) <= l) {
        break; // addr is likely to work!
      }
      if (addr < u) {
        addr = roundup(u, len); // Move past u, to next multiple of len
      }
    }
  }
  fclose(fp);

  // Check for overflow: If we would run off the edge of the address map, fail
  if ((addr + len) < addr) {
    return {};
  }
  return addr;
}


/// @brief A Segment object to represent a slab of RDMA memory
/// @details
/// A contiguous region of remotely accessible memory.  Size is always a power
/// of 2, and it's always aligned to its size.  The interface is really just
/// "raw pointer".  If you want something more complex (e.g., an allocator
/// inside of it), you need to make it yourself.
class Segment {
  Segment(const Segment &) = delete;            // No copy constructor
  Segment &operator=(const Segment &) = delete; // No copy assignment operator
  Segment &operator=(Segment &&) = delete;      // No move assignment operator

  /// The name of the Linux proc file that contains the number of huge pages
  static constexpr char HUGE_PAGE_PATH[] = "/proc/sys/vm/nr_hugepages";

  /// The default flags we use when registering a memory region with RDMA.  In
  /// our usage scenarios, we pretty much want everything turned on.
  static constexpr int DEFAULT_ACCESS_MODE =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  const uint64_t capacity_; // Size of the memory segment
  uint8_t *raw_;            // Pointer to the raw memory segment
  bool from_huge_;          // Was this allocated via huge pages?

  /// Return the number of available Huge Pages, so we know if it's worth trying
  /// to use them.
  ///
  /// NB: This only works on Linux
  ///
  /// TODO: When the unit of allocation does not match the huge page size, this
  ///       may not be valid.  The r320s on CloudLab don't have huge page
  ///       support turned on, so it isn't really an issue for now, but
  ///       eventually this is something we should work on.
  static int GetNumHugePages() {
    // NB: The file just contains a single int
    std::ifstream file(HUGE_PAGE_PATH);
    if (!file.is_open()) {
      REMUS_DEBUG("Failed to open file: {}", HUGE_PAGE_PATH);
      return 0;
    }
    int nr_huge_pages;
    file >> nr_huge_pages;
    if (file.fail()) {
      REMUS_DEBUG("Failed to read nr_huge_pages");
      return 0;
    }
    return nr_huge_pages;
  }

public:
  /// Destruct by unmapping the region, using the capacity.
  ~Segment() { munmap((void *)raw_, capacity_); }

  Segment(Segment &&) = default; // Default move constructor

  /// Construct a slab of RDMA memory by allocating a region of memory (from
  /// huge pages if possible) and registering it with RDMA
  ///
  /// @param cap  The size (in bytes) of the region to allocate
  Segment(uint64_t cap) : capacity_(cap) {
    // Get aligned memory via mmap
    auto hint = find_mmap_location((1UL << 35), capacity_);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE;
    if (GetNumHugePages() <= 0) {
      from_huge_ = false;
    } else {
      from_huge_ = true;
      flags = flags | MAP_HUGETLB;
    }
    raw_ = (uint8_t *)mmap((void *)hint.value(), capacity_,
                           PROT_READ | PROT_WRITE, flags, -1, 0);
    REMUS_ASSERT(((void *)raw_) != MAP_FAILED, "mmap failed.");
  }

  /// Register this Segment with a Protection Domain, so that the RNIC can use
  /// this memory region
  ///
  /// @param pd
  /// @return
  [[nodiscard]]
  ibv_mr_ptr registerWithPd(ibv_pd *pd) {
    if (pd == nullptr) {
      REMUS_FATAL("Cannot register segment with null PD");
    }

    // See <https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html>
    auto flags = from_huge_ ? DEFAULT_ACCESS_MODE | IBV_ACCESS_HUGETLB
                            : DEFAULT_ACCESS_MODE;
    auto ptr = ibv_reg_mr(pd, raw_, capacity_, flags);
    if (ptr == nullptr) {
      REMUS_FATAL("RegisterMemoryRegion :: ibv_reg_mr failed: {}",
                  strerror(errno))
    }
    REMUS_INFO("  Registered region 0x{:x} (length=0x{:x}) ({} pages)",
               (uintptr_t)(raw_), capacity_, from_huge_ ? "2MB" : "4KB");
    return ibv_mr_ptr(std::move(ptr));
  }

  /// Return the local address of the start of this Segment
  uint8_t *raw() const { return raw_; }

  /// Return the Segment size
  uint64_t capacity() const { return capacity_; }
};
} // namespace remus::internal
