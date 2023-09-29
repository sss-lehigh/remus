#pragma once

#include <deque>
#include <experimental/memory_resource>
#include <list>
#include <memory>

#include "../../util/memory_resource.h"
#include "../rdma_memory.h"

template class std::unordered_map<size_t, std::deque<void *>>;

namespace rome::rdma {

// Specialization of a `memory_resource` that wraps RDMA accessible memory.
class rdma_memory_resource : public util::pmr::memory_resource {
public:
  virtual ~rdma_memory_resource() {}
  explicit rdma_memory_resource(size_t bytes, ibv_pd *pd)
      : rdma_memory_(std::make_unique<RdmaMemory>(
            bytes, "/proc/sys/vm/nr_hugepages", pd)),
        head_(rdma_memory_->raw() + bytes) {
    std::memset(alignments_.data(), 0, sizeof(alignments_));
    ROME_DEBUG("rdma_memory_resource: {} to {} (length={})",
               fmt::ptr(rdma_memory_->raw()), fmt::ptr(head_.load()), bytes);
  }

  rdma_memory_resource(const rdma_memory_resource &) = delete;
  rdma_memory_resource &operator=(const rdma_memory_resource &) = delete;
  ibv_mr *mr() const { return rdma_memory_->GetDefaultMemoryRegion(); }

private:
  static constexpr uint8_t kMinSlabClass = 3;
  static constexpr uint8_t kMaxSlabClass = 20;
  static constexpr uint8_t kNumSlabClasses = kMaxSlabClass - kMinSlabClass + 1;
  static constexpr size_t kMaxAlignment = 1 << 8;
  static constexpr char kLogTable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
      -1,    0,     1,     1,     2,     2,     2,     2,
      3,     3,     3,     3,     3,     3,     3,     3,
      LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6), LT(7),
      LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7),
  };

  inline unsigned int UpperLog2(size_t x) {
    size_t r;
    size_t p, q;
    if ((q = x >> 16)) {
      r = (p = q >> 8) ? 24 + kLogTable[p] : 16 + kLogTable[q];
    } else {
      r = (p = x >> 8) ? 8 + kLogTable[p] : kLogTable[x];
    }
    return ((1ul << r) < x) ? r + 1 : r;
  }

  // Returns a region of RDMA-accessible memory that satisfies the given memory
  // allocation request of `bytes` with `alignment`. First, it checks whether
  // there exists a region in `freelists_` that satisfies the request, then it
  // attempts to allocate a new region. If the request cannot be satisfied, then
  // `nullptr` is returned.
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (alignment > bytes)
      bytes = alignment;
    auto slabclass = UpperLog2(bytes);
    slabclass = std::max(kMinSlabClass, static_cast<uint8_t>(slabclass));
    auto slabclass_idx = slabclass - kMinSlabClass;
    ROME_ASSERT(slabclass_idx >= 0 && slabclass_idx < kNumSlabClasses,
                "Invalid allocation requested: {} bytes", bytes);
    ROME_ASSERT(alignment <= kMaxAlignment, "Invalid alignment: {} bytes",
                alignment);

    if (alignments_[slabclass_idx] & alignment) {
      auto *freelist = &freelists_[slabclass_idx];
      ROME_ASSERT_DEBUG(!freelist->empty(), "Freelist should not be empty");
      for (auto f = freelist->begin(); f != freelist->end(); ++f) {
        if (f->first >= alignment) {
          auto *ptr = f->second;
          f = freelist->erase(f);
          if (f == freelist->end())
            alignments_[slabclass_idx] &= ~alignment;
          std::memset(ptr, 0, 1 << slabclass);
          ROME_TRACE("(Re)allocated {} bytes @ {}", bytes, fmt::ptr(ptr));
          return ptr;
        }
      }
    }

    uint8_t *__e = head_, *__d;
    do {
      __d = (uint8_t *)((uint64_t)__e & ~(alignment - 1)) - bytes;
      if ((void *)(__d) < rdma_memory_->raw()) {
        ROME_CRITICAL("OOM!");
        return nullptr;
      }
    } while (!head_.compare_exchange_strong(__e, __d));

    ROME_TRACE("Allocated {} bytes @ {}", bytes, fmt::ptr(__d));
    return reinterpret_cast<void *>(__d);
  }

  void do_deallocate(void *p, size_t bytes, size_t alignment) override {
    ROME_TRACE("Deallocating {} bytes @ {}", bytes, fmt::ptr(p));
    auto slabclass = UpperLog2(bytes);
    slabclass = std::max(kMinSlabClass, static_cast<uint8_t>(slabclass));
    auto slabclass_idx = slabclass - kMinSlabClass;

    alignments_[slabclass_idx] |= alignment;
    freelists_[slabclass_idx].emplace_back(alignment, p);
  }

  // Only equal if they are the same object.
  bool
  do_is_equal(const util::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

  std::unique_ptr<RdmaMemory> rdma_memory_;
  std::atomic<uint8_t *> head_;

  // Stores addresses of freed memory for a given slab class.
  inline static thread_local std::array<uint8_t, kNumSlabClasses> alignments_;
  inline static thread_local std::array<std::list<std::pair<size_t, void *>>,
                                        kNumSlabClasses>
      freelists_;
};

// An allocator wrapping `rdma_memory_resouce` to be used to allocate new
// RDMA-accessible memory.
template <typename T> class rdma_allocator {
public:
  typedef T value_type;

  rdma_allocator() : memory_resource_(nullptr) {}
  rdma_allocator(rdma_memory_resource *memory_resource)
      : memory_resource_(memory_resource) {}
  rdma_allocator(const rdma_allocator &other) = default;

  template <typename U>
  rdma_allocator(const rdma_allocator<U> &other) noexcept {
    memory_resource_ = other.memory_resource();
  }

  rdma_allocator &operator=(const rdma_allocator &) = delete;

  // Getters
  rdma_memory_resource *memory_resource() const { return memory_resource_; }

  [[nodiscard]] constexpr T *allocate(std::size_t n = 1) {
    return reinterpret_cast<T *>(memory_resource_->allocate(sizeof(T) * n, 64));
  }

  constexpr void deallocate(T *p, std::size_t n = 1) {
    memory_resource_->deallocate(reinterpret_cast<T *>(p), sizeof(T) * n, 64);
  }

private:
  rdma_memory_resource *memory_resource_;
};

} // namespace rome::rdma