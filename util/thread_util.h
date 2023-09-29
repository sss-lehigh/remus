#pragma once

#include <cstdint>
#include <vector>

#define XSTR(x) STR(x)
#define STR(x) #x

static inline void cpu_relax() { asm volatile("pause\n" ::: "memory"); }

#define PAD_SIZEOF(type) (*(*(type))(nullptr))

#define __PAD(pad, x, prefix, id) [[maybe_unused]] char prefix##id[(pad) - (x)]

#define CACHELINE_SIZE 64
#define _CACHELINE_PAD(x, prefix, id) __PAD(CACHELINE_SIZE, x, prefix, id)
#define CACHELINE_PAD(x) _CACHELINE_PAD(x, __pad, __LINE__)
#define CACHELINE_PAD1(a) _CACHELINE_PAD(sizeof(a), __pad, __LINE__)
#define CACHELINE_PAD2(a, b) \
  _CACHELINE_PAD(sizeof(a) + sizeof(b), __pad, __LINE__)
#define CACHELINE_PAD3(a, b, c) \
  _CACHELINE_PAD(sizeof(a) + sizeof(b) + sizeof(c), __pad, __LINE__)
#define CACHELINE_PAD4(a, b, c, d) \
  _CACHELINE_PAD(sizeof(a) + sizeof(b) + sizeof(c) + sizeof(d), __pad, __LINE__)
#define CACHELINE_PAD5(a, b, c, d, e)                                       \
  _CACHELINE_PAD(sizeof(a) + sizeof(b) + sizeof(c) + sizeof(d) + sizeof(e), \
                 __pad, __LINE__)

#define PREFETCH_SIZE 128
#define _PREFETCH_PAD(x, prefix, id) __PAD(PREFETCH_SIZE, x, prefix, id)
#define PREFETCH_PAD() _PREFETCH_PAD(0, __pad, __LINE__)
#define PREFETCH_PAD1(a) _PREFETCH_PAD(sizeof(a), __pad, __LINE__)
#define PREFETCH_PAD2(a, b) \
  _PREFETCH_PAD(sizeof(a) + sizeof(b), __pad, __LINE__)
#define PREFETCH_PAD3(a, b, c) \
  _PREFETCH_PAD(sizeof(a) + sizeof(b) + sizeof(c), __pad, __LINE__)
#define PREFETCH_PAD4(a, b, c, d) \
  _PREFETCH_PAD(sizeof(a) + sizeof(b) + sizeof(c) + sizeof(d), __pad, __LINE__)
#define PREFETCH_PAD5(a, b, c, d, e)                                       \
  _PREFETCH_PAD(sizeof(a) + sizeof(b) + sizeof(c) + sizeof(d) + sizeof(e), \
                __pad, __LINE__)

#define EAGER_AWAIT(cond) \
  if (!(cond)) {          \
    while (!(cond))       \
      ;                   \
  }

#define RELAXED_AWAIT(cond) \
  if (!(cond)) {            \
    while (!(cond)) {       \
      cpu_relax();          \
    }                       \
  }

#define YIELDING_AWAIT(cond)     \
  if (!(cond)) {                 \
    cpu_relax();                 \
    while (!(cond)) {            \
      std::this_thread::yield(); \
    }                            \
  }

// A class to track thread binding and enforce various policies. The basic
// principle is to map a policy to a set of CPUs that threads are then bound to
// in a circular fashion. High-level policies can generate the CPU list. For
// example, NUMA-fill would schedule all cores in a NUMA zone then all
// hyperthreaded cores, before moving on to the next NUMA zone.
// template <typename Policy>
// class CpuBinder : public Policy {
//  public:
//   static CpuBinder& GetInstance() {
//     static CpuBinder instance;
//     return instance;
//   }

//   CpuBinder(const CpuBinder&) = delete;
//   void operator=(const CpuBinder&) = delete;

//   // Compile time polymorphism.
//   void Bind() { uint32_t cpu = static_cast<Policy*>(this)->GetNextCpu(); }

//  private:
//   CpuBinder();
// };

// class NumaFillPolicy {
//  public:
//   NumaFillPolicy() {}
//   void Bind();
// };