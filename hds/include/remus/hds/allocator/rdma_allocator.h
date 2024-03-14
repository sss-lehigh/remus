#include <remus/rdma/rdma.h>

#pragma once

namespace remus::hds::allocator {

class rdma_allocator {
public:
  rdma_allocator(remus::rdma::rdma_capability *ctx_) : ctx(ctx_) {}

  template <typename T> remus::rdma::rdma_ptr<T> allocate(size_t elements) { return ctx->Allocate<T>(elements); }

  template <typename T> void deallocate(remus::rdma::rdma_ptr<T> ptr, size_t elements) {
    return ctx->Deallocate(ptr, elements);
  }

private:
  remus::rdma::rdma_capability *ctx;
};

} // namespace remus::hds::allocator
