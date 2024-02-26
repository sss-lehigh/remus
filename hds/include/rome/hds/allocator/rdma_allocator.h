#include <rome/rdma/rdma.h>

#pragma once

namespace rome::hds::allocator {

struct rdma_allocator {

  rdma_allocator(rome::rdma::rdma_capability* ctx_) : ctx(ctx_) {}

  template<typename T>
  rome::rdma::rdma_ptr<T> allocate(size_t elements) {
    return ctx->Allocate<T>(elements);
  }

  template<typename T>
  void deallocate(rome::rdma::rdma_ptr<T> ptr, size_t elements) {
    return ctx->Deallocate(ptr, elements);
  }

  rome::rdma::rdma_capability* ctx;
};

}

