#pragma once
#include "absl/status/status.h"

namespace rome {

// A wrapper for clients, which are effectively the interface to some system to
// be tested. It can wrap a data structure or it can wrap a client in a
// distributed system. The point is that it is an abstract way to represent
// applying an operation produced by a `Stream` and the entity performing the
// operation.
//
// As an example, consider a data structure with a simple set API (i.e., Set,
// Get, and Delete). An operation in this scenario would be some struct defining
// the operation type and the target key.
template <typename T>
class ClientAdaptor {
 public:
  virtual ~ClientAdaptor() = default;
  virtual absl::Status Start() = 0;
  virtual absl::Status Apply(const T &op) = 0;
  virtual absl::Status Stop() = 0;
};

}  // namespace rome