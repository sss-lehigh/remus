#pragma once

#include <algorithm>

namespace remus {

class EmptyMetadata {
public:
  typedef void *data_type;
  void set_metadata(data_type) { return; }
  data_type metadata() const { return nullptr; }
};

namespace internal {

template <typename T> class __attribute__((packed)) Metadata {
public:
  typedef T data_type;
  Metadata(const T &value) : metadata_(value) {}
  Metadata(const Metadata &copy) : metadata_(copy.metadata_) {}
  Metadata(Metadata &&copy) : metadata_(std::move(copy.metadata_)) {}

  inline void set_metadata(const data_type &metadata) { metadata_ = metadata; }
  inline data_type metadata() const { return metadata_; }

private:
  data_type metadata_;
};

} // namespace internal

} // namespace remus
