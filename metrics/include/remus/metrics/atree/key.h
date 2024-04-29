#pragma once

#include "macros.h"

namespace remus {

// Necessary forward declaration (used in `ATreeRequiredInfo`).
template <class Key, class V, class M, typename value_type, typename metadata_type> class ATreeNode;

// Encapsulates the required information of a node, these are never empty-base
// optimized. The simplest form of the `ATree` is a set containing only nodes
// with a key and pointers to its children.
template <class Key, class... OptionalInfo> class REMUS_PACKED ATreeNodeRequiredInfo {
  typedef ATreeNode<Key, OptionalInfo...> node_type;

public:
  ATreeNodeRequiredInfo(Key key) : key_(key) {
    children_[0] = nullptr;
    children_[1] = nullptr;
  }
  inline Key key() const { return key_; }

  inline node_type *left() const { return children_[0]; }
  inline node_type *right() const { return children_[1]; }

  inline void set_left(node_type *n) { children_[0] = n; }
  inline void set_right(node_type *n) { children_[1] = n; }

private:
  const Key key_;
  node_type *children_[2];
};

} // namespace remus
