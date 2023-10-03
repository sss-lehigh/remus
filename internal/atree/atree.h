#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "../../util/macros.h"
#include "metadata.h"
#include "value.h"
#include "visitor.h"

// `ATree` implements an augmented binary search tree that uses empty-base
// optimization to represent both an augmented set and a map. The below classes
// are empty classes passed to the template when creating the object to disable
// various parts of the underlying representation. For example,
//
//    ATree<int, EmptyValue, EmptyMetadata, EmptyVisistor>
//
// is just a set while
//
//    ATree<int, EmptyValue, int, CountTraversalsVisitor<NodeType>>
//
// augments the datastructure so that it counts the number of times a node is
// traversed.
//
// Note that when constructing an `ATree` only the internal representation is
// passed in the template and the object takes care of wrapping it for internal
// use. Meanwhile, the Visitor must implement both `OnInsert` and `OnRemove`.
// The definition for the above struct would be:
//
// template <typename NodeType>
// struct CountTraversalsVisitor {
//  void OnInsert(NodeType* curr) { curr->set_metadata(curr->metadata() + 1);  }
//  void OnRemove(NodeType* curr) { curr->set_metadata(curr->metadata() +  1); }
// };

namespace rome {

// Necessary forward declaration (used in `ATreeRequiredInfo`).
template <class Key, class V, class M, typename value_type,
          typename metadata_type>
class ATreeNode;

// Encapsulates the required information of a node, these are never empty-base
// optimized. The simplest form of the `ATree` is a set containing only nodes
// with a key and pointers to its children.
template <class Key, class... OptionalInfo>
class __attribute__((packed)) ATreeNodeRequiredInfo {
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

// Represents the optional information that is empty-base optimized whenever
// either the `ValueType` or the `MetadataType` is empty. This makes the node
// more memory efficient since we do not allocate any memory for the empty base
// (https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Empty_Base_Optimization).
template <class ValueType, class MetadataType>
class __attribute__((packed)) ATreeNodeOptionalInfo : ValueType, MetadataType {
public:
  ATreeNodeOptionalInfo(const ValueType &value, const MetadataType &metadata)
      : ValueType(value), MetadataType(metadata) {}

  // Getters
  inline typename ValueType::data_type value() const {
    return dynamic_cast<const ValueType *>(this)->value();
  }
  inline typename MetadataType::data_type metadata() const {
    return dynamic_cast<const MetadataType *>(this)->metadata();
  }

  // Setters
  template <class V = ValueType,
            std::enable_if_t<!std::is_empty<V>::value, bool> = true>
  inline void set_value(const typename V::data_type &value) {
    dynamic_cast<V *>(this)->set_value(value);
  }
  template <class M = MetadataType,
            std::enable_if_t<!std::is_empty<M>::value, bool> = true>
  inline void set_metadata(const typename M::data_type &metadata) {
    dynamic_cast<M *>(this)->set_metadata(metadata);
  }
};

// Encapsulates both the required and optional information in the node and
// provides an API that abstracts away the internal representation of values and
// metadata. If concrete types are passed for `V` and `M` template parameters,
// then they are wrapped in the respective classes (see `Value` and `Metadata`
// above).
template <class Key, class V, class M,
          typename ValueType = typename std::conditional<
              std::is_empty<V>::value, EmptyValue, internal::Value<V>>::type,
          typename MetadataType =
              typename std::conditional<std::is_empty<M>::value, EmptyMetadata,
                                        internal::Metadata<M>>::type>
class ATreeNode : protected ATreeNodeRequiredInfo<Key, V, M>,
                  protected ATreeNodeOptionalInfo<ValueType, MetadataType> {
  typedef ATreeNodeRequiredInfo<Key, V, M> required_info;
  typedef ATreeNodeOptionalInfo<ValueType, MetadataType> optional_info;

public:
  typedef V value_type;
  typedef M metadata_type;

  ATreeNode(const Key &key, const V &value, const M &metadata)
      : required_info(key),
        optional_info(ValueType(value), MetadataType(metadata)) {}

  // Getters
  inline Key key() const { return required_info::key(); }
  inline typename ValueType::data_type value() const {
    return optional_info::value();
  }
  inline typename MetadataType::data_type metadata() {
    return optional_info::metadata();
  }
  inline ATreeNode *left() const { return required_info::left(); }
  inline ATreeNode *right() const { return required_info::right(); }

  // Setters
  template <class _V_t = ValueType,
            std::enable_if_t<!std::is_empty<_V_t>::value, bool> = true>
  inline void set_value(const V &value) {
    optional_info::set_value(value);
  }
  template <class _M_t = MetadataType,
            std::enable_if_t<!std::is_empty<_M_t>::value, bool> = true>
  inline void set_metadata(const M &metadata) {
    optional_info::set_metadata(metadata);
  }
  inline void set_left(ATreeNode *n) { required_info::set_left(n); }
  inline void set_right(ATreeNode *n) { required_info::set_right(n); }
};

template <typename T> struct AccessorWrapper {
  typedef T type;
};

// The augmented binary search tree implementation. The tree is built using
// `ATreeNode`s that hold various data and metadata. A user can disable either
// values or metadata by passing in empty base classes (i.e., `EmptyValue` and
// `EmptyMetadata`) as the template parameters.
template <class Key, class V, class M, class Visitor, class Accessor = void>
class ATree {
public:
  ~ATree();
  ATree() : size_(0), root_(Key(), V(), M()) {}
  typedef ATreeNode<Key, V, M> node_type;
  std::optional<node_type *> Find(const Key &key);

  bool Insert(const Key &key, const V &value, const M &metadata);

  void InsertOrUpdate(const Key &key, const V &value, const M &metadata);

  bool Remove(const Key &key);

  void Clear();

  inline int Size() const { return size_; }

private:
  friend Accessor;

  // Returns a pair of node pointers to the parent of where `key` is located,
  // and to the node corresponding to `key` if one exists. If `stack` is not
  // `std::nullopt` then all nodes found in the traversal will be pushed onto
  // the back of the stack.
  std::pair<node_type *, node_type *>
  FindInternal(const Key &key, std::optional<std::deque<node_type *> *> stack);

  // Returns a pointer to a newly inserted node initialized with `key`, `value`
  // and `metadata`. The inserted node will be the child of `parent`.
  node_type *InsertInternal(const Key &key, const V &value, const M &metadata,
                            node_type *parent);

  std::pair<node_type *, node_type *> FindSuccessors(node_type *curr);

  int size_;
  node_type root_;
  Visitor visitor_;
  std::vector<node_type *> cleared_;
};

// ---------------|
// Implementation |
// ---------------|

template <class K, class V, class M, class Visitor, class Accessor>
ATree<K, V, M, Visitor, Accessor>::~ATree() {
  // if (root_ == nullptr) return;  // Nothing to do
  cleared_.push_back(root_.left());
  for (auto *c : cleared_) {
    std::deque<node_type *> to_delete;
    to_delete.push_back(c);
    while (!to_delete.empty()) {
      auto curr = to_delete.back();
      to_delete.pop_back();
      if (curr != nullptr) {
        to_delete.push_back(curr->left());
        to_delete.push_back(curr->right());
        delete curr;
      }
    }
  }
}

template <class K, class V, class M, class Visitor, class Accessor>
std::pair<ATreeNode<K, V, M> *, ATreeNode<K, V, M> *>
ATree<K, V, M, Visitor, Accessor>::FindInternal(
    const K &key, std::optional<std::deque<node_type *> *> stack) {
  node_type *parent = &root_;
  node_type *curr = root_.left();
  while (curr != nullptr && curr->key() != key) {
    parent = curr;
    if (key > curr->key()) {
      curr = curr->right();
    } else {
      curr = curr->left();
    }
    if (stack.has_value())
      stack.value()->push_back(parent);
  }
  return {parent, curr};
}

template <class K, class V, class M, class Visitor, class Accessor>
std::optional<ATreeNode<K, V, M> *>
ATree<K, V, M, Visitor, Accessor>::Find(const K &key) {
  auto found = FindInternal(key, std::nullopt);
  if (found.second != nullptr && found.second->key() == key) {
    return {found.second};
  } else {
    return {};
  }
}

template <class K, class V, class M, class Visitor, class Accessor>
typename ATree<K, V, M, Visitor, Accessor>::node_type *
ATree<K, V, M, Visitor, Accessor>::InsertInternal(const K &key, const V &value,
                                                  const M &metadata,
                                                  node_type *parent) {
  node_type *new_node = new node_type(key, value, metadata);
  if (parent == &root_) {
    parent->set_left(new_node);
  } else if (key < parent->key()) {
    parent->set_left(new_node);
  } else {
    parent->set_right(new_node);
  }
  ++size_;
  return new_node;
}

template <class K, class V, class M, class Visitor, class Accessor>
bool ATree<K, V, M, Visitor, Accessor>::Insert(const K &key, const V &value,
                                               const M &metadata) {
  std::deque<node_type *> stack;
  auto found = FindInternal(key, &stack);
  if (found.second != nullptr && found.second->key() == key) {
    return false;
  } else {
    auto *new_node = InsertInternal(key, value, metadata, found.first);
    visitor_.OnInsert(new_node);
    while (!stack.empty()) {
      visitor_.OnInsert(stack.back());
      stack.pop_back();
    }
    return true;
  }
}

template <class K, class V, class M, class Visitor, class Accessor>
void ATree<K, V, M, Visitor, Accessor>::InsertOrUpdate(const K &key,
                                                       const V &value,
                                                       const M &metadata) {
  std::deque<node_type *> stack;
  auto found = FindInternal(key, &stack);
  if (found.second != nullptr && found.second->key() == key) {
    visitor_.Update(found.second, value, metadata);
    while (!stack.empty()) {
      visitor_.OnUpdate(stack.back());
      stack.pop_back();
    }
  } else {
    auto *new_node = InsertInternal(key, value, metadata, found.first);
    visitor_.OnInsert(new_node);
    while (!stack.empty()) {
      visitor_.OnInsert(stack.back());
      stack.pop_back();
    }
  }
}

namespace internal {

template <typename NodeType>
inline void ReplaceChild(NodeType *parent, const NodeType *child,
                         NodeType *new_node) {
  if (parent->left() == child) {
    parent->set_left(new_node);
  } else {
    assert(parent->right() == child);
    parent->set_right(new_node);
  }
}

} // namespace internal

template <class K, class V, class M, class Visitor, class Accessor>
std::pair<ATreeNode<K, V, M> *, ATreeNode<K, V, M> *>
ATree<K, V, M, Visitor, Accessor>::FindSuccessors(ATreeNode<K, V, M> *curr) {
  if (curr == nullptr)
    return {nullptr, nullptr};
  if (curr->right() == nullptr)
    return {curr, nullptr};
  auto parent = curr;
  curr = curr->right();
  while (curr->left() != nullptr) {
    curr = curr->left();
  }
  return {parent, curr};
}

template <class K, class V, class M, class Visitor, class Accessor>
bool ATree<K, V, M, Visitor, Accessor>::Remove(const K &key) {
  std::deque<node_type *> stack;
  auto found = FindInternal(key, &stack);
  if (found.second == nullptr) {
    return false;
  } else {
    auto successors = FindSuccessors(found.second);
    if (successors.second == nullptr) {
      // No successor exists.
      internal::ReplaceChild<node_type>(found.first, found.second, nullptr);
    } else if (successors.first == found.second) {
      // A successor exists but it is the child of the node being removed.
      successors.second->set_left(found.second->left());
      successors.second->set_right(found.second->right());
      internal::ReplaceChild<node_type>(found.first, found.second,
                                        successors.second);
    } else {
      // Otherwise, a successor exists and its parent must be dealt with.
      successors.first->set_left(successors.second->right());
      successors.second->set_left(found.second->left());
      successors.second->set_right(found.second->right());
      internal::ReplaceChild<node_type>(found.first, found.second,
                                        successors.second);
    }

    while (!stack.empty()) {
      visitor_.OnRemove(stack.back());
      stack.pop_back();
    }
    --size_;
    delete found.second;
    return true;
  }
}

template <class K, class V, class M, class Visitor, class Accessor>
void ATree<K, V, M, Visitor, Accessor>::Clear() {
  auto old = root_.left();
  cleared_.push_back(old);
  root_.set_left(nullptr);
  size_ = 0;
}

} // namespace rome