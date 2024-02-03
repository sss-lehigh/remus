#pragma once

namespace rome {

template <class NodeType, class V, class M> class VisitorInterface {
public:
  ~VisitorInterface() = default;

  // Called on all nodes in a newly inserted nodes path (including itself).
  virtual void OnInsert(NodeType *) = 0;

  // Called on all nodes in a removed node's path.
  virtual void OnRemove(NodeType *) = 0;

  // Called on all nodes in the updated node's path (excluding the updated
  // node).
  virtual void OnUpdate(NodeType *) = 0;

  // Called on the node that is updated. The second argument will be the value
  // passed by user to the update call that in turn calls the visitor. The
  // thirds is the metadata also passed during the initial call that ends up
  // here.
  virtual void Update(NodeType *, const V &, const M &) = 0;
};

class EmptyVisitor : public VisitorInterface<void, void *, void *> {
public:
  void OnInsert(void *) override { return; }
  void OnRemove(void *) override { return; }
  void Update(void *, void *const &, void *const &) override { return; }
  void OnUpdate(void *) override { return; }
};

} // namespace rome