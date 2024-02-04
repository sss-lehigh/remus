#pragma once

namespace hds::vcas {

/// TBD value
template<typename int_t>
struct tbd {
  /// Value is -1 for most types
  static constexpr int_t value = -1;
};

/// TBD_v constexpr
template<typename int_t>
inline constexpr int_t tbd_v = tbd<int_t>::value;

/// Camera object
template<typename int_t_, bool increment_on_snapshot>
class camera {
public:
  /// Timestamp type
  using int_t = int_t_;

  /// Default constructor
  camera() = default;
 
  /// Take a snapshot
  HDS_HOST_DEVICE int_t take_snapshot() {
    if constexpr (IncrementOnSnapshot) {
      return increment_timestamp();
    } else {
      return timestamp();
    }
  }

  /// Increment the timestamp
  HDS_HOST_DEVICE int_t increment_timestamp() {
    auto ts = timestamp_.load(memory_order_relaxed);
    auto expected = ts;
    timestamp_.compare_exchange_strong(expected, ts + 1, memory_order_relaxed);
    return ts;
  }

  /// Get the timestamp
  HDS_HOST_DEVICE int_t timestamp() {
    return timestamp_.load(memory_order_relaxed);
  }

private:
  /// Timestamp
  alignas(hardware_destructive_interference_size) atomic<int_t> timestamp_ = 0;

  static_assert(atomic<int_t>::is_always_lock_free);
};

template<typename T, typename int_t>
struct vnode {
 
  /// Create vnode with value and next
  HDS_HOST_DEVICE vnode(Value v, vnode* n) : val_(v), nextv_(n), ts_(TBD_v<int_t>) {}

  /// Create default vnode
  vnode() = default;
  /// Copy construct vnode
  vnode(const vnode&) = default;
  /// Move construct vnod
  vnode(vnode&&) = default;

  /// Copy vnode
  vnode& operator=(const vnode&) = default;
  /// Move vnode
  vnode& operator=(vnode&&) = default;

  /// Load timestamp
  HDS_HOST_DEVICE int_t ts() {
    return ts_.load(memory_order_relaxed); 
  }

  /// Cas timestamp
  HDS_HOST_DEVICE bool cas_ts(int_t expected, int_t desired) {
    int_t tmp = expected;
    return ts_.compare_exchange_strong(tmp, desired, memory_order_relaxed); 
  }

  /// Get next vnode
  HDS_HOST_DEVICE vnode* nextv() {
    return nextv_.load(memory_order_relaxed);
  }

  /// Get value
  HDS_HOST_DEVICE Value val() {
    return val_; 
  }

private:
  /// value
  alignas(hardware_destructive_interference_size) T val_;
  /// next
  alignas(hardware_destructive_interference_size) atomic<vnode*> nextv_;
  /// timestamp
  alignas(hardware_destructive_interference_size) atomic<int_t> ts_;
};

/// VersionedCAS object
template<typename T, typename int_t, bool increment_on_snapshot>
class vcas_object {
public:
  /// Camera
  using camera_t = camera<int_t, increment_on_snapshot>;
  /// vnode
  using vnode_t = vnode<T, int_t>;

  /// Create versioned cas object
  template<typename allocator>
  HDS_HOST_DEVICE vcas_object(T v, camera_t* S, allocator& alloc) : VHead_(allocate_and_construct_at<vnode_t>(alloc, v, nullptr)) {

    LOG("In versioned cas creation with camera %p", S);

    if constexpr (!IncrementOnSnapshot) {
      S->increment_timestamp(); 
    }

    auto head = VHead();
    
    LOG("head is %p", head);
    assert(head != nullptr);
    initTs(head, S);
  }

  template<typename Allocator>
  HDS_HOST_DEVICE void unsafe_init(Value v, Camera_t* S, Allocator& alloc) {

    LOG("In versioned cas unsafe init");
    
    VHead_.store(allocate_and_construct_at<vnode_t>(alloc, v, nullptr));

    if constexpr (!IncrementOnSnapshot) {
      S->increment_timestamp(); 
    }

    auto head = VHead();
    assert(head != nullptr);
    initTs(head, S);
  }

  /// @{ 
  /** 
   * Default constructors
   */
  VersionedCAS() = default;
  VersionedCAS(const VersionedCAS&) = default;
  VersionedCAS(VersionedCAS&& other) = default;
  /// @}
 
  /// Read version
  HDS_HOST_DEVICE Value readVersion(int_t ts, Camera_t* S) {
    auto node = VHead();
    assert(node != nullptr);
    initTs(node, S);
    while(node->ts() > ts) {
      node = node->nextv();
    }
    return node->val();
  }

  /// Read latest
  HDS_HOST_DEVICE Value vRead(Camera_t* S) {
    auto head = VHead();
    assert(head != nullptr);
    initTs(head, S);
    return head->val();
  }

  /// CAS
  template<typename Allocator>
  HDS_HOST_DEVICE bool vCAS(Value oldV, Value newV, Camera_t* S, Allocator& alloc) {
    auto head = VHead();
    assert(head != nullptr);
    initTs(head, S);
    if(head->val() != oldV) return false;
    if(newV == oldV) return true;
    vnode_t* newN = allocate_and_construct_at<vnode_t>(alloc, newV, head);
    if(casVHead(head, newN)) {
      initTs(newN, S);
      return true;
    } else {
      alloc.deallocate(newN, 1);
      auto head = VHead();
      assert(head != nullptr);
      initTs(head, S);
      return false;
    }
  }

private:

  /// Init timestamp
  HDS_HOST_DEVICE void initTs(vnode_t* n, Camera_t* S) {
    if(n->ts() == tbd) {
      LOG("Updating timestamp");
      auto curTs = S->timestamp();
      LOG("Timestamp is now %d", curTs);
      n->cas_ts(tbd, curTs); 
    }
  }

  /// Cas head
  HDS_HOST_DEVICE bool casVHead(auto expected, auto desired) {
    auto tmp = expected;
    return VHead_.compare_exchange_strong(tmp, desired, memory_order_acq_rel);
  }

  /// Load head
  HDS_HOST_DEVICE auto VHead() {
    return VHead_.load(memory_order_acquire);
  }

  /// Head vnode
  alignas(hardware_destructive_interference_size) atomic<vnode_t*> VHead_;
};

}
