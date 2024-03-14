#include <concepts>
#include <numeric>
#include <type_traits>

#include "../../utility/array.h"
#include "../../utility/atomic.h"
#include "../../utility/bitset.h"
#include "../../utility/optional.h"
#include "utility.h"

#pragma once

namespace remus::hds::kv_linked_list {

template <typename K, typename V, int N, template <typename, typename, int> typename node_pointer_, typename Allocator,
          typename Constructor = kv_inplace_construct<K, V, N, node_pointer_>>
class kv_lazy_linked_list {
private:
  using node_pointer = node_pointer_<K, V, N>;
  using node = typename node_pointer::node;

public:
  HDS_HOST_DEVICE kv_lazy_linked_list() : kv_lazy_linked_list(Allocator{}) {}

  template <typename A>
    requires std::convertible_to<A, Allocator>
  HDS_HOST_DEVICE kv_lazy_linked_list(A &&alloc_) : alloc(std::forward<A>(alloc_)) {
    root = construct(alloc.template allocate<node>(1));
  }

  template <typename A, typename C>
    requires std::convertible_to<A, Allocator> and std::convertible_to<C, Constructor>
  HDS_HOST_DEVICE kv_lazy_linked_list(A &&alloc_, C &&construct_)
    : alloc(std::forward<A>(alloc_)), construct(std::forward<C>(construct_)) {
    root = construct(alloc.template allocate<node>(1));
  }

  HDS_HOST_DEVICE ~kv_lazy_linked_list() {

    using pointer_t = decltype(alloc.template allocate<node>(1));

    node_pointer prev = root;
    node_pointer current = root.unsafe_next();

    while (current != nullptr) {
      alloc.deallocate(static_cast<pointer_t>(prev), 1);
      prev = current;
      current = prev.unsafe_next();
    }

    alloc.deallocate(static_cast<pointer_t>(prev), 1);
  }

  template <typename Group> HDS_HOST_DEVICE void print(Group &group) {

    node_pointer current = root.next(group);
    node_pointer next = nullptr;

    if (current == nullptr) {
      return;
    }

    auto current_ref = current.load(group);
    next = current_ref.next(group);

    while (true) {

      current.print();

      current = next;
      if (current == nullptr) {
        return;
      }
      current_ref = current.load(group);
      next = current_ref.next(group);
    }
  }

  template <typename Group> HDS_HOST_DEVICE bool validate(Group &group) {

    auto current = root.next(group);
    node_pointer next = nullptr;

    if (current == nullptr) {
      return true;
    }

    auto current_ref = current.load(group);
    next = current_ref.next(group);

    K prev_max = current_ref.max_key(group);

    while (true) {

      current = next;
      if (current == nullptr) {
        return true;
      }

      current_ref = current.load(group);

      K current_min = current_ref.min_key(group);
      if (current_min <= prev_max) {
        return false;
      }

      prev_max = current_ref.max_key(group);

      next = current_ref.next(group);
    }
  }

  template <typename Group> HDS_HOST_DEVICE optional<V> get(K x, Group &group) {

    while (true) {
      node_pointer current = root.next(group);
      node_pointer next = nullptr;

      if (current == nullptr) {
        return nullopt;
      }

      auto current_ref = current.load(group);
      next = current_ref.next(group);

      while (true) {

        if (current_ref.has_key(x, group) && !current.is_deleted()) {

          int idx = current_ref.index_of_key(x, group);

          optional<V> value = nullopt;

          if (group.is_leader()) {
            value = current_ref.get(idx);
          }

          return value;

        } else if (current_ref.has_any_gt_key(x, group) || next == nullptr) {
          return nullopt;
        }

        current = next;

        current_ref = current.load(group);
        next = current_ref.next(group);
      }
    }
  }

  template <typename Group> HDS_HOST_DEVICE bool insert(K k, V v, Group &group) {

    while (true) {
      auto prev(root);

      auto current = root.next(group);

      if (current == nullptr) {
        // insert new node at current

        prev.lock(group);
        current = root.next(group);
        if (current != nullptr) {
          prev.unlock(group);
          continue; // restart
        }

        node_pointer new_node = nullptr;
        if (group.is_leader()) {
          new_node = construct(alloc.template allocate<node>(1));
          node_pointer::init_locked_node(new_node);
        }

        new_node = group.shfl(new_node, group.leader_rank());

        prev.store_next(new_node);

        // node is locked and we can insert in it
        if (group.is_leader()) {
          new_node.set(0, k, v);
        }

        prev.unlock(group);
        new_node.unlock(group);

        return true;
      }

      assert(current != nullptr);

      auto current_ref = current.load(group);
      auto next = current_ref.next(group);

      while (true) {

        assert(prev != nullptr);

        int idx = current_ref.index_of_key(k, group);

        if (idx != -1) {

          prev.lock_unsync(group);
          current.lock_unsync(group);
          group.sync();

          bool valid =
            current.has_key_at(idx, k) and !current.is_deleted() and !prev.is_deleted() and prev.next(group) == current;

          group.sync();
          prev.unlock_unsync(group);
          current.unlock_unsync(group);

          if (valid) {
            return false;
          }
          break; // must retry
        } else if (current_ref.has_any_gt_key(k, group)) {
          prev.lock_unsync(group);
          current.lock_unsync(group);
          group.sync();

          // revalidate

          current_ref = current.load(group);

          bool valid = !current_ref.has_key(k, group) and current_ref.has_any_gt_key(k, group) and
                       !current.is_deleted() and !prev.is_deleted() and prev.next(group) == current;

          if (!valid) {
            group.sync();
            prev.unlock_unsync(group);
            current.unlock_unsync(group);
            break; // must retry
          }

          bool prev_is_root = prev == root;
          bool any_lt_key = current_ref.has_any_lt_key(k, group);

          if (!prev_is_root && !any_lt_key) {
            // prev is not root and current only has keys gt our key
            auto prev_ref = prev.load(group); // reload node
            // try to insert into prev
            int idx = prev_ref.empty_index(group);
            if (idx != -1) {
              if (group.is_leader()) {
                prev.set(idx, k, v);
              }
              group.sync();
              current.unlock_unsync(group);
              prev.unlock_unsync(group);
              return true;
            }
          }

          // insert into curr and maybe split
          int idx = current_ref.empty_index(group);

          if (idx != -1) {
            // insert into current
            if (group.is_leader()) {
              current.set(idx, k, v);
            }
          } else {
            node_pointer left = nullptr;
            if (group.is_leader()) {
              left = construct(alloc.template allocate<node>(1));
              node_pointer::init_locked_node(left);
              left.store_next(current);
              prev.store_next(left);
            }
            left = group.shfl(left, group.leader_rank());
            current_ref.partition_by_and_insert(k, v, left, group);

            left.unlock(group);
          }

          group.sync();
          current.unlock_unsync(group);
          prev.unlock_unsync(group);
          return true;

        } else if (next == nullptr) {

          prev.lock(group);
          current.lock(group);

          current_ref = current.load(group); // reload node

          bool valid = !current_ref.has_key(k, group) and !current.is_deleted() and !prev.is_deleted() and
                       prev.next(group) == current and current_ref.next(group) == nullptr;

          if (!valid) {
            prev.unlock(group);
            current.unlock(group);
            break;
          }

          // insert into or after current
          int idx = current_ref.empty_index(group);
          if (idx != -1) {
            // insert into current
            if (group.is_leader()) {
              current.set(idx, k, v);
            }
            prev.unlock(group);
            current.unlock(group);
            return true;
          } else {
            // insert after current

            if (group.is_leader()) {
              next = construct(alloc.template allocate<node>(1));
              node_pointer::init_locked_node(next);
              current.store_next(next);
              next.set(0, k, v);
              next.unlock_unsync_leader();
            }

            prev.unlock(group);
            current.unlock(group);
            return true;
          }
        }

        assert(next != nullptr);

        prev = current;
        current = next;

        current_ref = current.load(group);
        next = current_ref.next(group);
      }
    }
  }

  template <typename Group> HDS_HOST_DEVICE bool remove(K x, Group &group) {

    while (true) {
      node_pointer prev = root;
      node_pointer current = root.next(group);

      if (current == nullptr) {

        prev.lock(group);

        current = root.next(group); // reload
        bool still_holds = current == nullptr;

        prev.unlock(group);

        if (still_holds) {
          return false;
        } else {
          continue; // retry
        }
      }

      auto current_ref = current.load(group);

      while (true) {

        auto next = current_ref.next(group);
        int idx = current_ref.index_of_key(x, group);

        if (idx != -1) {

          prev.lock_unsync(group);
          current.lock(group);

          // revalidate
          bool valid =
            current.has_key_at(idx, x) and !current.is_deleted() and !prev.is_deleted() and prev.next(group) == current;

          if (valid) {
            next = current_ref.next(group); // reload next
            if (group.is_leader()) {
              current.remove(idx);
              if (current.is_empty()) {
                current.mark_deleted();
                prev.store_next(next);
                // TODO can reclaim current
              }
            }
          }

          current.unlock(group);
          prev.unlock_unsync(group);

          if (valid)
            return true;
          break; // retry

        } else if (current_ref.has_any_gt_key(x, group)) {

          prev.lock_unsync(group);
          current.lock(group);

          // TODO do we have to load whole node
          current_ref = current.load(group);

          // Must assert current does not contain key and has a key gt key

          bool valid = !current.is_deleted() and !prev.is_deleted() and prev.next(group) == current and
                       current_ref.has_any_gt_key(x, group) and !current_ref.has_key(x, group);

          current.unlock(group);
          prev.unlock_unsync(group);

          if (valid) {
            return false;
          }

          break;
        } else if (next == nullptr) {

          prev.lock_unsync(group);
          current.lock(group);

          bool valid = !current.is_deleted() and !prev.is_deleted() and prev.next(group) == current and
                       !current_ref.has_key(x, group) and current.next(group) == nullptr;

          // reached the end
          prev.unlock(group);
          current.unlock_unsync(group);

          if (valid) {
            return false;
          }
          break;
        }

        prev = current;
        current = next;
        current_ref = current.load(group);
      }
    }
  }

private:
  node_pointer root = nullptr;
  Allocator alloc;
  Constructor construct;
};

}; // namespace remus::hds::kv_linked_list
