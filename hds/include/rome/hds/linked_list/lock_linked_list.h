#include <type_traits>
#include <concepts>
#include <numeric>

#include "../utility/optional.h"
#include "../utility/atomic.h"
#include "../utility/array.h"
#include "../utility/bitset.h"

#pragma once

namespace rome::hds {

template<typename T, 
         int N, 
         template<typename, int> typename node_pointer_>
struct inplace_construct {

  using node_pointer = node_pointer_<T, N>;
  using node = typename node_pointer::node;
  
  constexpr HDS_HOST_DEVICE
  node_pointer operator()(node* ptr) const {
    return new (ptr) node(); 
  }

};

template<typename T,
         int N,
         template<typename, int> typename node_pointer_,
         typename Allocator,
         typename Constructor = inplace_construct<T, N, node_pointer_>>
class lock_linked_list {
private:

  using node_pointer = node_pointer_<T, N>;
  using node = typename node_pointer::node;

public:

  HDS_HOST_DEVICE lock_linked_list() : lock_linked_list(Allocator{}) {}

  template<typename A>
    requires std::convertible_to<A, Allocator>
  HDS_HOST_DEVICE lock_linked_list(A&& alloc_) : alloc(std::forward<A>(alloc_)) {
    root = construct(alloc.template allocate<node>(1));
  }

  template<typename A, typename C>
    requires std::convertible_to<A, Allocator> and std::convertible_to<C, Constructor>
  HDS_HOST_DEVICE lock_linked_list(A&& alloc_, C&& construct_) 
    : alloc(std::forward<A>(alloc_)), 
      construct(std::forward<C>(construct_)) {
    root = construct(alloc.template allocate<node>(1));
  }

  HDS_HOST_DEVICE ~lock_linked_list() {
    
    using pointer_t = decltype(alloc.template allocate<node>(1));

    node_pointer prev = root; 
    node_pointer current = root.unsafe_next();

    while(current != nullptr) {
      alloc.deallocate(static_cast<pointer_t>(prev), 1);
      prev = current;
      current = prev.unsafe_next();
    }

    alloc.deallocate(static_cast<pointer_t>(prev), 1);
  }

  template<typename Group>
  HDS_HOST_DEVICE void print(Group& group) {

    node_pointer current = root.next(group);
    node_pointer next = nullptr;

    if(current == nullptr) {
      return;
    }

    auto current_ref = current.load(group);
    next = current_ref.next(group);

    while(true) {
  
      current.print();

      current = next;
      if(current == nullptr) {
        return; 
      }
      current_ref = current.load(group);
      next = current_ref.next(group);
    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool validate(Group& group) {

    auto current = root.next(group);
    node_pointer next = nullptr;

    if(current == nullptr) {
      return true;
    }

    auto current_ref = current.load(group);
    next = current_ref.next(group);

    T prev_max = current_ref.max_key(group);

    while(true) {

      current = next;
      if(current == nullptr) {
        return true; 
      }

      current_ref = current.load(group);

      T current_min = current_ref.min_key(group);
      if(current_min <= prev_max) {
        //logging::print("Current min in ", current, " of ", current_min, " is not valid\n");
        return false;
      }

      prev_max = current_ref.max_key(group);

      next = current_ref.next(group);
    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool contains(T x, Group& group) {

    while(true) {
      root.lock(group);
      node_pointer current = root.next(group);
      node_pointer next = nullptr;

      if(current == nullptr) {
        root.unlock(group);
        return false;
      }

      current.lock_unsync(group);
      root.unlock_unsync(group);
      group.sync();

      auto current_ref = current.load(group);
      next = current_ref.next(group);

      while(true) {

        if(current_ref.has_key(x, group)) {
          current.unlock_unsync(group);
          return true;
        } else if(current_ref.has_any_gt_key(x, group) || next == nullptr) {
          current.unlock_unsync(group);
          return false;
        }

        next.lock_unsync(group);
        current.unlock_unsync(group);
        group.sync();

        current = next;

        current_ref = current.load(group);
        next = current_ref.next(group);
      }
    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool insert(T x, Group& group) {

    while(true) {
      auto prev(root);
      prev.lock(group);

      auto current = root.next(group);

      if (current == nullptr) {
        // insert new node at current
        
        node_pointer new_node = nullptr;
        if (group.is_leader()) {
          new_node = construct(alloc.template allocate<node>(1));
          node_pointer::init_locked_node(new_node);
        }
        
        new_node = group.shfl(new_node, group.leader_rank());

        prev.store_next(new_node);

        // node is locked and we can insert in it
        if(group.is_leader()) {
          new_node.set(0, x);
        }

        prev.unlock(group);
        new_node.unlock(group);

        return true;
      }

      assert(current != nullptr);

      current.lock(group);

      auto current_ref = current.load(group);
      auto next = current_ref.next(group);

      while(true) {

        // prev and current are locked

        assert(prev != nullptr);

        //logging::print("Current:\n");

        //logging::print("Anything gt key ", x ," ? ",static_cast<int>(current->has_any_gt_key(x, current_res, group)), "\n");

        if (current_ref.has_key(x, group)) {
          group.sync();
          current.unlock_unsync(group);
          prev.unlock_unsync(group);
          return false;
        } else if (current_ref.has_any_gt_key(x, group)) {

          //logging::print("Current has something gt key\n");

          bool prev_is_root = prev == root;
          bool any_lt_key = current_ref.has_any_lt_key(x, group);

          if(!prev_is_root && !any_lt_key) {
            //logging::print("Prev is not root and nothing lt key so try prev\n");

            // prev is not root and current only has keys gt our key
            auto prev_ref = prev.load(group); // reload node
            // try to insert into prev
            int idx = prev_ref.empty_index(group);
            if(idx != -1) {
              if(group.is_leader()) {
                prev.set(idx, x);
              }
              group.sync();
              current.unlock_unsync(group);
              prev.unlock_unsync(group);
              return true;
            }
          }

          //logging::print("Insert into current and maybe split\n");
          // insert into curr and maybe split
          int idx = current_ref.empty_index(group);
          
          if(idx != -1) {
            //logging::print("Inserting in current ", static_cast<node_pointer>(current), "\n");
            // insert into current
            if(group.is_leader()) {
              current.set(idx, x);
            }
          } else {

            //logging::print("Splitting ", static_cast<node_pointer>(current), "\n");

            node_pointer left = nullptr;
            if(group.is_leader()) {
              left = construct(alloc.template allocate<node>(1));
              node_pointer::init_locked_node(left);
              left.store_next(current);
              prev.store_next(left);
            }
            left = group.shfl(left, group.leader_rank());
            current_ref.partition_by_and_insert(x, left, group);
            
            left.unlock(group);
          }

          group.sync();
          current.unlock_unsync(group);
          prev.unlock_unsync(group);
          return true;
          
        } else if (next == nullptr) {
          //logging::print("Next is nullptr\n");
          // insert into or after current
          int idx = current_ref.empty_index(group);
          if(idx != -1) {
            //logging::print("Inserting in current ", static_cast<node_pointer>(current), "\n");
            // insert into current
            if(group.is_leader()) {
              current.set(idx, x);
            }
            group.sync();
            current.unlock_unsync(group);
            prev.unlock_unsync(group);
            return true;
          } else {
            //logging::print("Have to insert after current\n");
            // insert after current

            if(group.is_leader()) {
              next = construct(alloc.template allocate<node>(1));
              node_pointer::init_locked_node(next);
              current.store_next(next);
              next.set(0, x);
              next.unlock_unsync_leader();
            }
            
            group.sync();
            current.unlock_unsync(group);
            prev.unlock_unsync(group);
            return true;
          }
        }

        assert(next != nullptr);

        next.lock(group);
        prev.unlock(group);
        prev = current;
        current = next;
        
        current_ref = current.load(group);
        next = current_ref.next(group);

      }

    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool remove(T x, Group& group) {

    while(true) {
      node_pointer prev = root;
      root.lock(group);
      node_pointer current = root.next(group);

      if(current == nullptr) {
        root.unlock(group);
        return false;
      }

      current.lock(group);

      auto current_ref = current.load(group);

      while(true) {

        auto next = current_ref.next(group);
        int idx = current_ref.index_of_key(x, group);

        if(idx != -1) {

          bool reclaimed = false;
          if (group.is_leader()) {
            current.remove(idx);
            if (current.is_empty()) {
              prev.store_next(next);
              reclaimed = true;
            }
          }
          current.unlock(group);
          prev.unlock(group); 

          if(reclaimed) {
            using pointer_t = decltype(alloc.template allocate<node>(1));
            alloc.deallocate(static_cast<pointer_t>(current), 1);
          }

          return true;
        } else if(current_ref.has_any_gt_key(x, group)) {
          current.unlock(group);
          prev.unlock(group); 
          return false;
        }

        if(next == nullptr) {
          // reached the end
          prev.unlock(group);
          current.unlock(group);
          return false;
        }

        next.lock(group);
        prev.unlock(group);
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

};

