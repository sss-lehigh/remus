#include "../utility/optional.h"
#include "../utility/atomic.h"
#include "../utility/array.h"
#include "../utility/marked_ptr.h"

#pragma once

namespace hds {

template<typename T, int N>
class linked_list {
public:

  linked_list() : root(new node()) {}


  template<typename Group>
  void print(Group&& group) {

    utility::marked_ptr<node, 1> current = root->next(std::forward<Group>(group));
    utility::marked_ptr<node, 1> next = nullptr;

    if(static_cast<node*>(current) == nullptr) {
      return;
    }

    auto res = current->load(std::forward<Group>(group));
    next = current->next(res, std::forward<Group>(group));

    while(true) {
  
      current->print();

      current = next;
      if(current == nullptr) {
        return; 
      }
      res = current->load(std::forward<Group>(group));
      next = current->next(res, std::forward<Group>(group));
    }
  }

  template<typename Group>
  bool validate(Group&& group) {

    utility::marked_ptr<node, 1> current = root->next(std::forward<Group>(group));
    utility::marked_ptr<node, 1> next = nullptr;

    if(static_cast<node*>(current) == nullptr) {
      return true;
    }

    auto res = current->load(std::forward<Group>(group));
    next = current->next(res, std::forward<Group>(group));

    T prev_max = current->max_key(res, std::forward<Group>(group));

    while(true) {

      current = next;
      if(current == nullptr) {
        return true; 
      }

      res = current->load(std::forward<Group>(group));

      T current_min = current->min_key(res, std::forward<Group>(group));
      if(current_min <= prev_max) {
        printf("Current min in %p of %d is not valid\n", static_cast<node*>(current), current_min);
        return false;
      }

      prev_max = current->max_key(res, std::forward<Group>(group));

      next = current->next(res, std::forward<Group>(group));
    }
  }

  template<typename Group>
  bool contains(T x, Group&& group) {

    printf("Begining contains for %d\n", x);

    while(true) {
      utility::marked_ptr<node, 1> prev(root);
      utility::marked_ptr<node, 1> current = prev->next(std::forward<Group>(group));
      utility::marked_ptr<node, 1> next = nullptr;

      if(static_cast<node*>(current) == nullptr) {
        printf("Current is nullptr\n");
        return false;
      }

      auto res = current->load(std::forward<Group>(group));
      next = current->next(res, std::forward<Group>(group));

      while(true) {

        if(current->has_key(x, res, std::forward<Group>(group))) {
          printf("Current has key %d\n", x);
          if(next.is_marked(0)) {
            printf("Current is deleted\n");
            return false;
          }
          return true;
        } else if(current->has_any_gt_key(x, res, std::forward<Group>(group))) {
          printf("Current a key gt key and does not have key\n");
          current->print();
          return false;
        }
        
        prev = current;
        current = next;
        if(current == nullptr) {
          printf("Current is null\n");
          return false;
        }
        res = current->load(std::forward<Group>(group));
        next = current->next(res, std::forward<Group>(group));
      }
    }
  }

  template<typename Group>
  bool insert(T x, Group&& group) {
    printf("Inserting %d\n", x);
    static_assert(Group::size <= N);

    while(true) {
      utility::marked_ptr<node, 1> prev(root);
      utility::marked_ptr<node, 1> current = root->next(group);
      utility::marked_ptr<node, 1> next = nullptr;

      if (static_cast<node*>(current) == nullptr) {
        printf("Inserting at head\n");
        prev->lock_unsync(group);
        group.sync();

        current = prev->next(group);
        if(static_cast<node*>(current) != nullptr) {
          prev->unlock_unsync(group);
          continue;
        }

        // insert new node at current
        
        node* new_node = nullptr;
        if (group.is_leader()) {
          new_node = new node();
          node::init_locked_node(new_node);
        }
        
        new_node = group.shfl(new_node, group.leader_rank());

        prev->store_next(utility::marked_ptr<node, 1>(new_node));
        // node is locked and we can insert in it

        if(group.is_leader()) {
          new_node->set(0, x);
        }

        group.sync();
        prev->unlock_unsync(group);
        new_node->unlock_unsync(group);

        return true;
      }

      assert(static_cast<node*>(current) != nullptr);
      auto current_res = current->load(std::forward<Group>(group));
      next = current->next(current_res, std::forward<Group>(group));

      while(true) {
  
        assert(static_cast<node*>(prev) != nullptr);

        if(current->has_any_gte_key(x, current_res, std::forward<Group>(group)) || static_cast<node*>(next) == nullptr) {

          prev->lock_unsync(group);
          current->lock_unsync(group);
          group.sync();


          current_res = current->load(std::forward<Group>(group)); // reload node

          auto prev_next = prev->next(std::forward<Group>(group)); // just load next
          next = current->next(current_res, std::forward<Group>(group));

          bool valid = !prev_next.is_marked(0) && 
                       !next.is_marked(0) && 
                       static_cast<node*>(prev_next) == static_cast<node*>(current);

          bool inserted = false;

          if(valid) {

            inserted = !current->has_key(x, current_res, std::forward<Group>(group));

            // can perform operation
            if(inserted) {

              bool prev_is_root = static_cast<node*>(prev) != root;
              bool any_lt_key = current->has_any_lt_key(x, current_res, std::forward<Group>(group));

              int idx = -1;

              // dont have key
              if(!prev_is_root && !any_lt_key) {
                auto prev_res = prev->load(std::forward<Group>(group)); // reload node
                // try to insert into prev
                prev->empty_index(prev_res, std::forward<Group>(group));
              }

              if(idx != -1) {
                printf("Inserting in prev %p\n", static_cast<node*>(prev));
                if(group.is_leader()) {
                  prev->set(idx, x);
                }
              } else {
                // insert into curr and maybe split
                idx = current->empty_index(current_res, std::forward<Group>(group));
               
                if(idx != -1) {
                  printf("Inserting in current %p\n", static_cast<node*>(current));
                  // insert into current
                  if(group.is_leader()) {
                    current->set(idx, x);
                  }
                } else {

                  printf("Splitting %p\n", static_cast<node*>(current));

                  // TODO check if all are lt to key and then insert after current
                  // TODO check if all are gt key and then insert before current

                  node* left = nullptr;
                  if(group.is_leader()) {
                    left = new node();
                    node::init_locked_node(left);
                    left->store_next(current);
                    prev->store_next(utility::marked_ptr<node, 1>(left));
                  }
                  left = group.shfl(left, group.leader_rank());
                  current->partition_by_and_insert(x, left, current_res, std::forward<Group>(group));
                  
                  group.sync();
                  left->unlock_unsync(group);
                }
              }
            }
          }

          group.sync();
          current->unlock_unsync(group);
          prev->unlock_unsync(group);

          if(valid) {
            return inserted;
          }
          break;
        } 

        prev = current;
        current = next;
        
        auto res = current->load(std::forward<Group>(group));
        next = current->next(res, std::forward<Group>(group));

      }

    }
  }

private:

  struct node {

    template<typename Group>
    struct TemporaryData {
      utility::uint32_bitset<N / Group::size> present;
      array<T, N / Group::size> regs;
    };

    template<typename Group>
    struct Tiling {
      HDS_HOST_DEVICE constexpr int operator()(int idx, int thread) {
        return thread + idx * Group::size;
      }
    };

    static node* init_locked_node(node* n) {
      atomic_ref(n->lock_).store(1, memory_order_relaxed);
      atomic_ref(n->present).store(0, memory_order_relaxed);
      atomic_ref(n->next_).store(nullptr, memory_order_relaxed);
      return n;
    }

    template<typename Group>
    HDS_HOST_DEVICE TemporaryData<Group> load(Group&& group) {
      static_assert(N % Group::size == 0);
      static_assert(Group::size <= N);

      Tiling<Group> tile{};
      TemporaryData<Group> result;

      auto tmp = atomic_ref(present).load(memory_order_relaxed);
      
      if constexpr (tile(N / Group::size - 1, Group::size - 1) < N) {
        for(int i = 0; i < N / Group::size; ++i) {
          result.regs[i] = atomic_ref(values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
        }
        
        for(int i = 0; i < N / Group::size; ++i) {
          result.present.set(i, tmp.get(tile(i, group.thread_rank())));
        }
      } else {

        for(int i = 0; i < N / Group::size; ++i) {
          if(tiling(i, group.thread_rank()) < N) {
            result.regs[i] = atomic_ref(values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
          }
        }
        
        for(int i = 0; i < N / Group::size; ++i) {
          if(tiling(i, group.thread_rank()) < N) {
            result.present.set(i, tmp.get(tile(i, group.thread_rank())));
          }
        }
      }

      return result;
    }

    template<typename Group>
    HDS_HOST_DEVICE int index_of_key(T x, TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] == x) {
          found = true;
          break;
        }
      }

      int index = group.ballot_index(found);

      if(index != -1) {
        Tiling<Group> tile{};
        int offset = tile(index, group.thread_rank());
        return group.shfl(offset, index);
      } else {
        return -1;
      }
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_key(T x, const TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] == x) {
          printf("Found key %d\n", x);
          found = true;
          break;
        }
      }
      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE int empty_index(TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      int idx = 0;
      for(int i = 0; i < N / Group::size; ++i) {
        if(!result.present.get(i)) {
          found = true;
          idx = i;
          break;
        }
      }

      int index = group.ballot_index(found);

      if(index != -1) {
        Tiling<Group> tile{};
        int offset = tile(idx, group.thread_rank());
        return group.shfl(offset, index);
      }

      return -1;
    }

    HDS_HOST_DEVICE void set(int offset, T x) {
      atomic_ref(values[offset]).store(x, memory_order_relaxed);

      auto tmp = atomic_ref(present).load(memory_order_relaxed);
      tmp.set(offset, true);

      atomic_ref(present).store(tmp, memory_order_relaxed);
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_lt_key(T x, TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] < x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_gt_key(T x, TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] > x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_gte_key(T x, TemporaryData<Group>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] >= x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }


    template<typename Group>
    HDS_HOST_DEVICE T max_key(TemporaryData<Group>& result, Group&& group) {
      T max_key = std::numeric_limits<T>::min();
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] >= max_key) {
          max_key = result.regs[i];
        }
      }

      return group.reduce_max(max_key);
    }


    template<typename Group>
    HDS_HOST_DEVICE T min_key(TemporaryData<Group>& result, Group&& group) {
      T min_key = std::numeric_limits<T>::max();
      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] <= min_key) {
          min_key = result.regs[i];
        }
      }

      return group.reduce_min(min_key);
    }

    template<typename Group>
    HDS_HOST_DEVICE void partition_by_and_insert(T x, node* left, TemporaryData<Group>& result, Group&& group) {

      Tiling<Group> tile{};

      utility::uint32_bitset<N> my_bits = 0;

      // Store all keys in left
      for(int i = 0; i < N / Group::size; ++i) {
        int offset = tile(i, group.thread_rank());
        atomic_ref<T>(left->values[offset]).store(result.regs[i], memory_order_relaxed);
      }

      for(int i = 0; i < N / Group::size; ++i) {
        if(result.present.get(i) && result.regs[i] < x) {
          int offset = tile(i, group.thread_rank());
          // tell leader to store present bit set at i
          my_bits.set(offset);
        }
      }

      printf("My bits is 0x%x this will reduce to 0x%x\n", static_cast<uint32_t>(my_bits), group.reduce_or(static_cast<uint32_t>(my_bits)));

      auto reduction = group.reduce_or(static_cast<uint32_t>(my_bits));

      // reduce or the bits
      auto bits = utility::uint32_bitset<N>(reduction);

      if(group.is_leader()) {

        printf("Left bits are 0x%x\n", static_cast<uint32_t>(bits));

        atomic_ref(left->present).store(bits, memory_order_relaxed);
        uint32_t xored_bits = static_cast<uint32_t>(bits) ^ static_cast<uint32_t>(atomic_ref(present).load(memory_order_relaxed));
        auto new_present = utility::uint32_bitset<N>(xored_bits);
        atomic_ref(present).store(new_present, memory_order_seq_cst);

        for(int i = 0; i < N; ++i) {
          if(!bits.get(i)) {
            left->set(i, x);
            break;
          }
          if(!new_present.get(i)) {
            set(i, x);
            break;
          }
        }
      }

    }


    template<typename Group>
    HDS_HOST_DEVICE utility::marked_ptr<node, 1> next(TemporaryData<Group>& result, Group&& group) {
      return atomic_ref(next_).load(memory_order_relaxed);
    }

    template<typename Group>
    HDS_HOST_DEVICE utility::marked_ptr<node, 1> next(Group&& group) {
      return atomic_ref(next_).load(memory_order_relaxed);
    }

    HDS_HOST_DEVICE void store_next(utility::marked_ptr<node, 1> val) {
      atomic_ref(next_).store(val, memory_order_relaxed);
    }

    template<typename Group>
    HDS_HOST_DEVICE void lock_unsync(Group&& group) {

      printf("Trying to lock %p\n", this);

      if(group.is_leader()) {
        while(true) {
          auto expected = atomic_ref(lock_).load(memory_order_relaxed);
          if(expected == 0) {
            if(atomic_ref(lock_).compare_exchange_strong(expected, 1, memory_order_relaxed)) {
              return;
            }
          }
        }
      }
      hds::atomic_thread_fence(memory_order_seq_cst);

    }

    template<typename Group>
    HDS_HOST_DEVICE void unlock_unsync(Group&& group) {

      printf("Unlocking %p\n", this);
      hds::atomic_thread_fence(memory_order_seq_cst);
      if(group.is_leader()) {
        atomic_ref(lock_).store(0, memory_order_relaxed);
      }
    }

    HDS_HOST_DEVICE void print() {
      printf("Node: %p\n", this);
      if(next_.is_marked(0)) {
        printf("Is deleted\n");
      }

      bool have_min = false;
      T min_key;

      bool have_max = false;
      T max_key;
      printf("Present : %x\n", static_cast<uint32_t>(present));
      for(int i = 0; i < N; ++i) {
        if(present.get(i)) {
          printf("Key : %d\n", values[i]);
          if(!have_min || values[i] < min_key) {
            min_key = values[i];
            have_min = true;
          }
          if(!have_max || values[i] > max_key) {
            max_key = values[i];
            have_max = true;
          }
        }
      }

      printf("Min : %d | Max : %d\n", min_key, max_key);
      printf("Next : %p\n", static_cast<node*>(next_));
      printf("\n");
    }

    private:
      alignas(128) uint64_t lock_ = 0;
      alignas(128) utility::uint32_bitset<N> present = 0;
      alignas(128) T values[N];
      utility::marked_ptr<node, 1> next_ = nullptr;
  };

  static_assert(utility::marked_ptr<node, 1>::is_valid());

  node* root = nullptr;

};

};

