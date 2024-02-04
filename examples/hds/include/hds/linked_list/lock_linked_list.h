#include <type_traits>
#include <concepts>
#include <numeric>

#include "../logging/print.h"
#include "../utility/optional.h"
#include "../utility/atomic.h"
#include "../utility/array.h"
#include "../utility/bitset.h"

#pragma once

namespace hds {

template<typename T, int N, typename Allocator>
class linked_list {
public:

  HDS_HOST_DEVICE linked_list() : root(new (Allocator{}.template allocate<node>(1)) node()) {}

  template<typename A>
    requires std::convertible_to<A, Allocator>
  HDS_HOST_DEVICE linked_list(A&& alloc_) : alloc(std::forward<A>(alloc_)) {
    root = new (alloc.template allocate<node>(1)) node();
  }

  HDS_HOST_DEVICE ~linked_list() {
    node* prev = root; 
    node* current = root->unsafe_next();

    while(current != nullptr) {
      alloc.deallocate(prev, 1);
      prev = current;
      current = prev->unsafe_next();
    }

    alloc.deallocate(prev, 1);
  }

  template<typename Group>
  HDS_HOST_DEVICE void print(Group&& group) {

    node* current = root->next(std::forward<Group>(group));
    node* next = nullptr;

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
  HDS_HOST_DEVICE bool validate(Group&& group) {

    auto current = root->next(std::forward<Group>(group));
    node* next = nullptr;

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
        //logging::print("Current min in ", current, " of ", current_min, " is not valid\n");
        return false;
      }

      prev_max = current->max_key(res, std::forward<Group>(group));

      next = current->next(res, std::forward<Group>(group));
    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool contains(T x, Group&& group) {

    while(true) {
      root->lock(std::forward<Group>(group));
      node* current = root->next(std::forward<Group>(group));
      node* next = nullptr;

      if(static_cast<node*>(current) == nullptr) {
        root->unlock(std::forward<Group>(group));
        return false;
      }

      current->lock_unsync(std::forward<Group>(group));
      root->unlock_unsync(std::forward<Group>(group));
      group.sync();

      auto res = current->load(std::forward<Group>(group));
      next = current->next(res, std::forward<Group>(group));

      

      while(true) {

        if(current->has_key(x, res, std::forward<Group>(group))) {
          current->unlock_unsync(std::forward<Group>(group));
          return true;
        } else if(current->has_any_gt_key(x, res, std::forward<Group>(group)) || next == nullptr) {
          current->unlock_unsync(std::forward<Group>(group));
          return false;
        }

        next->lock_unsync(std::forward<Group>(group));
        current->unlock_unsync(std::forward<Group>(group));
        group.sync();

        current = next;

        res = current->load(std::forward<Group>(group));
        next = current->next(res, std::forward<Group>(group));
      }
    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool insert(T x, Group&& group) {

    while(true) {
      auto prev(root);
      prev->lock(std::forward<Group>(group));

      auto current = root->next(std::forward<Group>(group));

      if (current == nullptr) {
        // insert new node at current
        
        node* new_node = nullptr;
        if (std::forward<Group>(group).is_leader()) {
          new_node = alloc.template allocate<node>(1);
          node::init_locked_node(new_node);
        }
        
        new_node = group.shfl(new_node, group.leader_rank());

        prev->store_next(new_node);

        // node is locked and we can insert in it
        if(group.is_leader()) {
          new_node->set(0, x);
        }

        prev->unlock(std::forward<Group>(group));
        new_node->unlock(std::forward<Group>(group));

        return true;
      }

      assert(current != nullptr);

      current->lock(std::forward<Group>(group));

      auto current_res = current->load(std::forward<Group>(group));
      auto next = current->next(current_res, std::forward<Group>(group));

      while(true) {

        // prev and current are locked

        assert(prev != nullptr);

        //logging::print("Current:\n");

        //logging::print("Anything gt key ", x ," ? ",static_cast<int>(current->has_any_gt_key(x, current_res, std::forward<Group>(group))), "\n");

        if (current->has_key(x, current_res, std::forward<Group>(group))) {
          group.sync();
          current->unlock_unsync(std::forward<Group>(group));
          prev->unlock_unsync(std::forward<Group>(group));
          return false;
        } else if (current->has_any_gt_key(x, current_res, std::forward<Group>(group))) {

          //logging::print("Current has something gt key\n");

          bool prev_is_root = prev == root;
          bool any_lt_key = current->has_any_lt_key(x, current_res, std::forward<Group>(group));

          if(!prev_is_root && !any_lt_key) {
            //logging::print("Prev is not root and nothing lt key so try prev\n");

            // prev is not root and current only has keys gt our key
            auto prev_res = prev->load(std::forward<Group>(group)); // reload node
            // try to insert into prev
            int idx = prev->empty_index(prev_res, std::forward<Group>(group));
            if(idx != -1) {
              if(group.is_leader()) {
                prev->set(idx, x);
              }
              group.sync();
              current->unlock_unsync(std::forward<Group>(group));
              prev->unlock_unsync(std::forward<Group>(group));
              return true;
            }
          }

          //logging::print("Insert into current and maybe split\n");
          // insert into curr and maybe split
          int idx = current->empty_index(current_res, std::forward<Group>(group));
          
          if(idx != -1) {
            //logging::print("Inserting in current ", static_cast<node*>(current), "\n");
            // insert into current
            if(group.is_leader()) {
              current->set(idx, x);
            }
          } else {

            //logging::print("Splitting ", static_cast<node*>(current), "\n");

            node* left = nullptr;
            if(group.is_leader()) {
              left = alloc.template allocate<node>(1);
              node::init_locked_node(left);
              left->store_next(current);
              prev->store_next(left);
            }
            left = group.shfl(left, group.leader_rank());
            current->partition_by_and_insert(x, left, current_res, std::forward<Group>(group));
            
            left->unlock(group);
          }

          group.sync();
          current->unlock_unsync(std::forward<Group>(group));
          prev->unlock_unsync(std::forward<Group>(group));
          return true;
          
        } else if (next == nullptr) {
          //logging::print("Next is nullptr\n");
          // insert into or after current
          int idx = current->empty_index(current_res, std::forward<Group>(group));
          if(idx != -1) {
            //logging::print("Inserting in current ", static_cast<node*>(current), "\n");
            // insert into current
            if(group.is_leader()) {
              current->set(idx, x);
            }
            group.sync();
            current->unlock_unsync(std::forward<Group>(group));
            prev->unlock_unsync(std::forward<Group>(group));
            return true;
          } else {
            //logging::print("Have to insert after current\n");
            // insert after current

            if(group.is_leader()) {
              next = alloc.template allocate<node>(1);
              node::init_locked_node(next);
              current->store_next(next);
              next->set(0, x);
              next->unlock_unsync_leader();
            }
            
            group.sync();
            current->unlock_unsync(std::forward<Group>(group));
            prev->unlock_unsync(std::forward<Group>(group));
            return true;
          }
        }

        assert(next != nullptr);

        next->lock(std::forward<Group>(group));
        prev->unlock(std::forward<Group>(group));
        prev = current;
        current = next;
        
        current_res = current->load(std::forward<Group>(group));
        next = current->next(current_res, std::forward<Group>(group));

      }

    }
  }

  template<typename Group>
  HDS_HOST_DEVICE bool remove(T x, Group&& group) {

    while(true) {
      node* prev = root;
      root->lock(std::forward<Group>(group));
      node* current = root->next(std::forward<Group>(group));

      if(static_cast<node*>(current) == nullptr) {
        root->unlock(std::forward<Group>(group));
        return false;
      }

      current->lock(std::forward<Group>(group));

      auto res = current->load(std::forward<Group>(group));

      while(true) {

        auto next = current->next(res, std::forward<Group>(group));
        int idx = current->index_of_key(x, res, std::forward<Group>(group));

        if(idx != -1) {

          bool reclaimed = false;
          if (std::forward<Group>(group).is_leader()) {
            current->remove(idx);
            if (current->is_empty()) {
              prev->store_next(next);
              reclaimed = true;
            }
          }
          current->unlock(std::forward<Group>(group));
          prev->unlock(std::forward<Group>(group)); 

          if(reclaimed) {
            alloc.deallocate(current, 1);
          }

          return true;
        } else if(current->has_any_gt_key(x, res, std::forward<Group>(group))) {
          current->unlock(std::forward<Group>(group));
          prev->unlock(std::forward<Group>(group)); 
          return false;
        }

        if(next == nullptr) {
          // reached the end
          prev->unlock(std::forward<Group>(group));
          current->unlock(std::forward<Group>(group));
          return false;
        }

        next->lock(std::forward<Group>(group));
        prev->unlock(std::forward<Group>(group));
        prev = current;
        current = next;
        res = current->load(std::forward<Group>(group));
      }
    }
  }

private:

  template<size_t size>
  struct TemporaryData {
    utility::uint32_bitset<N / size> present;
    array<T, N / size> regs;
  };

  template<size_t size>
  struct Tiling {
    HDS_HOST_DEVICE constexpr int operator()(int idx, int thread) {
      return thread + idx * size;
    }
  };

  struct node {

    HDS_HOST_DEVICE static node* init_locked_node(node* n) {
      atomic_ref(n->lock_).store(1, memory_order_relaxed);
      atomic_ref(n->present).store(0, memory_order_relaxed);
      atomic_ref(n->next_).store(nullptr, memory_order_relaxed);
      return n;
    }

    template<typename Group>
    HDS_HOST_DEVICE TemporaryData<std::remove_cvref_t<Group>::size> load(Group&& group) {
      static_assert(N % std::remove_cvref_t<Group>::size == 0);
      static_assert(std::remove_cvref_t<Group>::size <= N);

      Tiling<std::remove_cvref_t<Group>::size> tile{};
      TemporaryData<std::remove_cvref_t<Group>::size> result;

      auto tmp = atomic_ref(present).load(memory_order_relaxed);
      
      if constexpr (tile(N / std::remove_cvref_t<Group>::size - 1, std::remove_cvref_t<Group>::size - 1) < N) {
        for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
          result.regs[i] = atomic_ref(values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
        }
        
        for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
          result.present.set(i, tmp.get(tile(i, group.thread_rank())));
        }

      } else {

        for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
          if(tiling(i, group.thread_rank()) < N) {
            result.regs[i] = atomic_ref(values[tile(i, group.thread_rank())]).load(memory_order_relaxed);
          }
        }
        
        for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
          if(tiling(i, group.thread_rank()) < N) {
            result.present.set(i, tmp.get(tile(i, group.thread_rank())));
          }
        }
      }

      return result;
    }

    template<typename Group>
    HDS_HOST_DEVICE int index_of_key(T x, TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] == x) {
          found = true;
          break;
        }
      }

      int index = group.ballot_index(found);

      if(index != -1) {
        Tiling<std::remove_cvref_t<Group>::size> tile{};
        int offset = tile(index, group.thread_rank());
        return group.shfl(offset, index);
      } else {
        return -1;
      }
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_key(T x, const TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] == x) {
          found = true;
          break;
        }
      }
      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE int empty_index(TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      int idx = 0;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(!result.present.get(i)) {
          found = true;
          idx = i;
          break;
        }
      }

      int index = group.ballot_index(found);

      if(index != -1) {
        Tiling<std::remove_cvref_t<Group>::size> tile{};
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

    HDS_HOST_DEVICE void remove(int offset) {
      auto tmp = atomic_ref(present).load(memory_order_relaxed);
      tmp.set(offset, false);
      atomic_ref(present).store(tmp, memory_order_relaxed);
    }

    HDS_HOST_DEVICE bool is_empty() {
      auto tmp = atomic_ref(present).load(memory_order_relaxed);

      for (int i = 0; i < N; ++i) {
        if (tmp.get(i)) {
          return false;
        }
      }

      return true;
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_lt_key(T x, TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] < x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_gt_key(T x, TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] > x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }

    template<typename Group>
    HDS_HOST_DEVICE bool has_any_gte_key(T x, TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      bool found = false;
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] >= x) {
          found = true;
          break;
        }
      }

      return group.any(found);
    }


    template<typename Group>
    HDS_HOST_DEVICE T max_key(TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      T max_key = std::numeric_limits<T>::min();
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] >= max_key) {
          max_key = result.regs[i];
        }
      }

      return group.reduce_max(max_key);
    }


    template<typename Group>
    HDS_HOST_DEVICE T min_key(TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      T min_key = std::numeric_limits<T>::max();
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] <= min_key) {
          min_key = result.regs[i];
        }
      }

      return group.reduce_min(min_key);
    }

    template<typename Group>
    HDS_HOST_DEVICE void partition_by_and_insert(T x, node* left, TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {

      Tiling<std::remove_cvref_t<Group>::size> tile{};

      utility::uint32_bitset<N> my_bits = 0;

      // Store all keys in left
      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        int offset = tile(i, group.thread_rank());
        atomic_ref<T>(left->values[offset]).store(result.regs[i], memory_order_relaxed);
      }

      for(int i = 0; i < N / std::remove_cvref_t<Group>::size; ++i) {
        if(result.present.get(i) && result.regs[i] < x) {
          int offset = tile(i, group.thread_rank());
          // tell leader to store present bit set at i
          my_bits.set(offset);
        }
      }

      auto reduction = group.reduce_or(static_cast<uint32_t>(my_bits));

      // reduce or the bits
      auto bits = utility::uint32_bitset<N>(reduction);

      if(group.is_leader()) {

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
    HDS_HOST_DEVICE node* next(TemporaryData<std::remove_cvref_t<Group>::size>& result, Group&& group) {
      return atomic_ref(next_).load(memory_order_relaxed);
    }

    template<typename Group>
    HDS_HOST_DEVICE node* next(Group&& group) {
      return atomic_ref(next_).load(memory_order_relaxed);
    }

    HDS_HOST_DEVICE node* unsafe_next() {
      return atomic_ref(next_).load(memory_order_relaxed);
    }

    HDS_HOST_DEVICE void store_next(node* val) {
      atomic_ref(next_).store(val, memory_order_relaxed);
    }

    template<typename Group>
    HDS_HOST_DEVICE void lock_unsync(Group&& group) {
      if(group.is_leader()) {
        lock_unsync_leader(); 
      }
      hds::atomic_thread_fence(memory_order_acquire);
    }

    HDS_HOST_DEVICE void lock_unsync_leader() {
      while(true) {
        uint64_t expected = 0;
        if (atomic_ref(lock_).compare_exchange_strong(expected, 1, memory_order_acq_rel)) {
          return;
        }
        while(!atomic_ref(lock_).load(memory_order_relaxed) == 0) {
          #if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__)
          __builtin_ia32_pause();
          #endif
        }
      }
    }

    template<typename Group>
    HDS_HOST_DEVICE void lock(Group&& group) {
      lock_unsync(std::forward<Group>(group));
      group.sync();
    }

    template<typename Group>
    HDS_HOST_DEVICE void unlock_unsync(Group&& group) {
      hds::atomic_thread_fence(memory_order_release);
      if(group.is_leader()) {
        unlock_unsync_leader();
      }
    }
 
    HDS_HOST_DEVICE void unlock_unsync_leader() {
      atomic_ref(lock_).store(0, memory_order_release);
    }   

    template<typename Group>
    HDS_HOST_DEVICE void unlock(Group&& group) {
      group.sync();
      unlock_unsync(std::forward<Group>(group));
    }

    HDS_HOST_DEVICE void print() {
      printf("Node: %p\n", this);

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
      printf("Next : %p\n", next_);
      printf("\n");
    }

    private:
      alignas(128) uint64_t lock_ = 0;
      alignas(128) utility::uint32_bitset<N> present = 0;
      alignas(128) T values[N];
      node* next_ = nullptr;
  };

  node* root = nullptr;
  Allocator alloc;
};

};

