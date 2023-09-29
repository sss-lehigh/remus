#pragma once
#include <functional>

#include "absl/status/statusor.h"

namespace rome {

// Implements a heap of a given size, known at compile time. This is used by the
// distribution metric since it also knows its window size at compile time and
// can therefore avoid
template <typename T, int HeapSize>
class FixedHeap {
 public:
  explicit FixedHeap(std::function<bool(T, T)> comparator)
      : size_(0), comparator_(comparator) {}

  absl::Status Push(const T& value) {
    if (size_ + 1 > HeapSize) {
      return absl::FailedPreconditionError("Cannot push any more values.");
    }
    values_[size_] = value;
    Heapify(size_);
    ++size_;
    return absl::OkStatus();
  }

  absl::StatusOr<T> GetRoot() {
    if (size_ == 0) return absl::NotFoundError("Heap is empty");
    return values_[0];
  }

  absl::StatusOr<T> Pop() {
    if (size_ == 0) return absl::NotFoundError("Heap is empty");
    auto value = values_[0];
    values_[0] = values_[size_ - 1];
    --size_;
    if (size_ != 0) {
      Pushdown(0);
    }
    return value;
  }

  inline int Size() { return size_; }

 private:
  int Parent(int index) { return (index - 1) / 2; }
  int LeftChild(int index) { return (index * 2) + 1; }
  int RightChild(int index) { return (index * 2) + 2; }

  void Heapify(int index) {
    if (size_ == 0 || index == 0) return;
    auto* curr = &values_[index];
    auto* parent = &values_[Parent(index)];
    if (comparator_(*curr, *parent)) {
      auto temp = *parent;
      *parent = *curr;
      *curr = temp;
    }
    return Heapify(Parent(index));
  }

  void Pushdown(int index) {
    if (size_ == 0 || index == size_ - 1) return;
    auto* curr = &values_[index];
    int left_child_index = LeftChild(index);
    auto* left_child =
        left_child_index < size_ ? &values_[left_child_index] : nullptr;

    int right_child_index = RightChild(index);
    auto* right_child =
        right_child_index < size_ ? &values_[right_child_index] : nullptr;

    // Base case (end of heap).
    if (left_child == nullptr && right_child == nullptr) return;

    bool swap_left = false;
    auto* swap_child = right_child;
    if (left_child != nullptr && right_child != nullptr) {
      if (comparator_(*left_child, *right_child)) {
        swap_left = true;
        swap_child = left_child;
      }
    } else if (left_child != nullptr) {
      swap_left = true;
      swap_child = left_child;
    }

    if (*curr < *swap_child) {
      auto temp = *curr;
      *curr = *swap_child;
      *swap_child = temp;
      return Pushdown(swap_left ? left_child_index : right_child_index);
    }
  }

  int size_;
  T values_[HeapSize];
  std::function<bool(T, T)> comparator_;
};

template <typename T, int HeapSize>
class FixedMaxHeap : public FixedHeap<T, HeapSize> {
 public:
  FixedMaxHeap()
      : FixedHeap<T, HeapSize>([](T v1, T v2) -> bool { return v1 > v2; }) {}
};

}  // namespace rome