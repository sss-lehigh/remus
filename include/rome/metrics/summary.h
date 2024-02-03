#pragma once

#include <array>
#include <cmath>
#include <protos/metrics.pb.h>
#include <sstream>
#include <type_traits>

#include "../vendor/sss/status.h"
#include "atree/atree.h"
#include "metric.h"

namespace rome::metrics {

// A `Summary` maintains summary statistics about a given stream of values. The
// computed statistics include the min and max, the 50th, 90th, 95th, 99th,
// and 99.9th percentiles, and the mean and standard deviation.
//
// Internally,the values are maintained by respective member variables tracking
// the average (of the statistic) over time. As new samples arrive, they are
// placed in a binary search tree (i.e, `rome::ATree`) that augments each node
// with the total number of nodes in the subtree rooted at the node. This is
// then used to query the tree for the node with a given rank, which is computed
// from the given percentile and the total number of samples in the tree. To
// save space, whenever a sample matches an existing node, we update the nodes
// value to reflect the number of matching samples. The above is peformed for
// some given window of samples, at which point the summary statistics are
// updated and the sample tree is cleared for the next window. Each of the
// quantile statistics are updated by computing the weighted
//
// The approach is naive and there are other ways to accomplish the same goal.
// For example,
// http://infolab.stanford.edu/~datar/courses/cs361a/papers/quantiles.pdf gives
// an algorithm to maintain online summary statistics in a very memory friendly
// way. From their results, it takes fewer than 2500 tuples to achieve an error
// bound of 0.005 for 1B samples. Another option is a C++ implementation of the
// T-Digest data structure (https://github.com/tdunning/t-digest,
// https://github.com/derrickburns/tdigest), which uses k-means clustering to
// efficiently compute approximate quantiles.
template <typename T>
class Summary : public Metric, public Accumulator<Summary<T>> {
  static_assert(std::is_arithmetic_v<T>);

public:
  explicit Summary(std::string_view id, std::string_view units, int window_size)
      : Metric(id), units_(units), window_size_(window_size),
        initialized_(false), min_(0.0), p50_(0.0), p90_(0.0), p95_(0.0),
        p99_(0.0), p999_(0.0), max_(0.0), total_samples_(0), mean_(0.0),
        squared_total_(0.0), variance_(0.0) {}

  double GetMin() { return GetPercentileInternal(0.0); }
  double Get50thPercentile() { return GetPercentileInternal(50.0); }
  double Get90thPercentile() { return GetPercentileInternal(90.0); }
  double Get95thPercentile() { return GetPercentileInternal(95.0); }
  double Get99thPercentile() { return GetPercentileInternal(99.0); }
  double Get999thPercentile() { return GetPercentileInternal(99.9); }
  double GetMax() { return GetPercentileInternal(100.0); }
  double GetMean() { return mean_; }
  double GetStddev() { return std::sqrt(variance_); }
  int64_t GetNumSamples() { return total_samples_; }

  Summary &operator<<(const T &value);

  sss::Status Accumulate(const sss::StatusVal<Summary<T>> &other) override;

  std::string ToString() override;

  MetricProto ToProto() override;

private:
  // A visitor that counts the total number of values in the subtree rooted by a
  // given node. Instead of maintaining a value in conjunction with the
  // metadata, we just use the metadata to keep a multi-set like structure. An
  // update is semantically equivalent to inserting another node, but we avoid
  // physically doing so. Instead, we increment the subtree counter.
  //
  // When calculating percentiles, we know how many samples have been collected
  // and can then find the node whose right subtree count is equal to the
  // expected number of samples greater than the target percentile.
  template <class NodeType>
  class SubtreeCountVisitor
      : public VisitorInterface<NodeType, typename NodeType::value_type,
                                typename NodeType::metadata_type> {
  public:
    void OnInsert(NodeType *curr) override { Visit(curr); }
    void OnRemove(NodeType *curr) override { Visit(curr); }
    void OnUpdate(NodeType *curr) override { Visit(curr); }
    void Update(NodeType *curr, const typename NodeType::value_type &,
                const typename NodeType::metadata_type &) override {
      curr->set_value(curr->value() + 1);
      Visit(curr);
    }

  private:
    void Visit(NodeType *curr) {
      auto *right = curr->right();
      auto *left = curr->left();
      auto left_total = left != nullptr ? left->metadata() : 0;
      auto right_total = right != nullptr ? right->metadata() : 0;
      curr->set_metadata(left_total + right_total + curr->value());
    }
  };

  class PercentileAccessor {
  public:
    template <typename ATreeType>
    sss::StatusVal<typename ATreeType::node_type *>
    FindNthPercentile(const ATreeType &tree, double n);

    template <typename ATreeType> int GetSize(const ATreeType &tree);
  };

  double GetPercentileInternal(double p);

  void UpdatePercentilesAndClearSamples();
  void UpdatePercentile(double p);

  std::string units_;

  // Number of samples collected online before recomputing statistics.
  const int window_size_;

  // For those values greater than the median, we use an augmented tree that
  // tracks the total subtree count and increments the count by one for every
  // duplicate (i.e., is actually a multi-set). When searching for a given
  // percentile, the tree is traversed to find the node whose right subtree
  // count is equal to the expected number of samples greater than the given
  // percentile.
  //
  // E.g., Given 1000 samples and the 99th percentile, we expect that the right
  // subtree of the 99th percentile value will contain 10 nodes.
  typedef ATreeNode<T, int, int> atree_node_type;
  ATree<T, int, int, SubtreeCountVisitor<atree_node_type>, PercentileAccessor>
      samples_;

  // Statistics collected during the lifetime of this distribution object.
  bool initialized_;
  double min_;
  double p50_;
  double p90_;
  double p95_;
  double p99_;
  double p999_;
  double max_;

  int64_t total_samples_;
  double mean_;
  double squared_total_;
  double variance_;
};

template <typename T> void Summary<T>::UpdatePercentilesAndClearSamples() {
  PercentileAccessor accessor;
  if (accessor.GetSize(samples_) == 0)
    return;
  if (!initialized_) {
    min_ = accessor.FindNthPercentile(samples_, 0.0).val.value()->key();
    p50_ = accessor.FindNthPercentile(samples_, 50.0).val.value()->key();
    p90_ = accessor.FindNthPercentile(samples_, 90.0).val.value()->key();
    p95_ = accessor.FindNthPercentile(samples_, 95.0).val.value()->key();
    p99_ = accessor.FindNthPercentile(samples_, 99.0).val.value()->key();
    p999_ = accessor.FindNthPercentile(samples_, 99.9).val.value()->key();
    max_ = accessor.FindNthPercentile(samples_, 100.0).val.value()->key();
    initialized_ = true;
  } else {
    UpdatePercentile(0.0);
    UpdatePercentile(50.0);
    UpdatePercentile(90.0);
    UpdatePercentile(95.0);
    UpdatePercentile(99.0);
    UpdatePercentile(99.9);
    UpdatePercentile(100.0);
  }
  samples_.Clear();
}

template <typename T> void Summary<T>::UpdatePercentile(double p) {
  PercentileAccessor accessor;
  auto node_or = accessor.FindNthPercentile(samples_, p);
  if (node_or.status.t != sss::Ok)
    exit(1);
  auto value = double(node_or.val.value()->key());
  if (p == 0.0) {
    min_ = std::min(min_, value);
  } else if (p == 50.0) {
    auto delta = (value - p50_);
    p50_ += delta / double(total_samples_);
  } else if (p == 90.0) {
    auto delta = (value - p90_);
    p90_ += delta / double(total_samples_);
  } else if (p == 95.0) {
    auto delta = (value - p95_);
    p95_ += delta / double(total_samples_);
  } else if (p == 99.0) {
    auto delta = (value - p99_);
    p99_ += delta / double(total_samples_);
  } else if (p == 99.9) {
    auto delta = (value - p999_);
    p999_ += delta / double(total_samples_);
  } else if (p == 100.0) {
    max_ = std::max(max_, value);
  } else {
    exit(1);
  }
}

template <typename T> Summary<T> &Summary<T>::operator<<(const T &value) {
  PercentileAccessor accessor;
  if (accessor.GetSize(samples_) == window_size_) {
    // Whenever we have seen the maximum number of samples to maintain, we
    // update the internal metrics and reset the samples.
    UpdatePercentilesAndClearSamples();
  }
  samples_.InsertOrUpdate(value, 1, 1);
  ++total_samples_;

  // Compute the running mean and variance.
  double delta = (value - mean_);
  mean_ += delta / double(total_samples_);
  squared_total_ += (value * value);
  variance_ = (squared_total_ / double(total_samples_)) - (mean_ * mean_);
  return *this;
}

template <typename T> double Summary<T>::GetPercentileInternal(double p) {
  UpdatePercentilesAndClearSamples();
  if (p == 0.0) {
    return min_;
  } else if (p == 50.0) {
    return p50_;
  } else if (p == 90.0) {
    return p90_;
  } else if (p == 95.0) {
    return p95_;
  } else if (p == 99.0) {
    return p99_;
  } else if (p == 99.9) {
    return p999_;
  } else if (p == 100.0) {
    return max_;
  } else {
    exit(1);
  }
}

template <typename T>
template <typename ATreeType>
sss::StatusVal<typename ATreeType::node_type *>
Summary<T>::PercentileAccessor::FindNthPercentile(const ATreeType &tree,
                                                  double percentile) {
  if (percentile > 100.0 || percentile < 0.0) {
    // return FailedPrecondition ("Unexpected value for n");
    std::cerr << percentile << std::endl;
    exit(1);
  }

  // Convert the percentile into rank.
  int rank;
  if (percentile == 0.0) {
    rank = 1;
  } else {
    auto size = GetSize(tree);
    rank = (percentile / 100.0) * size;
  }

  auto *curr = tree.root_.left();
  auto *left = curr->left();
  auto *right = curr->right();
  while (left != nullptr || right != nullptr) {
    if (curr->metadata() - (right != nullptr ? right->metadata() : 0) >= rank &&
        (left != nullptr ? left->metadata() : 0) < rank) {
      // If the rank lies somewhere excluding the right subtree, and does not
      // include the left subtree then it must be this node.
      return {sss::Status::Ok(), curr};
    } else if (left != nullptr && left->metadata() >= rank) {
      // If there are `rank` nodes smaller than this node, the n-th node must be
      // in the left subtree.
      curr = left;
    } else {
      // If the number of nodes smaller than `curr` is less than n, the n-th
      // node must be the (n-(left subtree + curr))-th node in the right
      // subtree.
      assert(right != nullptr); // n-th must exist somewhere
      rank = rank - (curr->metadata() - right->metadata());
      curr = right;
    }
    left = curr->left();
    right = curr->right();
  }
  assert(curr->metadata() >= rank);
  return {sss::Status::Ok(), curr};
}

template <typename T>
template <typename ATreeType>
int Summary<T>::PercentileAccessor::GetSize(const ATreeType &tree) {
  if (tree.root_.left() != nullptr) {
    return tree.root_.left()->metadata();
  } else {
    return 0;
  }
}

template <typename T>
sss::Status Summary<T>::Accumulate(const sss::StatusVal<Summary<T>> &other) {
  if (other.status.t != sss::Ok)
    return other.status;

  auto d = other.val.value();
  if (d.name_ != name_) {
    return {sss::FailedPrecondition, "Unexpected metric ID"};
  }

  // Compute the weighted average of the two distribution statistics.
  UpdatePercentilesAndClearSamples();
  min_ += (d.min_ * d.total_samples_ - min_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  p50_ += (d.p50_ * d.total_samples_ - p50_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  p90_ += (d.p90_ * d.total_samples_ - p90_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  p95_ += (d.p95_ * d.total_samples_ - p95_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  p99_ += (d.p99_ * d.total_samples_ - p99_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  p999_ += (d.p999_ * d.total_samples_ - p999_ * total_samples_) /
           double(total_samples_ + d.total_samples_);
  max_ += (d.max_ * d.total_samples_ - max_ * total_samples_) /
          double(total_samples_ + d.total_samples_);
  mean_ += (d.mean_ * d.total_samples_ - mean_ * total_samples_) /
           double(total_samples_ + d.total_samples_);
  variance_ += (d.variance_ * d.total_samples_ - variance_ * total_samples_) /
               double(total_samples_ + d.total_samples_);
  return sss::Status::Ok();
}

template <typename T> std::string Summary<T>::ToString() {
  UpdatePercentilesAndClearSamples();
  std::stringstream ss;
  ss << "units: \"" << units_ << "\", ";
  ss << "summary: ";
  ss << "{mean: " << mean_ << ", ";
  ss << "stddev: " << std::sqrt(variance_) << ", ";
  ss << "samples: " << total_samples_ << "}, ";
  ss << "percentiles: ";
  ss << "{min: " << min_ << ", ";
  ss << "p50: " << p50_ << ", ";
  ss << "p90: " << p90_ << ", ";
  ss << "p95: " << p95_ << ", ";
  ss << "p99: " << p99_ << ", ";
  ss << "p999: " << p999_ << ", ";
  ss << "max: " << max_ << "}";
  return ss.str();
}

template <typename T> MetricProto Summary<T>::ToProto() {
  UpdatePercentilesAndClearSamples();
  MetricProto proto;
  proto.set_name(name_);
  SummaryProto *summary = proto.mutable_summary();
  *(summary->mutable_units()) = units_;
  summary->set_count(total_samples_);
  summary->set_mean(mean_);
  summary->set_stddev(std::sqrt(variance_));
  summary->set_min(min_);
  summary->set_p50(p50_);
  summary->set_p90(p90_);
  summary->set_p95(p95_);
  summary->set_p99(p99_);
  summary->set_p999(p999_);
  summary->set_max(max_);
  return proto;
}

} // namespace rome::metrics