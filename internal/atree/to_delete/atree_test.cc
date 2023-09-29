#include "atree.h"

#include <cstddef>
#include <limits>
#include <random>
#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "rome/internal/atree/value.h"
#include "rome/testutil/status_matcher.h"

namespace rome {
namespace {

using ::testing::Eq;
using ::testutil::IsOk;
using ::testutil::StatusIs;

#if defined(__clang__)
TEST(NodeTest, KeyTest) {
  // Test plan: Construct a node with only a key to ensure that the empty base
  // optimization is working correctly. Check that the size is equal to the two
  // children pointers plus the size of the key type.
  auto node = ATreeNode<int, EmptyValue, EmptyMetadata>(18015, {}, {});
  EXPECT_EQ(sizeof(node), sizeof(int) + (2 * sizeof(std::nullptr_t)));
}

TEST(NodeTest, KeyAndValueTest) {
  // Test plan: Construct a node with a key and value. Check that the size is
  // equal to the two children pointers plus the size of the key type and the
  // value type.
  auto node = ATreeNode<int64_t, int, EmptyMetadata>(18015, 10, {});
  EXPECT_EQ(sizeof(node),
            sizeof(int64_t) + sizeof(int) + (2 * sizeof(std::nullptr_t)));
  EXPECT_EQ(node.value(), 10);
}

TEST(NodeTest, StringValueTest) {
  // Test plan: Test that the size of a node constructed from a `std::string` is
  // expected.
  std::string key = "TestKey";
  std::string value = "AFDFJAE923R29T90GSBHAJRW134A";
  int expected_size = sizeof(key) + sizeof(value);
  auto node = ATreeNode<std::string, std::string, EmptyMetadata>(
      key, std::string(value), {});
  EXPECT_EQ(sizeof(node), expected_size + (2 * sizeof(std::nullptr_t)));
}

TEST(NodeTest, KeyValueAndMetadataTest) {
  // Test plan: Create a node with all three types of info and check that the
  // size is as expected.
  typedef struct __attribute__((packed)) metadata {
    int x;
    double y;
  } metadata_t;

  auto node =
      ATreeNode<int, double, metadata_t>(07030, 42.0, metadata_t{59718, 2.6});

  int expected_size = sizeof(int) + sizeof(double) + sizeof(metadata_t) +
                      (2 * sizeof(std::nullptr_t));
  EXPECT_EQ(sizeof(node), expected_size);
}
#endif

TEST(ATreeTest, FindReturnsNotFound) {
  // Test plan: Call `Find` on an empty tree and verify that it returns
  // `kNotFound`.
  ATree<std::string, EmptyValue, EmptyMetadata, EmptyVisitor> atree;
  EXPECT_THAT(atree.Find("key"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(ATreeTest, InsertInsertsKey) {
  // Test plan: Call `Find` on an inserted key and verify that it returns
  // the node corresponding to the key.
  ATree<std::string, EmptyValue, EmptyMetadata, EmptyVisitor> atree;
  EXPECT_OK(testutil::GetStatus(atree.Insert("key", {}, {})));
  auto found = atree.Find("key");
  ASSERT_THAT(testutil::GetStatus(found), IsOk());
  EXPECT_THAT(found.value()->key(), Eq("key"));
}

TEST(ATreeTest, AllInsertedKeysCanBeFound) {
  // Test plan: Insert several (`int`) keys and check that each of them can be
  // found correctly.
  ATree<int, EmptyValue, EmptyMetadata, EmptyVisitor> atree;

  std::mt19937_64 mt;
  std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max());
  std::vector<int> keys;
  for (int i = 0; i < 10000; ++i) {
    auto key = dist(mt);
    EXPECT_OK(testutil::GetStatus(atree.Insert(key, {}, {})));
    keys.push_back(key);
  }

  for (auto k : keys) {
    EXPECT_OK(atree.Find(k));
  }
}

template <class T>
class SubtreeCountVisitor : public VisitorInterface<T, typename T::value_type,
                                                    typename T::metadata_type> {
 public:
  void OnInsert(T* curr) override { Visit(curr); }
  void OnRemove(T* curr) override { Visit(curr); }

  // Do nothing for updates
  void OnUpdate(T* curr) override {}
  void Update(T* curr, const typename T::value_type& value,
              const typename T::metadata_type& metadata) override {}

 private:
  void Visit(T* curr) {
    auto* right = curr->right();
    auto* left = curr->left();
    auto left_total = left != nullptr ? left->metadata() : 0;
    auto right_total = right != nullptr ? right->metadata() : 0;
    curr->set_metadata({left_total + right_total + 1});
  }
};

TEST(ATreeTest, InsertAugmentsKeys) {
  using NodeType = ATreeNode<std::string, EmptyValue, int>;
  ATree<std::string, EmptyValue, int, SubtreeCountVisitor<NodeType>> atree;
  EXPECT_OK(atree.Insert("aaa", {}, 1));
  auto found = atree.Find("aaa");
  ASSERT_OK(found);
  EXPECT_THAT(found.value()->metadata(), Eq(1));

  EXPECT_OK(atree.Insert("zzz", {}, 1));
  found = atree.Find("aaa");
  ASSERT_OK(found);
  EXPECT_THAT(found.value()->metadata(), Eq(2));

  EXPECT_OK(atree.Insert("jjj", {}, 1));
  found = atree.Find("aaa");
  ASSERT_OK(found);
  EXPECT_THAT(found.value()->metadata(), Eq(3));
  found = atree.Find("zzz");
  ASSERT_OK(found);
  EXPECT_THAT(found.value()->metadata(), Eq(2));

  EXPECT_OK(atree.Insert("a", {}, 1));
  found = atree.Find("aaa");
  ASSERT_OK(found);
  EXPECT_THAT(found.value()->metadata(), Eq(4));
}

class SubtreeCountAccessor {
 public:
  template <typename ATreeType>
  absl::StatusOr<typename ATreeType::node_type*> FindNthNode(
      const ATreeType& tree, int n) {
    if (n > tree.Size() || n <= 0) {
      return absl::FailedPreconditionError("Unexpected value for n");
    }
    auto* curr = tree.root_.left();
    auto* left = curr->left();
    auto* right = curr->right();
    while (left != nullptr || right != nullptr) {
      if (curr->metadata() - (right != nullptr ? right->metadata() : 0) == n) {
        return curr;
      } else if (left != nullptr && left->metadata() >= n) {
        // If there are at least n nodes smaller than this node, the n-th node
        // must be in the left subtree. We include n here because the n-th node
        // might be the rightmost node in the left subtree.
        curr = left;
      } else {
        // If the number of nodes smaller than `curr` is less than n, the n-th
        // node must be the (n-(left subtree + curr))-th node in the right
        // subtree.
        assert(right != nullptr);  // n-th must exist somewhere
        n = n - (curr->metadata() - right->metadata());
        curr = right;
      }
      left = curr->left();
      right = curr->right();
    }
    return curr;
  }
};

TEST(ATreeTest, AccessorAccessesTree) {
  // Test plan: Create a tree with an accessor that will find the node whose
  // subtree is equal to a certain size.
  using NodeType = ATreeNode<int, EmptyValue, int>;
  auto atree = ATree<int, EmptyValue, int, SubtreeCountVisitor<NodeType>,
                     SubtreeCountAccessor>();

  // Tree:
  //       root
  //       /
  //      10
  //      / \
  //     0  30
  //        / \
  //       20 40
  //            \
  //            50

  ASSERT_OK(atree.Insert(10, {}, 1));
  ASSERT_OK(atree.Insert(0, {}, 1));
  ASSERT_OK(atree.Insert(30, {}, 1));
  ASSERT_OK(atree.Insert(20, {}, 1));
  ASSERT_OK(atree.Insert(40, {}, 1));
  ASSERT_OK(atree.Insert(50, {}, 1));

  SubtreeCountAccessor accessor;
  auto found = accessor.FindNthNode(atree, 0);
  EXPECT_THAT(found, StatusIs(absl::StatusCode::kFailedPrecondition));

  found = accessor.FindNthNode(atree, 1);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 0);

  found = accessor.FindNthNode(atree, 2);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 10);

  found = accessor.FindNthNode(atree, 3);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 20);

  found = accessor.FindNthNode(atree, 4);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 30);

  found = accessor.FindNthNode(atree, 5);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 40);

  found = accessor.FindNthNode(atree, 6);
  EXPECT_OK(found);
  EXPECT_EQ(found.value()->key(), 50);
}

template <typename NodeType>
class UpdateCountVisitor
    : public VisitorInterface<NodeType, typename NodeType::value_type,
                              typename NodeType::metadata_type> {
 public:
  void OnInsert(NodeType* curr) {}
  void OnRemove(NodeType* curr) {}
  void OnUpdate(NodeType* curr) {}
  void Update(NodeType* curr, const typename NodeType::value_type& value,
              const typename NodeType::metadata_type& metadata) {
    curr->set_value(value);
    curr->set_metadata(curr->metadata() + 1);
  }
};

TEST(ATreeTest, InsertOrUpdateInsertsAndUpdates) {
  // Test plan: Create a tree that stores integer values and augments each node
  // with its subtree count. Insert a few nodes then use `InsertOrUpdate` to
  // insert a new node. Check that the newly inserted node conforms to expected
  // values. Then, perform `InsertOrUpdate` on the same node and check that the
  // value is updated according the `UpdateCountVisitor`.
  using NodeType = ATreeNode<std::string, int, int>;
  ATree<std::string, int, int, UpdateCountVisitor<NodeType>> atree;
  ASSERT_OK(atree.Insert("j", 1, {0}));
  ASSERT_OK(atree.Insert("z", 1, {0}));
  ASSERT_OK(atree.Insert("n", 1, {0}));
  ASSERT_OK(atree.Insert("a", 1, {0}));

  EXPECT_OK(atree.InsertOrUpdate("f", -1, {0}));

  auto found = atree.Find("f");
  EXPECT_OK(found);
  EXPECT_THAT(found.value()->value(), Eq(-1));
  EXPECT_THAT(found.value()->metadata(), Eq(0));

  EXPECT_OK(atree.InsertOrUpdate("f", 42, {100}));
  EXPECT_THAT(found.value()->value(), Eq(42));
  EXPECT_THAT(found.value()->metadata(), Eq(1));
}

TEST(ATreeTest, RemovedKeyNotFound) {
  // Test plan: Insert a handful of keys then remove one and check that `Find`
  // called on the removed key returns `kNotFound`.
  auto atree = ATree<int, EmptyValue, EmptyMetadata, EmptyVisitor>();

  std::mt19937_64 mt;
  std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max());
  std::vector<int> keys;
  for (int i = 0; i < 10000; ++i) {
    auto key = dist(mt);
    EXPECT_OK(testutil::GetStatus(atree.Insert(key, {}, {})));
    keys.push_back(key);
  }

  auto index = dist(mt) % keys.size();
  EXPECT_OK(atree.Remove(keys[index]));
  EXPECT_THAT(atree.Find(keys[index]), StatusIs(absl::StatusCode::kNotFound));
}

TEST(ATreeTest, RemoveAugmentsKeys) {
  using NodeType = ATreeNode<std::string, EmptyValue, int>;
  ATree<std::string, EmptyValue, int, SubtreeCountVisitor<NodeType>> atree;
  ASSERT_OK(atree.Insert("j", {}, {1}));
  ASSERT_OK(atree.Insert("z", {}, {1}));
  ASSERT_OK(atree.Insert("n", {}, {1}));
  ASSERT_OK(atree.Insert("a", {}, {1}));
  EXPECT_OK(atree.Remove("n"));
  auto found = atree.Find("j");
  ASSERT_OK(found);
  EXPECT_EQ(found.value()->metadata(), 3);
}

}  // namespace
}  // namespace rome
