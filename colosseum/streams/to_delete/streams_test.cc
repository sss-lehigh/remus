#include "streams.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "rome/testutil/status_matcher.h"
#include "rome/testutil/stream_matcher.h"

namespace rome {

TEST(StreamsTest, TestStreamTest) {
  auto stream = TestStream<int>({1, 2, 3});
  EXPECT_THAT(stream.Next(), testutil::IsOkAndHolds(1));
  EXPECT_THAT(stream.Next(), testutil::IsOkAndHolds(2));
  EXPECT_THAT(stream.Next(), testutil::IsOkAndHolds(3));
  EXPECT_THAT(stream.Next(), testutil::IsStreamTerminatedStatus());
}

TEST(StreamsTest, RandomUniformIntStreamTest) {
  auto stream =
      RandomDistributionStream<std::uniform_int_distribution<int>, int, int>(
          0, 100);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_GE(stream.Next().value(), 0);
    EXPECT_LE(stream.Next().value(), 100);
  }
  EXPECT_THAT(stream.Next(),
              testing::Not(testutil::IsStreamTerminatedStatus()));
}

TEST(StreamsTest, RandomBernoulliStreamTest) {
  auto stream =
      RandomDistributionStream<std::bernoulli_distribution, double>(1);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(stream.Next().value());
  }
  EXPECT_THAT(stream.Next(),
              testing::Not(testutil::IsStreamTerminatedStatus()));
}

enum TestEnum { kOne, kTwo, kThree };

TEST(StreamsTest, WeightedStreamTest) {
  // Test plan: Make a weighted stream an repeatedly sample it, counting the
  // number of events. Compare the observed events to the expected distribution.
  constexpr int kNumTrials = 1000000;

  std::mt19937 mt;
  std::uniform_int_distribution<uint32_t> dist(1, 10);
  std::vector<uint32_t> weights = {dist(mt), dist(mt), dist(mt)};
  int total_weight = std::accumulate(weights.begin(), weights.end(), 0);
  auto stream = WeightedStream<TestEnum>(weights);

  std::vector<int> counts = {0, 0, 0};
  for (int i = 0; i < kNumTrials; ++i) {
    auto next = stream.Next();
    EXPECT_OK(next);
    ++counts[next.value()];
  }

  for (uint32_t i = 0; i < weights.size(); ++i) {
    EXPECT_NEAR(counts[i] / double(kNumTrials),
                weights[i] / double(total_weight), 0.01);
  }
}

TEST(StreamsTest, MappedStreamTest) {
  auto stream_1 = std::make_unique<TestStream<int>>(std::vector<int>{1, 1, 1});
  auto stream_2 = std::make_unique<TestStream<int>>(std::vector<int>{2, 2, 2});
  auto mapped_stream = MappedStream<int, int, int>::Create(
      [](const auto &s1, const auto &s2) -> absl::StatusOr<int> {
        auto x_or = s1->Next();
        auto y_or = s2->Next();
        if (!x_or.ok() || !y_or.ok()) return StreamTerminatedStatus();
        return x_or.value() + y_or.value();
      },
      std::move(stream_1), std::move(stream_2));
  for (int i = 0; i < 3; ++i) {
    EXPECT_THAT(mapped_stream->Next(), testutil::IsOkAndHolds(3));
  }
  EXPECT_THAT(mapped_stream->Next(), testutil::IsStreamTerminatedStatus());
}

}  // namespace rome