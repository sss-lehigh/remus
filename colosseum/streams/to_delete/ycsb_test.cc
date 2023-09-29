
#include "ycsb.h"
namespace rome {
namespace {

TEST(StreamsTest, YcsbAStreamTest) {
  auto ycsb_a = YcsbStreamFactory<uint32_t>::YcsbA(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_a->Next();
    ROME_INFO("key={}", op->key);
  }
}

TEST(StreamsTest, YcsbBStreamTest) {
  auto ycsb_b = YcsbStreamFactory<uint32_t>::YcsbB(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_b->Next();
    ROME_INFO("key={}", op->key);
  }
}

TEST(StreamsTest, YcsbCStreamTest) {
  auto ycsb_c = YcsbStreamFactory<uint32_t>::YcsbC(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_c->Next();
    ROME_INFO("key={}", op->key);
  }
}

TEST(StreamsTest, YcsbDStreamTest) {
  auto ycsb_d = YcsbStreamFactory<uint32_t>::YcsbD(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_d->Next();
    ROME_INFO("key={}", op->key);
  }
}

TEST(StreamsTest, YcsbEStreamTest) {
  auto ycsb_e = YcsbStreamFactory<uint32_t>::YcsbE(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_e->Next();
    ROME_INFO("key={}", op->key);
  }
}

TEST(StreamsTest, YcsbRStreamTest) {
  auto ycsb_e = YcsbStreamFactory<uint32_t>::YcsbRmw(0, 100);
  for (int i = 0; i < 10; ++i) {
    auto op = ycsb_e->Next();
    ROME_INFO("key={}", op->key);
  }
}


}  // namespace
}  // namespace rome