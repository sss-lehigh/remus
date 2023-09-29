#pragma once
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <valarray>
#include <vector>

#include "absl/status/statusor.h"
#include "rome/logging/logging.h"
#include "rome/util/distribution_util.h"
#include "rome/util/status_util.h"
#include "streams.h"

namespace rome {

// A `LatestStream` maintains a history of the last `_size` elements,
// implemented as a circular buffer. To draw from the recent insertions,
// `Latest` samples a zipfian distribution in the range [0, `_size`) to
// determine an offset from `newest_` (modulo the size of the history).
template <typename T, int _size>
class LatestStream : public Stream<T> {
 public:
  LatestStream(std::unique_ptr<Stream<T>> stream)
      : stream_(std::move(stream)), rand_(rd_()), latest_dist_(0, _size - 1) {
    for (auto i = 0; i < _size; ++i) {
      [[maybe_unused]] auto init = this->NextInternal();  // Initialize
    }
  }

  inline absl::StatusOr<T> Latest() {
    auto i = latest_dist_(rand_);
    return latest_[(newest_ + i) % _size];
  }

 private:
  absl::StatusOr<T> NextInternal() override {
    auto next = stream_->Next();
    if (next.ok()) {
      latest_[newest_] = *next;
      if (newest_ - 1 > 0) {
        newest_ -= 1;
      } else {
        newest_ = _size - 1;
      }
    }
    return next;
  }

  std::unique_ptr<Stream<T>> stream_;
  std::array<T, _size> latest_;
  int newest_ = _size - 1;

  std::random_device rd_;
  std::default_random_engine rand_;
  rome::zipfian_int_distribution<> latest_dist_;
};

template <typename T>
struct YcsbOp {
  T key;
  enum class Type : uint8_t {
    kGet = 0,
    kInsert,
    kUpdate,
    kScan,
  } type;
  int range;
};

// This is a helper class to generate random keys for YCSB. The idea here is
// pulled directly from the paper (as faithfully as possible). A random key is
// generated and then hashed using a FNV hash and modulo'ed to fit the input
// range. Notably, we choose an upper bound 2x larger than the provided one to
// ensure that the keys are not too clustered (this is a limitation but seems to
// work decently well in practice).
template <typename T, typename D, typename... Args>
class YcsbKeyGenerator : public Stream<T> {
 public:
  YcsbKeyGenerator(uint64_t lo, uint64_t hi, Args... args)
      : mt_(9272), dist_(lo, hi, args...), hi_(hi), lo_(lo) {}

 private:
  absl::StatusOr<T> NextInternal() override {
    auto v = dist_(mt_);
    // auto v = uniform_(mt_);
    uint64_t hash = fnv_offset_basis_;
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[0];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[1];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[2];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[3];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[4];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[5];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[6];
    hash *= fnv_prime_;
    hash ^= ((uint8_t *)&v)[7];
    return (T(hash) % hi_) + lo_;
  }

  std::random_device rd_;
  std::mt19937 mt_;
  D dist_;
  // rome::zipfian_int_distribution<uint64_t> zipf_;
  // std::uniform_int_distribution<> uniform_;

  const T hi_;
  const T lo_;

  static constexpr uint64_t fnv_prime_ = 0x100000001b3;
  static constexpr uint64_t fnv_offset_basis_ = 0xcbf29ce484222325;
};

template <typename T, typename D, typename... Args>
class YcsbStreamFactory : public Stream<T> {
  using op_type = typename YcsbOp<T>::Type;
  using key_type = T;

  static_assert(std::is_integral_v<key_type>);

  static constexpr double kDefaultTheta = 0.99;

  using YcsbAStream = MappedStream<YcsbOp<T>, key_type, op_type>;
  using YcsbBStream = MappedStream<YcsbOp<T>, key_type, op_type>;
  using YcsbCStream = MappedStream<YcsbOp<T>, key_type>;
  using YcsbDStream = MappedStream<YcsbOp<T>, key_type, op_type>;
  using YcsbEStream = MappedStream<YcsbOp<T>, key_type, op_type, int>;

 public:
  static std::unique_ptr<YcsbAStream> YcsbA(const key_type &lo,
                                            const key_type &hi, Args... args) {
    auto key_stream = std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(
        lo, hi, args...);
    auto op_stream = std::make_unique<WeightedStream<op_type>>(
        std::vector<uint32_t>({50, 0, 50, 0}));
    return YcsbAStream::Create(
        [](const auto &keys,
           const auto &ops) -> absl::StatusOr<YcsbOp<key_type>> {
          auto k = keys->Next();
          ROME_CHECK_QUIET(ROME_RETURN(k.status()), k.ok());
          auto o = ops->Next();
          ROME_CHECK_QUIET(ROME_RETURN(o.status()), o.ok());
          return YcsbOp<key_type>{*k, *o, -1};
        },
        std::move(key_stream), std::move(op_stream));
  }

  static std::unique_ptr<YcsbBStream> YcsbB(const key_type &lo,
                                            const key_type &hi, Args... args) {
    auto key_stream = std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(
        lo, hi, args...);
    auto op_stream = std::make_unique<WeightedStream<op_type>>(
        std::vector<uint32_t>({95, 0, 5, 0}));
    return YcsbBStream::Create(
        [](const auto &keys,
           const auto &ops) -> absl::StatusOr<YcsbOp<key_type>> {
          auto k = keys->Next();
          ROME_CHECK_QUIET(ROME_RETURN(k.status()), k.ok());
          auto o = ops->Next();
          ROME_CHECK_QUIET(ROME_RETURN(o.status()), o.ok());
          return YcsbOp<key_type>{*k, *o, -1};
        },
        std::move(key_stream), std::move(op_stream));
  }

  static std::unique_ptr<YcsbCStream> YcsbC(const key_type &lo,
                                            const key_type &hi, Args... args) {
    auto key_stream = std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(
        lo, hi, args...);
    return YcsbCStream::Create(
        [](const auto &keys) -> absl::StatusOr<YcsbOp<key_type>> {
          auto k = keys->Next();
          ROME_CHECK_QUIET(ROME_RETURN(k.status()), k.ok());
          return YcsbOp<key_type>{*k, op_type::kGet, -1};
        },
        std::move(key_stream));
  }

  static std::unique_ptr<YcsbDStream> YcsbD(const key_type &lo,
                                            const key_type &hi, Args... args) {
    auto key_stream = std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(
        lo, hi, args...);
    auto latest_key_stream =
        std::make_unique<LatestStream<key_type, 10000>>(std::move(key_stream));
    auto op_stream = std::make_unique<WeightedStream<op_type>>(
        std::vector<uint32_t>({95, 5, 0, 0}));
    return YcsbDStream::Create(
        [](const auto &keys,
           const auto &ops) -> absl::StatusOr<YcsbOp<key_type>> {
          auto o = ops->Next();
          ROME_CHECK_QUIET(ROME_RETURN(o.status()), o.ok());
          auto k = (*o == op_type::kInsert
                        ? keys->Next()
                        : reinterpret_cast<LatestStream<T, 10000> *>(keys.get())
                              ->Latest());
          ROME_CHECK_QUIET(ROME_RETURN(k.status()), k.ok());
          return YcsbOp<key_type>{*k, *o, -1};
        },
        std::move(latest_key_stream), std::move(op_stream));
  }

  static std::unique_ptr<YcsbEStream> YcsbE(const key_type &lo,
                                            const key_type &hi,
                                            int min_rq_size = 0,
                                            int max_rq_size = 100,
                                            Args... args) {
    auto key_stream = std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(
        lo, hi, args...);
    auto op_stream = std::make_unique<WeightedStream<op_type>>(
        std::vector<uint32_t>({0, 5, 0, 95}));
    auto range_stream =
        std::make_unique<UniformIntStream<int, int>>(min_rq_size, max_rq_size);

    return YcsbEStream::Create(
        [](const auto &keys, const auto &ops,
           const auto &ranges) -> absl::StatusOr<YcsbOp<key_type>> {
          auto k = keys->Next();
          ROME_CHECK_QUIET(ROME_RETURN(k.status()), k.ok())
          auto o = ops->Next();
          ROME_CHECK_QUIET(ROME_RETURN(o.status()), o.ok());
          absl::StatusOr<int> r = (*o == op_type::kScan ? ranges->Next() : -1);
          return YcsbOp<key_type>{*k, *o, *r};
        },
        std::move(key_stream), std::move(op_stream), std::move(range_stream));
  }

  class YcsbRmwStream : public Stream<YcsbOp<key_type>> {
   public:
    YcsbRmwStream(const key_type &lo, const key_type &hi, Args... args)
        : get_next_(true) {
      auto key_stream =
          std::make_unique<YcsbKeyGenerator<key_type, D, Args...>>(lo, hi,
                                                                   args...);
    }

   private:
    absl::StatusOr<YcsbOp<key_type>> NextInternal() override {
      if (get_next_) {
        last_key_ = VALUE_OR_DIE(key_stream_->Next());
      }
      get_next_ = !get_next_;
      return YcsbOp<key_type>{
          last_key_, !get_next_ ? op_type::kInsert : op_type::kGet, -1};
    }

    std::unique_ptr<YcsbKeyGenerator<key_type, D, Args...>> key_stream_;
    bool get_next_;
    key_type last_key_;
  };

  static std::unique_ptr<YcsbRmwStream> YcsbRmw(const key_type &lo,
                                                const key_type &hi,
                                                double theta = kDefaultTheta) {
    return std::make_unique<YcsbRmwStream>(lo, hi, theta);
  }
};

template <typename T>
using DefaultYcsbStreamFactory =
    YcsbStreamFactory<T, rome::zipfian_int_distribution<>, double>;

}  // namespace rome