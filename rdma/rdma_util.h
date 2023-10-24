#pragma once

#include <infiniband/verbs.h>

#include <cstdio>
#include <memory>
#include <regex>

#include "../logging/logging.h"
#include "../util/status_util.h"
#include "../vendor/sss/status.h"

#define RDMA_CM_CHECK(func, ...)                                               \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    if (ret != 0) {                                                            \
      sss::Status err = {sss::InternalError, ""};                              \
      err << #func << "(): " << strerror(errno);                               \
      return err;                                                              \
    }                                                                          \
  }

#define RDMA_CM_CHECK_TOVAL(func, ...)                                         \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    if (ret != 0) {                                                            \
      sss::Status err = {sss::InternalError, ""};                              \
      err << #func << "(): " << strerror(errno);                               \
      return {err, {}};                                                        \
    }                                                                          \
  }

#define RDMA_CM_ASSERT(func, ...)                                              \
  {                                                                            \
    int ret = func(__VA_ARGS__);                                               \
    ROME_ASSERT(ret == 0, "{}{}{}", #func, "(): ", strerror(errno));           \
  }

struct ibv_context_deleter {
  void operator()(ibv_context *device_ctx) { ibv_close_device(device_ctx); }
};
using ibv_context_unique_ptr =
    std::unique_ptr<ibv_context, ibv_context_deleter>;

struct ibv_pd_deleter {
  void operator()(ibv_pd *pd) { ibv_dealloc_pd(pd); }
};
using ibv_pd_unique_ptr = std::unique_ptr<ibv_pd, ibv_pd_deleter>;

struct ibv_device_list_deleter {
  void operator()(ibv_device **devices) { ibv_free_device_list(devices); }
};
using ibv_device_list_unique_ptr =
    std::unique_ptr<ibv_device *, ibv_device_list_deleter>;

struct ibv_mr_deleter {
  void operator()(ibv_mr *mr) { ibv_dereg_mr(mr); }
};
using ibv_mr_unique_ptr = std::unique_ptr<ibv_mr, ibv_mr_deleter>;

// [mfs]  C++ regexes are *very slow*.  Consider refactor?
//
// [mfs]  If this is on the critical path, the lambdas are going to be a source
//        of latency.
inline sss::StatusVal<std::string> ibdev2netip(std::string_view ib_dev) {
  auto Call = [](std::string cmd) {
    std::array<char, 1024> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.data(), "r"),
                                                  pclose);
    ROME_ASSERT(pipe != nullptr, "Failed open pipe: {}", cmd);
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
    }
    return result;
  };

  auto Split = [](std::string_view str, char delim) {
    std::vector<std::string> tokens;
    size_t start = 0, end = str.find(delim);
    while (end != std::string::npos) {
      tokens.emplace_back(str.substr(start, end - start));
      start = end + 1;
      end = str.find(delim, start);
    }
    return tokens;
  };

  auto ExtractIp = [](const std::string &str) -> sss::StatusVal<std::string> {
    const std::regex r("^\\s*inet\\s+((\\d{1,3}\\.){3}\\d{1,3}).*\n$");
    std::smatch m;
    if (str.empty() || !std::regex_match(str, m, r)) {
      ROME_TRACE("Match: {}", m[0].str());
      sss::Status err = {sss::NotFound, "No IP address found for netdev: "};
      err << str;
      return {err, {}};
    }
    return {sss::Status::Ok(), m[1].str()};
  };

  auto result = Call("ibdev2netdev");
  if (!result.empty()) {
    auto lines = Split(result, '\n');
    for (const auto &l : lines) {
      auto line = Split(l, ' ');
      // Expected: ["mlx5_0","port","1","==>","ibp153s0","(Up)"]
      if (line[0] == ib_dev) {
        auto net_dev = line[4];
        result = Call("ip addr show dev " + net_dev + " | grep 'inet '");
        return ExtractIp(result);
      }
    }
  }
  sss::Status err = {sss::NotFound, "Device address not found: "};
  err << ib_dev;
  return {err, {}};
}