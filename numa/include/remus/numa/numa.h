#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numa.h>
#include <optional>
#include <pthread.h>
#include <regex>
#include <sstream>
#include <stdexcept>

#pragma once

/// numa namespace
namespace remus::numa {

/// Can use numa
inline bool can_use_numa() { return numa_available() != -1; }

/// Get numa nodes
inline int numa_nodes() { return numa_max_node() + 1; }

/// Get numa node of a cpu
inline int numa_node_of_cpu(int cpu) {
  int node = ::numa_node_of_cpu(cpu);
  if (errno == EINVAL) {
    throw std::runtime_error("Invalid cpu: " + std::to_string(cpu));
  }
  return node;
}

/// Get number of cpus
inline int num_cpus() {
  std::fstream possible("/sys/devices/system/cpu/online", std::ios_base::in);
  std::string res;
  possible >> res;
  std::regex e("-\\d+");
  std::smatch m;
  std::regex_search(res, m, e);
  if (m.begin() == m.end()) {
    throw std::runtime_error("Unable to get num cpus");
  }
  std::string s = *m.begin();
  return atoi(s.c_str() + 1) + 1;
}

/// Core information
struct Core {
  /// Core id
  int core_id;
  /// Sibling core id
  std::optional<int> sibling_core;
  /// Numa node of core
  int numa_node;
};

/// Get all cores
inline std::vector<Core> get_cores() {
  std::vector<Core> cores;

  auto cpus = num_cpus();

  cores.reserve(cpus);

  bool has_numa = can_use_numa();

  for (int i = 0; i < cpus; ++i) {

    int core_id = i;
    std::optional<int> sibling_core = std::nullopt;
    int numa_node = has_numa ? numa_node_of_cpu(i) : -1;

    std::fstream cpu_info("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list",
                          std::ios_base::in);
    std::string res;
    cpu_info >> res;
    std::stringstream cpu_info_stream(res);
    char search_by = res.find(",") < res.size() ? ',' : '-';
    std::string core;
    while (std::getline(cpu_info_stream, core, search_by)) {
      int tmp = atoi(core.c_str());
      if (tmp != core_id) {
        sibling_core = {tmp};
        break;
      }
    }

    cores.push_back(Core{core_id, sibling_core, numa_node});
  }

  return cores;
}

/// Pin thread to core
inline void pin_core(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0) {
    throw std::runtime_error("Unable to pin to CPU " + std::to_string(core));
  }
}

/// Check if core numa node is in valid numa nodes
inline bool core_is_in_valid_numa_node(const std::vector<int> &valid_numa_nodes, const Core &core) {
  for (auto &n : valid_numa_nodes) {
    if (n == core.numa_node) {
      return true;
    }
  }
  return false;
}

///
/// We have 3 policy pieces: filling cores within a numa node,
/// cores across a numa node, and filling sibling cores
///

enum class PolicyType {
  NEXT_CORE_IN_NODE,
  NEXT_CORE_SIBLING,
  NEXT_CORE_ACROSS_NODE,
};

template <PolicyType p> struct CompileTimePolicyType {
  constexpr static PolicyType value = p;
};

inline constexpr auto NEXT_CORE_IN_NODE = CompileTimePolicyType<PolicyType::NEXT_CORE_IN_NODE>{};
inline constexpr auto NEXT_CORE_SIBLING = CompileTimePolicyType<PolicyType::NEXT_CORE_SIBLING>{};
inline constexpr auto NEXT_CORE_ACROSS_NODE = CompileTimePolicyType<PolicyType::NEXT_CORE_ACROSS_NODE>{};

///
/// Create policy for core pinning
/// @brief Does FIRST, then SECOND, and then THIRD.
///
/// @tparam FIRST first step
/// @tparam SECOND second step
/// @tparam THIRD thirst step
///
template <PolicyType FIRST, PolicyType SECOND, PolicyType THIRD> class Policy {
public:
  static_assert(FIRST != SECOND and SECOND != THIRD and THIRD != FIRST, "Uniqueness of policies is required");

  /// Create policy
  Policy(CompileTimePolicyType<FIRST>, CompileTimePolicyType<SECOND>, CompileTimePolicyType<THIRD>) {
    auto topology = get_cores();

    /// core in node first, then across, then sibling

    static_assert(FIRST == PolicyType::NEXT_CORE_IN_NODE and SECOND == PolicyType::NEXT_CORE_ACROSS_NODE and
                    THIRD == PolicyType::NEXT_CORE_SIBLING,
                  "Only support cores in node, then across, and then sibling for now");

    if (can_use_numa()) {
      for (int node = 0; node < numa_nodes(); ++node) {

        for (auto &core : topology) {
          if (core.numa_node != node) {
            continue;
          }
          if (core.sibling_core == std::nullopt) {
            cores_map.push_back(core.core_id);
          } else if (core.core_id < *core.sibling_core) {
            cores_map.push_back(core.core_id);
          }
        }
      }
      for (int node = 0; node < numa_nodes(); ++node) {
        for (auto &core : topology) {
          if (core.numa_node != node) {
            continue;
          }
          if (core.sibling_core != std::nullopt and core.core_id > *core.sibling_core) {
            cores_map.push_back(core.core_id);
          }
        }
      }
    } else {

      for (auto &core : topology) {
        if (core.sibling_core == std::nullopt) {
          cores_map.push_back(core.core_id);
        } else if (core.core_id < *core.sibling_core) {
          cores_map.push_back(core.core_id);
        }
      }
      for (auto &core : topology) {
        if (core.sibling_core != std::nullopt and core.core_id > *core.sibling_core) {
          cores_map.push_back(core.core_id);
        }
      }
    }
  }

  /// Pin thread id to core in map (will loop around if oversubscribing cores)
  /// @param id thread id
  void pin(int id) const { pin_core(cores_map[id % cores_map.size()]); }

  auto num_cores() const { return cores_map.size(); }

  Policy(const Policy &) = default;
  Policy(Policy &&) = default;
  Policy &operator=(const Policy &) = default;
  Policy &operator=(Policy &&) = default;

private:
  std::vector<int> cores_map;
};

} // namespace remus::numa
