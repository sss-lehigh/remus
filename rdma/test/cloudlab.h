#pragma once

#include <cstdint>
#include <string>

/// A function that translates numerical ids into DNS names.  This version is
/// specific to CloudLab, where id "0" gets the name "node0", and so forth.
///
/// NB: This is not really necessary on cloudlab, because the mapping is so
///     easy, but we represent it like this so that the act of porting Remus to
///     non-CloudLab systems is more orthogonal.
///
/// @param id The numerical id to translate
///
/// @return The string DNS name for that id
std::string id_to_dns_name(uint64_t id) {
  return std::string("node") + std::to_string(id);
}