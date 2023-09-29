#pragma once
#include "google/protobuf/text_format.h"

#define ROME_DECLARE_PROTO_FLAG(proto_type)                    \
  bool AbslParseFlag(std::string_view text, proto_type* proto, \
                     std::string* error);                      \
  std::string AbslUnparseFlag(const proto_type& proto);

#define ROME_PROTO_FLAG(proto_type)                                          \
  inline bool AbslParseFlag(std::string_view text, proto_type* proto,        \
                            std::string* error) {                            \
    auto ok =                                                                \
        ::google::protobuf::TextFormat::ParseFromString(text.data(), proto); \
    if (!ok) *error = "Failed to parse proto flag";                          \
    return ok;                                                               \
  }                                                                          \
  inline std::string AbslUnparseFlag(const proto_type& proto) {              \
    return proto.DebugString();                                              \
  }