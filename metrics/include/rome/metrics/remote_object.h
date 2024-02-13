#pragma once

#include <cstdint>
#include <string>
#include "../vendor/json-3.11.2/json.hpp"
// https://github.com/nlohmann/json

using namespace std;
using namespace nlohmann;

namespace rome::metrics {

/// Represents a remotely accessible memory region. Used to convey the necessary information for remote nodes to interact with this memory, assuming that they have access to a QP that is connected to it.
class RemoteObject {
public:
    /// An string identifier for this object. Must be unique among remote objects.
    string id = "";
    /// Address of first byte in the memory region.
    uint64_t raddr;
    /// Size of the memory region
    uint32_t size;
    /// Local access key.
    uint32_t lkey;
    /// Remote access key.
    uint32_t rkey;

    /// Serialize to a raw json object
    json as_json(){
        json j;
        j["id"] = id;
        j["raddr"] = raddr;
        j["size"] = size;
        j["lkey"] = lkey;
        j["rkey"] = rkey;
        return j;
    }

    /// Serialize to a string
    std::string serialize(){
        return as_json().dump();
    }

    /// Deserialize a string to fill in the fields
    void deserialize(std::string data){
        json j = json::parse(data);
        id = j["id"];
        raddr = j["raddr"];
        size = j["size"];
        lkey = j["lkey"];
        rkey = j["rkey"];
    }
};

}