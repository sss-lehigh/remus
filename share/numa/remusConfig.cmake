
find_package(Protobuf REQUIRED) # protobuf
find_package(RDMA REQUIRED) # defines rdma::ibverbs and rdma::cm for linking
find_package(fmt 8.1...<8.2 REQUIRED) # defines fmt::fmt
find_package(spdlog 1.9...<1.10 REQUIRED) #defines spdlog::spdlog
find_package(nlohmann_json REQUIRED)
find_package(NUMA REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/remusTargets.cmake)
