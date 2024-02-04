
add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE include)

target_link_libraries(rdma INTERFACE rome::logging)

target_link_libraries(rdma INTERFACE rome::metrics) # TODO should be opt-in/out

target_link_libraries(rdma INTERFACE protos) # TODO Remove this bloat

target_link_libraries(rdma INTERFACE sss) # TODO factor in

target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)

add_library(rome::rdma ALIAS rdma)

