
add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(rdma INTERFACE rome::logging)

target_link_libraries(rdma INTERFACE rome::metrics) # TODO should be opt-in/out

target_link_libraries(rdma INTERFACE protos) # TODO Remove this bloat

target_link_libraries(rdma INTERFACE sss) # TODO factor in

target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(rome::rdma ALIAS rdma)

add_executable(loopback_test test/loopback.cc)
target_link_libraries(loopback_test PRIVATE rdma)

