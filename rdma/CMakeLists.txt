
add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(rdma INTERFACE remus::logging remus::util)

target_link_libraries(rdma INTERFACE remus::metrics)

target_link_libraries(rdma INTERFACE protos)

target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(remus::rdma ALIAS rdma)

add_executable(loopback_test test/loopback.cc)
target_link_libraries(loopback_test PRIVATE rdma)
add_test(loopback_test loopback_test)

add_executable(allocator_test test/allocator.cc)
target_link_libraries(allocator_test PRIVATE rdma)
add_test(allocator_test allocator_test)

add_executable(multinode_test test/multinode.cc)
target_link_libraries(multinode_test PRIVATE rdma)
add_test(multinode_test multinode_test)

