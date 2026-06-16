add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)

target_compile_definitions(rdma INTERFACE REMUS_LOG_LEVEL=${LOG_LEVEL})

add_library(remus::rdma ALIAS rdma)

# TESTS

add_executable(multinode_test test/multinode.cc)
target_link_libraries(multinode_test PRIVATE rdma)
add_test(multinode_test multinode_test)

add_executable(allocator_test test/allocator.cc)
target_link_libraries(allocator_test PRIVATE rdma)
add_test(allocator_test allocator_test)

add_executable(one_sided_test test/one_sided.cc)
target_link_libraries(one_sided_test PRIVATE rdma)
add_test(one_sided_test one_sided_test)

add_executable(root_test test/root.cc)
target_link_libraries(root_test PRIVATE rdma)
add_test(root_test root_test)


# BENCHMARKS

add_executable(one_sided_benchmark benchmark/one_sided.cc)
target_link_libraries(one_sided_benchmark PRIVATE rdma)