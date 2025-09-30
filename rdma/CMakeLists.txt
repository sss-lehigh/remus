# Build the RDMA library
add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)
target_compile_definitions(rdma INTERFACE REMUS_LOG_LEVEL=${LOG_LEVEL})
add_library(remus::rdma ALIAS rdma)

# Build the example programs (more like "functionality tests")

add_executable(root_test test/root.cc)
target_link_libraries(root_test PRIVATE rdma)

add_executable(read_test test/read.cc)
target_link_libraries(read_test PRIVATE rdma)

add_executable(write_test test/write.cc)
target_link_libraries(write_test PRIVATE rdma)

add_executable(helloworld_test test/helloworld.cc)
target_link_libraries(helloworld_test PRIVATE rdma)