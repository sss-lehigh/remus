# ---------------------Standard Interface---------------------

add_library(rdma INTERFACE)
target_include_directories(rdma INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(rdma INTERFACE remus::logging remus::util)

target_link_libraries(rdma INTERFACE remus::metrics) # TODO should be opt-in/out

target_link_libraries(rdma INTERFACE protos) # TODO Remove this bloat

target_link_libraries(rdma INTERFACE rdma::ibverbs rdma::cm)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(remus::rdma ALIAS rdma)

# ---------------------Shared lib---------------------

# Create the shared library from rdma_obj.cc
add_library(rdma_shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/rdma_obj.cc)  

# Set include directories for the shared library
target_include_directories(rdma_shared PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

# Link the shared library with the necessary dependencies
target_link_libraries(rdma_shared PUBLIC remus::logging remus::util)
target_link_libraries(rdma_shared PUBLIC remus::metrics) # TODO should be opt-in/out
target_link_libraries(rdma_shared PUBLIC protos) # TODO Remove this bloat
target_link_libraries(rdma_shared PUBLIC rdma::ibverbs rdma::cm)

# Install the shared library
install(TARGETS rdma_shared
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

# enable_testing()

# add_executable(loopback_test test/loopback.cc)
# target_link_libraries(loopback_test PRIVATE rdma)
# add_test(loopback_test loopback_test)

# add_executable(multinode_test test/multinode.cc)
# target_link_libraries(multinode_test PRIVATE rdma)
# add_test(multinode_test multinode_test)

# add_executable(multinode_simple_test test/multinode_simple.cc)
# target_link_libraries(multinode_simple_test PRIVATE rdma)
# add_test(multinode_simple_test multinode_simple_test)

