# Build the "alloc_pol" executable, with 8-byte types
add_executable(alloc_pol_8 alloc_pol.cc)
target_compile_definitions(alloc_pol_8 PUBLIC TYPE_SIZE=8)
target_link_libraries(alloc_pol_8 PRIVATE rdma)

# Build the "alloc_pol" executable, with 1024-byte types
add_executable(alloc_pol_1024 alloc_pol.cc)
target_compile_definitions(alloc_pol_1024 PUBLIC TYPE_SIZE=1024)
target_link_libraries(alloc_pol_1024 PRIVATE rdma)

# Build the "perftest" executable, with 8-byte types
add_executable(perftest_8 perftest.cc)
target_compile_definitions(perftest_8 PUBLIC TYPE_SIZE=8)
target_link_libraries(perftest_8 PRIVATE rdma)

# Build the "perftest" executable, with 1024-byte types
add_executable(perftest_1024 perftest.cc)
target_compile_definitions(perftest_1024 PUBLIC TYPE_SIZE=1024)
target_link_libraries(perftest_1024 PRIVATE rdma)

# Build the "lazylist" executable
add_executable(ds_launch ds_launch.cc)
target_link_libraries(ds_launch PRIVATE rdma)

# Build the "lazylist" executable
add_executable(lf_list lf_list.cc)
target_link_libraries(lf_list PRIVATE rdma)

# TUTORIAL
# Build the "hello" executable
add_executable(hello hello.cc)
target_link_libraries(hello PRIVATE rdma)

# Build the "hello" executable
add_executable(lazy_list lazy_list.cc)
target_link_libraries(lazy_list PRIVATE rdma)
