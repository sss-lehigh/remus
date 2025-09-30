# Build the "hello" executable
add_executable(hello hello.cc)
target_link_libraries(hello PRIVATE rdma)

# Build the "lazy list" executable
add_executable(lazy_list lazy_list.cc)
target_link_libraries(lazy_list PRIVATE rdma)
