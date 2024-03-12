add_executable(0_linked_list 0_linked_list.cc)
target_link_libraries(0_linked_list PRIVATE hds)
add_test(0_linked_list 0_linked_list)

if(GPU)
add_executable(1_gpu_linked_list 1_gpu_linked_list.cu)
target_link_libraries(1_gpu_linked_list PRIVATE hds)
add_test(1_gpu_linked_list 1_gpu_linked_list)
endif()

add_executable(2_rdma_linked_list 2_rdma_linked_list.cc)
target_link_libraries(2_rdma_linked_list PRIVATE hds)
add_test(2_rdma_linked_list 2_rdma_linked_list)

add_executable(3_kv_linked_list 3_kv_linked_list.cc)
target_link_libraries(3_kv_linked_list PRIVATE hds)
add_test(3_kv_linked_list 3_kv_linked_list)

if(GPU)
add_executable(4_gpu_kv_linked_list 4_gpu_kv_linked_list.cu)
target_link_libraries(4_gpu_kv_linked_list PRIVATE hds)
add_test(4_gpu_kv_linked_list 4_gpu_kv_linked_list)
endif()

add_executable(6_unordered_map 6_unordered_map.cc)
target_link_libraries(6_unordered_map PRIVATE hds)
add_test(6_unordered_map 6_unordered_map)

if(GPU)
add_executable(7_gpu_unordered_map 7_gpu_unordered_map.cu)
target_link_libraries(7_gpu_unordered_map PRIVATE hds)
add_test(7_gpu_unordered_map 7_gpu_unordered_map)
endif()


