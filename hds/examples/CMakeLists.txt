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

