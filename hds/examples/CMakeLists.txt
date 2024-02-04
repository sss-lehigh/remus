add_executable(0_linked_list 0_linked_list.cu)
target_link_libraries(0_linked_list PRIVATE hds)
add_test(0_linked_list 0_linked_list)
