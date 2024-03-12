## Benchmarking

add_executable(unordered_map unordered_map.cc)
target_link_libraries(unordered_map PRIVATE hds)

if(GPU)
  add_executable(gpu_unordered_map gpu_unordered_map.cu)
  target_link_libraries(gpu_unordered_map PRIVATE hds)
endif()

