
add_library(workload INTERFACE)
target_include_directories(workload INTERFACE include)
target_link_libraries(workload INTERFACE metrics)

add_library(rome::workload ALIAS workload)

