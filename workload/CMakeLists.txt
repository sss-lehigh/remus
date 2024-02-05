
add_library(workload INTERFACE)
target_include_directories(workload INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(workload INTERFACE metrics)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 
add_library(rome::workload ALIAS workload)

