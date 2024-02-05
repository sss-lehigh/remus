
add_library(metrics INTERFACE)
target_include_directories(metrics INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(metrics INTERFACE protos sss) # TODO Remove this bloat
target_link_libraries(metrics INTERFACE logging)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(rome::metrics ALIAS metrics)

