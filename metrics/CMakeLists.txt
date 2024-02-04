
add_library(metrics INTERFACE)
target_include_directories(metrics INTERFACE include)
target_link_libraries(metrics INTERFACE protos sss) # TODO Remove this bloat
target_link_libraries(metrics INTERFACE logging)

add_library(rome::metrics ALIAS metrics)

