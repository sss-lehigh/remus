
add_library(logging INTERFACE)
target_include_directories(logging INTERFACE include)
target_link_libraries(logging INTERFACE spdlog_vendor sss) #Remove this bloat

add_library(rome::logging ALIAS logging)

