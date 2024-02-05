
add_library(logging INTERFACE)
target_include_directories(logging INTERFACE 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(logging INTERFACE spdlog::spdlog fmt::fmt sss) #Remove this bloat
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(rome::logging ALIAS logging)

