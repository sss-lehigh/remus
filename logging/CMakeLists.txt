
add_library(logging INTERFACE)
target_include_directories(logging INTERFACE 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(logging INTERFACE remus::util)
target_link_libraries(logging INTERFACE spdlog::spdlog fmt::fmt) #Remove this bloat
target_compile_definitions(logging INTERFACE ROME_LOG_LEVEL=${LOG_LEVEL})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(remus::logging ALIAS logging)

