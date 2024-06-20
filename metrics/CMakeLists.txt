
add_library(metrics INTERFACE)
target_include_directories(metrics INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(metrics INTERFACE protos)
target_link_libraries(metrics INTERFACE logging util)
target_link_libraries(metrics INTERFACE nlohmann_json::nlohmann_json)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(remus::metrics ALIAS metrics)

