add_library(numa INTERFACE)
target_include_directories(numa INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(numa INTERFACE numa::numa)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(remus::numa ALIAS numa)

add_executable(pinning_test test/pinning_test.cc)
target_link_libraries(pinning_test PUBLIC numa)
add_test(pinning_test pinning_test)

