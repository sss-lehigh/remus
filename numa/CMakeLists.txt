
add_library(rome_numa INTERFACE)
target_include_directories(rome_numa INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(rome_numa INTERFACE numa::numa)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(rome::numa ALIAS rome_numa)

add_executable(pinning_test test/pinning_test.cc)
target_link_libraries(pinning_test PUBLIC rome_numa)
add_test(pinning_test pinning_test)

