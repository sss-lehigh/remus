add_library(hds INTERFACE)
target_include_directories(hds INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_compile_definitions(hds INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:GPU>)
target_compile_options(hds INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -Xptxas --verbose>)

target_link_libraries(hds INTERFACE rdma)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) 

add_library(rome::hds ALIAS hds)

add_subdirectory(examples)
