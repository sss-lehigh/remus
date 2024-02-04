add_library(hds INTERFACE)
target_include_directories(hds INTERFACE include)
target_compile_definitions(hds INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:GPU>)
target_compile_options(hds INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -Xptxas --verbose>)
target_link_libraries(hds INTERFACE rome)

add_subdirectory(examples)
