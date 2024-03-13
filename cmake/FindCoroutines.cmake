
if(TARGET std::coroutines)
    return()
endif()

include(CheckIncludeFileCXX)

set(_has_std FALSE)
set(_has_experimental FALSE)
set(_found FALSE)

check_include_file_cxx("coroutine" _CXX_COROUTINES_HAVE_HEADER)


if(NOT ${_CXX_COROUTINES_HAVE_HEADER})
    check_include_file_cxx("experimental/coroutine" _CXX_COROUTINES_HAVE_EXPERIMENTAL_HEADER)
    if(${_CXX_COROUTINES_HAVE_EXPERIMENTAL_HEADER})
        set(_has_experimental TRUE)
        set(_found TRUE)
    endif()
else()
    set(_has_std TRUE)
    set(_found TRUE)
endif()




if(${_found})
    add_library(std::coroutines INTERFACE IMPORTED)

    if("cpp${CMAKE_CXX_COMPILER_ID}" MATCHES "cpp.*Clang")
        target_compile_options(std::coroutines INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines-ts>)
        target_compile_options(std::coroutines INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fcoroutines-ts>)
    elseif("cpp${CMAKE_CXX_COMPILER_ID}" MATCHES "cpp.*GNU")
        target_compile_options(std::coroutines INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)
        target_compile_options(std::coroutines INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fcoroutines>)
    endif()
endif()

set(Coroutines_FOUND ${_found} CACHE BOOL "TRUE if we have some coroutines support" FORCE)

if(Coroutines_FIND_REQUIRED AND NOT Coroutines_FOUND)
    message(FATAL_ERROR "Could not find coroutines library")
endif()
