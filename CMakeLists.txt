cmake_minimum_required(VERSION 3.18)
project(remus LANGUAGES CXX VERSION 0.1.0)

set(LOG_LEVEL "RELEASE" CACHE STRING "Log level (DEBUG or RELEASE)")
set(CXX_STANDARD 20 CACHE STRING "Uses C++20 or 23")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using standard ${CXX_STANDARD}")
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})

# Turn on all warnings, and make them errors
SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")

# Ensure the LOG_LEVEL is correct
set(LOG_LEVELS "RELEASE;DEBUG")
foreach(L ${LOG_LEVELS})
  if("${LOG_LEVEL}" STREQUAL "${L}")
    set(CORRECT_LOG_LEVEL TRUE)
  endif()
endforeach()
if(NOT DEFINED CORRECT_LOG_LEVEL)
  message(FATAL_ERROR "LOG_LEVEL must be one of ${LOG_LEVELS}") 
endif()
message(STATUS "Using LOG_LEVEL=${LOG_LEVEL}")

# Find dependencies.  Use a custom rule (in `cmake/`) to find ibverbs and rdmacm
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(RDMA REQUIRED)

# Build in the RDMA subfolder and in the benchmark subfolder
add_subdirectory(rdma)
add_subdirectory(benchmark)