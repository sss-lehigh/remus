cmake_minimum_required(VERSION 3.18)
project(remus LANGUAGES CXX VERSION 0.0.0)

# Options
option(GPU "Compile for CUDA" OFF)
option(KEEP "Keep ptx and cubin files" OFF)

# Flags
set(LOG_LEVEL "DEBUG" CACHE STRING "Log level options include TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, and OFF")
set(CXX_STANDARD 20 CACHE STRING "Uses C++20 or 23")
set(CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to compile for")

#######################################################################################################################
# Setup

if(GPU)
  enable_language(CUDA)
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using standard ${CXX_STANDARD}")

set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

if(KEEP)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep>)
endif()

# Ensure the LOG_LEVEL is correct
set(LOG_LEVELS "TRACE;DEBUG;INFO;WARN;ERROR;CRITICAL;OFF")
foreach(L ${LOG_LEVELS})
  if("${LOG_LEVEL}" STREQUAL "${L}")
    set(CORRECT_LOG_LEVEL TRUE)
  endif()
endforeach()
if(NOT DEFINED CORRECT_LOG_LEVEL)
  message(FATAL_ERROR "LOG_LEVEL must be one of ${LOG_LEVELS}") 
endif()
message(STATUS "Using LOG_LEVEL=${LOG_LEVEL}")

# Use a custom rule (in `cmake/`) to find the ibverbs and rdmacm libraries.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(RDMA REQUIRED)

find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf version ${Protobuf_VERSION}")

#######################################################################################################################
## Libraries

add_subdirectory(protos) #TODO get rid of this requirement

add_library(spdlog_vendor INTERFACE) # TODO get rid of this requirement
target_include_directories(spdlog_vendor INTERFACE vendor/spdlog-1.12.0)

add_library(sss INTERFACE)
target_include_directories(sss INTERFACE vendor/sss)

## rome::workload
add_subdirectory(workload)

## rome::logging
add_subdirectory(logging)

## rome::metrics
add_subdirectory(metrics)

## rome::rdma
add_subdirectory(rdma)

## rome::util
add_subdirectory(util)

#add_library(rome INTERFACE)
#target_link_libraries(rome INTERFACE protos) # Remove this bloat
#target_link_libraries(rome INTERFACE spdlog_vendor) #Remove this bloat
#target_link_libraries(rome INTERFACE colosseum) #Remove this bloat
#
#target_link_libraries(rome INTERFACE rdma::ibverbs rdma::cm) # All we should link to
#target_include_directories(rome INTERFACE include)

#######################################################################################################################
## Examples

enable_testing()
add_subdirectory(examples/iht)
add_subdirectory(examples/hds)

