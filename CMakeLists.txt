cmake_minimum_required(VERSION 3.18)
project(remus LANGUAGES CXX VERSION 0.0.0)

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
message(STATUS "Using protobuf version ${Protobuf_VERSION}")

add_subdirectory(rdma)
option(BUILD_BENCHMARK "Build benchmark" OFF)
if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

###################################### Install ######################################
# Install the target itself
install(TARGETS rdma
        EXPORT remusTargets
        INCLUDES DESTINATION include)

# Install public headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rdma/include/ DESTINATION include)

# Package and export config
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/remusConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/remusConfig.cmake"
  INSTALL_DESTINATION lib/cmake/remus
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/remusConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT remusTargets
        FILE remusTargets.cmake
        NAMESPACE remus::
        DESTINATION lib/cmake/remus)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/remusConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/remusConfigVersion.cmake"
  DESTINATION lib/cmake/remus)

# Install find-modules needed by remusConfig
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRDMA.cmake"
  # "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindNUMA.cmake" TODO: Add NUMA support
  DESTINATION lib/cmake/remus)
