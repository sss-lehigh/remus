cmake_minimum_required(VERSION 3.18)
project(remus LANGUAGES CXX VERSION 0.0.0)

# Options
option(DOCS "Set to build documentation" OFF)
option(NUMA "Set to enable numa library support" OFF)

# Flags
set(LOG_LEVEL "DEBUG" CACHE STRING "Log level options include TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, and OFF")
set(CXX_STANDARD 20 CACHE STRING "Uses C++20 or 23")

#######################################################################################################################
# Setup

set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using standard ${CXX_STANDARD}")

set(CMAKE_CXX_STANDARD ${CXX_STANDARD})

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

if(DOCS)
    find_package(Doxygen REQUIRED)  # use doxygen for building docs
    add_custom_target(docs ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating docs through doxygen")    
endif()

# Use a custom rule (in `cmake/`) to find the ibverbs and rdmacm libraries.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(RDMA REQUIRED)

find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf version ${Protobuf_VERSION}")

find_package(fmt 8.1...<8.2 REQUIRED) # defines fmt::fmt

message(STATUS "Using fmt version ${fmt_VERSION}")

find_package(spdlog 1.9...<1.10 REQUIRED) #defines spdlog::spdlog
message(STATUS "Using spdlog version ${spdlog_VERSION}")

find_package(nlohmann_json REQUIRED) # should use 3.10 or 3.11
message(STATUS "Using nlohmann_json version ${nlohmann_json_VERSION}")

if(NUMA)
  message(STATUS "NUMA support enabled")
  find_package(NUMA REQUIRED)
endif()

enable_testing()

#######################################################################################################################
## Libraries

add_subdirectory(protos)

## remus::workload
add_subdirectory(workload)

## remus::logging
add_subdirectory(logging)

## remus::metrics
add_subdirectory(metrics)

## remus::rdma
add_subdirectory(rdma)

## remus::util
add_subdirectory(util)

if(NUMA)
  add_subdirectory(numa)
endif()


#######################################################################################################################
## Install

if(NUMA)

  install(TARGETS workload logging metrics rdma util protos numa
                  EXPORT remusTargets
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION bin
          INCLUDES DESTINATION include)
  
  install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake DESTINATION lib/cmake/)
  install(FILES ${PROJECT_SOURCE_DIR}/share/numa/remusConfig.cmake DESTINATION lib/cmake)

else()

  install(TARGETS workload logging metrics rdma util protos
                  EXPORT remusTargets
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION bin
          INCLUDES DESTINATION include)
  install(FILES ${PROJECT_SOURCE_DIR}/share/remusConfig.cmake DESTINATION lib/cmake)

endif()

# install the include dirs
install(FILES ${PROJECT_SOURCE_DIR}/protos/experiment.proto DESTINATION protos)
install(FILES ${PROJECT_SOURCE_DIR}/protos/rdma.proto DESTINATION protos)
install(FILES ${PROJECT_SOURCE_DIR}/protos/metrics.proto DESTINATION protos)
install(FILES ${PROJECT_SOURCE_DIR}/protos/workloaddriver.proto DESTINATION protos)
# install protos for reuse

install(EXPORT remusTargets
        FILE remusTargets.cmake
        NAMESPACE remus::
        DESTINATION lib/cmake)

install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindRDMA.cmake DESTINATION lib/cmake)
install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindCoroutines.cmake DESTINATION lib/cmake)

#######################################################################################################################
## DEB Package

set(CPACK_PACKAGE_VENDOR "Scalable Systems and Software Lab")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Scalable Systems and Software Lab")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)

#######################################################################################################################


