# Declare boilerplat cmake stuff: minimum version, project name
cmake_minimum_required(VERSION 3.18)
project(IHT_Rome LANGUAGES CXX VERSION 0.0.0)

# Declare flags that the user can override
set(LOG_LEVEL "DEBUG" CACHE STRING "Log level options include TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, and OFF")
set(CXX_STANDARD 23 CACHE STRING "Uses C++20 or 23")

# Ensure that the protobuf compiler is available
find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf version ${Protobuf_VERSION}")

# Use a custom rule (in `cmake/`) to find the ibverbs and rdmacm libraries.
# NB: The rule prints a message, so we don't need a `message()` here
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(RDMA REQUIRED)

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

# Ensure requested C++ is modern enough, then turn on that CXX standard
set(CXX_STANDARDS "20;23")
foreach(S ${CXX_STANDARDS})
    if("${CXX_STANDARD}" STREQUAL "${S}")
        set(CORRECT_CXX_STANDARD TRUE)
    endif()
endforeach()
if(NOT DEFINED CORRECT_CXX_STANDARD)
    message(FATAL_ERROR "CXX_STANDARD must be one of ${CXX_STANDARDS}") 
endif()
message(STATUS "Using CXX_STANDARD=${CXX_STANDARD}")
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})

# Compile `protos/*.proto` into headers and a static library (.a), putting them
# all in the `protos` subfolder of the build tree.
add_subdirectory(protos)

# Now declare the executables to build
add_executable(iht_rome iht/main.cc)
add_executable(iht_rome_test iht/test.cc)

# Make the vendored spdlog library easier to include
target_include_directories(iht_rome PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../vendor/spdlog-1.12.0>)
target_include_directories(iht_rome_test PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../vendor/spdlog-1.12.0>)

# Set up the logging level
target_compile_definitions(iht_rome PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})
target_compile_definitions(iht_rome_test PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})

# Use these if you want to turn on warnings during compilation
# target_compile_options(iht_rome PUBLIC -Wall -Wextra -Werror)
# target_compile_options(iht_rome_test PUBLIC -Wall -Wextra -Werror)

# Declare libraries that the targets depend upon
target_link_libraries(iht_rome PUBLIC protos rdma::ibverbs rdma::cm)
target_link_libraries(iht_rome_test PUBLIC protos rdma::ibverbs rdma::cm)
