### We're just building one executable, but it needs to link to rome
add_executable(iht main.cc)
add_executable(iht_test test.cc)


find_package(fmt 8.1...<8.2 REQUIRED) # defines fmt::fmt
include_directories(${fmt_INCLUDE_DIRS})
message(STATUS "Using fmt version ${fmt_VERSION}")

find_package(spdlog 1.9...<1.10 REQUIRED) #defines spdlog::spdlog
message(STATUS "Using spdlog version ${spdlog_VERSION}")

### IHT
target_link_libraries(iht PRIVATE fmt::fmt)
target_link_libraries(iht PRIVATE spdlog::spdlog)
target_link_libraries(iht PRIVATE rome)

### IHT Test
target_link_libraries(iht_test PRIVATE fmt::fmt)
target_link_libraries(iht_test PRIVATE spdlog::spdlog)
target_link_libraries(iht_test PRIVATE rome)