add_executable(iht_rome main.cc)
add_executable(iht_rome_test test.cc)

# Make the vendored spdlog library easier to include
target_link_libraries(iht_rome PUBLIC rome::rdma rome::workload rome::util)
target_link_libraries(iht_rome_test PUBLIC rome::rdma rome::workload rome::util)

# Set up the logging level
target_compile_definitions(iht_rome PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})
target_compile_definitions(iht_rome_test PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})

# Use these if you want to turn on warnings during compilation
# target_compile_options(iht_rome PUBLIC -Wall -Wextra -Werror)
# target_compile_options(iht_rome_test PUBLIC -Wall -Wextra -Werror)

