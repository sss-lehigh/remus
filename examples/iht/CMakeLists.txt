add_executable(iht_remus main.cc)
add_executable(iht_remus_test test.cc)

# Make the vendored spdlog library easier to include
target_link_libraries(iht_remus PUBLIC remus::rdma remus::workload remus::util)
target_link_libraries(iht_remus_test PUBLIC remus::rdma remus::workload remus::util)

# Set up the logging level
target_compile_definitions(iht_remus PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})
target_compile_definitions(iht_remus_test PUBLIC ROME_LOG_LEVEL=${LOG_LEVEL})

# Use these if you want to turn on warnings during compilation
# target_compile_options(iht_remus PUBLIC -Wall -Wextra -Werror)
# target_compile_options(iht_remus_test PUBLIC -Wall -Wextra -Werror)

