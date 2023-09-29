### We're just building one executable, but it needs to link to rome
add_executable(iht main.cc)
target_link_libraries(iht PRIVATE rome)