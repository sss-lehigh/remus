# Compile the .proto files into .cc (PROTO_SRCS) and .h (PROTO_HDRS) files. They
# will go in `$(root_builddir)/protos`, since this subfolder is named `protos`
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metrics.proto workloaddriver.proto experiment.proto rdma.proto)

# Build a static library from the generated PROTO_SRCS.
add_library(protos STATIC ${PROTO_SRCS})

# Declare that anyone who uses this `protos` library will also need -lprotobuf
target_link_libraries(protos PUBLIC protobuf::libprotobuf)

# Add the top-level `build` folder to the include path, so generated headers can
# be included as `<protos/x.pb.h>`
target_include_directories(protos PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)