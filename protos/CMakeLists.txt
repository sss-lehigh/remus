### Set up PROTO_SRCS with the names of the protobuf files to compile

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metrics.proto workloaddriver.proto experiment.proto rdma.proto)

### We're making a library called "protos"

add_library(protos SHARED ${PROTO_SRCS})

### Set up additional compiler and linker flags

target_link_libraries(protos PUBLIC protobuf::libprotobuf)
# NB: This makes it so that the produced protobuf headers are accessible to the
#     rest of the code base
target_include_directories(protos PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)