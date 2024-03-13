set(Protobuf_IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metrics.proto colosseum.proto testutil.proto)  
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metrics.proto workloaddriver.proto experiment.proto rdma.proto)
install(FILES ${PROTO_HDRS} DESTINATION include/protos)

add_library(protos STATIC ${PROTO_SRCS})
add_library(remus::protos ALIAS protos)
target_include_directories(protos PUBLIC 
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
                            $<INSTALL_INTERFACE:include>)
target_link_libraries(protos PUBLIC protobuf::libprotobuf)

## Compile the .proto files into .cc (PROTO_SRCS) and .h (PROTO_HDRS) files. They
## will go in `$(root_builddir)/protos`, since this subfolder is named `protos`
#
## Build a static library from the generated PROTO_SRCS.
#add_library(protos STATIC ${PROTO_SRCS})
#
## Declare that anyone who uses this `protos` library will also need -lprotobuf
#target_link_libraries(protos PUBLIC protobuf::libprotobuf)
#
## Add the top-level `build` folder to the include path, so generated headers can
## be included as `<protos/x.pb.h>`
#target_include_directories(protos PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)
