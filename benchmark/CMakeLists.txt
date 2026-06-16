set(EXEC_NAME "${EXEC_NAME}")
# Guard against duplicate target definition
if(NOT TARGET ${EXEC_NAME})
  add_executable(${EXEC_NAME} ${EXEC_NAME}.cc)
  target_link_libraries(${EXEC_NAME} PRIVATE rdma)
  target_compile_definitions(${EXEC_NAME} PRIVATE TYPE_SIZE=${TYPE_SIZE})
endif()


