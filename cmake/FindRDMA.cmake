if(TARGET rdma::ibverbs)
    return()
endif()

CHECK_LIBRARY_EXISTS(ibverbs ibv_create_qp "" _HAVE_IBVERBS)
CHECK_LIBRARY_EXISTS(rdmacm rdma_create_id "" _HAVE_RDMACM)

if(${_HAVE_IBVERBS})
    add_library(rdma::ibverbs INTERFACE IMPORTED)
    target_link_libraries(rdma::ibverbs INTERFACE ibverbs)
endif()

if(${_HAVE_RDMACM})
    add_library(rdma::cm INTERFACE IMPORTED)
    target_link_libraries(rdma::cm INTERFACE rdmacm)
endif()

set(_found FALSE)
if(${_HAVE_IBVERBS} AND ${_HAVE_RDMACM})
    set(_found TRUE)
endif()

set(RDMA_FOUND ${_found} CACHE BOOL "TRUE if have rdmacm and ibverbs" FORCE)

if(${RDMA_FIND_REQUIRED} AND NOT ${RDMA_FOUND})
    message(FATAL_ERROR "Could not find RDMA libraries")
endif()
