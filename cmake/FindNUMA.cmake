if(TARGET numa::numa)
  return()
endif()

include(CheckLibraryExists)

CHECK_LIBRARY_EXISTS(numa numa_available "" _HAVE_NUMA)

set(_found FALSE)

if(${_HAVE_NUMA})
  add_library(numa::numa INTERFACE IMPORTED)
  target_link_libraries(numa::numa INTERFACE numa)
  set(_found TRUE)
endif()

set(NUMA_FOUND ${_found} CACHE BOOL "TRUE if have libnuma" FORCE)

if(${NUMA_FIND_REQUIRED} AND NOT ${NUMA_FOUND})
  message(FATAL_ERROR "Could not find numa library")
endif()
