
function(add_test_executable TARGET SOURCE)
    set(options DISABLE_TEST)
    set(oneValueArgs "")
    set(multiValueArgs LIBS)

    cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    add_executable(${TARGET} ${SOURCE})
   
    foreach(NAME "${TARGET}")
        target_link_libraries(${NAME} PRIVATE remus GTest::gtest GTest::gtest_main GTest::gmock ${__LIBS})
    endforeach()
    
    if(NOT ${__DISABLE_TEST})
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    endif()
endfunction()

function(set_standard)
  # Ensure requested C++ is modern enough, then turn on that CXX standard
  set(CXX_STANDARDS "20;23")
  foreach(S ${CXX_STANDARDS})
    if("${CXX_STANDARD}" STREQUAL "${S}")
      set(CORRECT_CXX_STANDARD TRUE)
    endif()
  endforeach()
  if(NOT DEFINED CORRECT_CXX_STANDARD)
    message(FATAL_ERROR "CXX_STANDARD must be one of ${CXX_STANDARDS}") 
  endif()
  message(STATUS "Using CXX_STANDARD=${CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
endfunction()
