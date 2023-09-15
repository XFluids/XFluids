find_library(libmpiUtils NAMES libmpiUtils.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${libmpiUtils}" STREQUAL "libmpiUtils-NOTFOUND")
  include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})
  add_library(mpiUtils "")

  target_sources(mpiUtils
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestForException.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpiEnums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.h
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.h
  )

  IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET mpiUtils)
  ENDIF()

  target_link_libraries(mpiUtils
    PUBLIC
    ${MPI_CXX}
  )
ELSE()
  message(STATUS "${libmpiUtils} is located at ${LIBRARY_OUTPUT_PATH}")
  add_library(mpiUtils STATIC IMPORTED GLOBAL)
  set_property(TARGET mpiUtils
    PROPERTY IMPORTED_LOCATION ${libmpiUtils})
  target_link_libraries(mpiUtils
    INTERFACE
    ${MPI_CXX}
  )
ENDIF()
