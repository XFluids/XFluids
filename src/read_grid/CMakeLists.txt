find_library(readgrid NAMES libreadgrid.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${readgrid}" STREQUAL "readgrid-NOTFOUND")
  add_library(readgrid "")
  target_sources(readgrid
    PRIVATE
    readgrid.cpp
  )

  IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET readgrid)
  ENDIF()
ELSE()
  message(STATUS "${readgrid} is located at ${LIBRARY_OUTPUT_PATH}")
  add_library(readgrid STATIC IMPORTED GLOBAL)
  set_property(TARGET readgrid
    PROPERTY IMPORTED_LOCATION ${readgrid})
ENDIF()
