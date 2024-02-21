# # Assign
find_library(ndAssign NAMES libndAssign.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${ndAssign}" STREQUAL "ndAssign-NOTFOUND")
    add_library(ndAssign "")
    target_include_directories(ndAssign PUBLIC ${CMAKE_SOURCE_DIR}/src/include)

    target_sources(ndAssign
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ndassign.h
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ndassign.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
        add_sycl_to_target(TARGET ndAssign)
    ENDIF()
ELSE()
    message(STATUS "${ndAssign} is located at ${LIBRARY_OUTPUT_PATH}")
    add_library(ndAssign STATIC IMPORTED GLOBAL)
    set_property(TARGET ndAssign
        PROPERTY IMPORTED_LOCATION ${ndAssign})
ENDIF()
