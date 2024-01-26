# # Fmt
find_library(outFmt NAMES liboutFmt.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${outFmt}" STREQUAL "outFmt-NOTFOUND")
    add_library(outFmt "")
    target_include_directories(outFmt PUBLIC ${CMAKE_SOURCE_DIR}/src/include)

    IF(BOOST_CXX)
        target_compile_options(outFmt PUBLIC -DUSE_CXX_BOOST=1)
    ENDIF(BOOST_CXX)

    target_sources(outFmt
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/outformat.h
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/outformat.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
        add_sycl_to_target(TARGET outFmt)
    ENDIF()
ELSE()
    message(STATUS "${outFmt} is located at ${LIBRARY_OUTPUT_PATH}")
    add_library(outFmt STATIC IMPORTED GLOBAL)
    set_property(TARGET outFmt
        PROPERTY IMPORTED_LOCATION ${outFmt})
ENDIF()
