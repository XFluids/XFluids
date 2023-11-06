add_subdirectory(config)
find_library(readini NAMES libreadini.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${readini}" STREQUAL "readini-NOTFOUND")
    add_library(readini "")
    target_sources(readini
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/setup_ini.cpp
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/setupini.h
    )
    target_include_directories(readini
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
    )

    # #add libs
    target_link_libraries(readini
        PUBLIC
        config
        readgrid
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
        add_sycl_to_target(TARGET readini)
    ENDIF()
ELSE()
    message(STATUS "${readini} is located at ${LIBRARY_OUTPUT_PATH}")
    add_library(readini STATIC IMPORTED GLOBAL)
    set_property(TARGET readini
        PROPERTY IMPORTED_LOCATION ${readini})
    target_link_libraries(readini
        INTERFACE
        config
        readgrid
    )
ENDIF()
