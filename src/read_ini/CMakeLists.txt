add_subdirectory(src)
add_subdirectory(inishape)
add_subdirectory(settings)
add_subdirectory(strsplit)

# # add readini lib
find_library(readini NAMES libreadini.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${readini}" STREQUAL "readini-NOTFOUND")
    add_library(readini "")
    target_include_directories(readini PUBLIC ${CMAKE_SOURCE_DIR}/src/include)

    # #add libs
    target_link_libraries(readini
        PUBLIC
        outFmt
        readini_cos
        readini_set
        readini_rea
        readini_visc
        readini_them
        readini_copy
        readini_print
    )
    target_link_libraries(readini
        PUBLIC
        inishape
        readgrid
        settings
        strsplit
    )

    IF(BOOST_CXX)
        target_link_libraries(readini
            PUBLIC
            ${boost_filesystem}
        )
    ENDIF(BOOST_CXX)

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
        outFmt
        readini_cos
        readini_set
        readini_rea
        readini_visc
        readini_them
        readini_copy
        readini_print
    )
    target_link_libraries(readini
        INTERFACE
        inishape
        readgrid
        settings
        strsplit
    )

    IF(BOOST_CXX)
        target_link_libraries(readini
            INTERFACE
            ${boost_filesystem}
        )
    ENDIF(BOOST_CXX)
ENDIF()
