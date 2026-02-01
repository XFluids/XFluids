add_subdirectory(inishape)
add_subdirectory(settings)
add_subdirectory(outformat)
add_subdirectory(src)

# # # add readini lib
# find_library(readini NAMES libreadini.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini}" STREQUAL "readini-NOTFOUND")
    add_library(readini "")

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

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini)
    ENDIF()