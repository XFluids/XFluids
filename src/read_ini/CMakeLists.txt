add_subdirectory(inishape)
add_subdirectory(settings)
add_subdirectory(strsplit)

# # add readini lib
find_library(readini NAMES libreadini.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF(BOOST_CXX)
    add_compile_options(-DUSE_CXX_BOOST=1)
    find_library(boost_filesystem NAMES libboost_filesystem.a HINTS "$ENV{BOOST_ROOT}/lib")
    message(STATUS "Find boost_filesystem: ${boost_filesystem}")
ENDIF(BOOST_CXX)

IF("${readini}" STREQUAL "readini-NOTFOUND")
    add_library(readini "")
    target_sources(readini
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/setupini.h
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/setup_ini.cpp
    )
    target_include_directories(readini
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
    )

    # #add libs
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
            boost_filesystem
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
        inishape
        readgrid
        settings
        strsplit
    )

    IF(BOOST_CXX)
        target_link_libraries(readini
            INTERFACE
            boost_filesystem
        )
    ENDIF(BOOST_CXX)
ENDIF()
