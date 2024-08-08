# # # constructor
# find_library(readini_cos NAMES libreadini_cos.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_cos}" STREQUAL "readini_cos-NOTFOUND")
    add_library(readini_cos "")
    IF(BOOST_CXX)
        target_compile_options(readini_cos PUBLIC -DUSE_CXX_BOOST=1)
    ENDIF(BOOST_CXX)

    target_sources(readini_cos
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/constructor.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_cos)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_cos} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_cos STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_cos
#         PROPERTY IMPORTED_LOCATION ${readini_cos})
# ENDIF()
target_link_libraries(readini_cos
    PUBLIC
        cantera_interface)

# # # add readini_copy lib
# find_library(readini_copy NAMES libreadini_copy.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_copy}" STREQUAL "readini_copy-NOTFOUND")
    add_library(readini_copy "")
    target_sources(readini_copy
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpucopy.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_copy)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_copy} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_copy STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_copy
#         PROPERTY IMPORTED_LOCATION ${readini_copy})
# ENDIF()

# # # add readini_set lib
# find_library(readini_set NAMES libreadini_set.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_set}" STREQUAL "readini_set-NOTFOUND")
    add_library(readini_set "")
    target_sources(readini_set
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/iniset.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_set)
    ENDIF()

    target_link_libraries(readini_set
    PUBLIC
        ndAssign)

# ELSE()
#     message(STATUS "${readini_set} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_set STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_set
#         PROPERTY IMPORTED_LOCATION ${readini_set})
#     target_link_libraries(readini_set
#     INTERFACE
#         ndAssign)
# ENDIF()

# # # add readini_print lib
# find_library(readini_print NAMES libreadini_print.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_print}" STREQUAL "readini_print-NOTFOUND")
    add_library(readini_print "")
    target_sources(readini_print
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/print.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_print)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_print} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_print STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_print
#         PROPERTY IMPORTED_LOCATION ${readini_print})
# ENDIF()

# # # add readini_rea lib
# find_library(readini_rea NAMES libreadini_rea.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_rea}" STREQUAL "readini_rea-NOTFOUND")
    add_library(readini_rea "")
    target_sources(readini_rea
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/reaction.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_rea)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_rea} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_rea STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_rea
#         PROPERTY IMPORTED_LOCATION ${readini_rea})
# ENDIF()

# # # add readini_them lib
# find_library(readini_them NAMES libreadini_them.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_them}" STREQUAL "readini_them-NOTFOUND")
    add_library(readini_them "")
    target_sources(readini_them
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/thermal.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_them)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_them} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_them STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_them
#         PROPERTY IMPORTED_LOCATION ${readini_them})
# ENDIF()

# # # add readini_visc lib
# find_library(readini_visc NAMES libreadini_visc.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${readini_visc}" STREQUAL "readini_visc-NOTFOUND")
    add_library(readini_visc "")
    target_sources(readini_visc
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/viscfit.cpp
    )

    IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
        add_sycl_to_target(TARGET readini_visc)
    ENDIF()

# ELSE()
#     message(STATUS "${readini_visc} is located at ${LIBRARY_OUTPUT_PATH}")
#     add_library(readini_visc STATIC IMPORTED GLOBAL)
#     set_property(TARGET readini_visc
#         PROPERTY IMPORTED_LOCATION ${readini_visc})
# ENDIF()
