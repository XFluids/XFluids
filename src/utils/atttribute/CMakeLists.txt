# # add sample lib
find_library(attribute NAMES libattribute.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${attribute}" STREQUAL "attribute-NOTFOUND")
	add_library(attribute
		STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/attribute.cpp
	)
	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET attribute)
	ENDIF()

ELSE()
	message(STATUS "${attribute} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(attribute STATIC IMPORTED GLOBAL)
	set_property(TARGET attribute
		PROPERTY IMPORTED_LOCATION ${attribute})
ENDIF()
