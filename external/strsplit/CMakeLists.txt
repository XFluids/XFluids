find_library(strsplit NAMES libstrsplit.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${strsplit}" STREQUAL "strsplit-NOTFOUND")
	add_library(strsplit "")
	target_sources(strsplit
		PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/strsplit.h
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/strsplit.cpp
	)

ELSE()
	message(STATUS "${strsplit} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(strsplit STATIC IMPORTED GLOBAL)
	set_property(TARGET strsplit
		PROPERTY IMPORTED_LOCATION ${strsplit})
ENDIF()
