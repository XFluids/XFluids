# find_library(inishape NAMES libinishape.a HINTS "${LIBRARY_OUTPUT_PATH}")

# IF("${inishape}" STREQUAL "inishape-NOTFOUND")
	add_library(inishape "")
	target_sources(inishape
		PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/inishape.h
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/inishape.cpp
	)

	IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
		add_sycl_to_target(TARGET inishape)
	ENDIF()

# ELSE()
# 	message(STATUS "${inishape} is located at ${LIBRARY_OUTPUT_PATH}")
# 	add_library(inishape STATIC IMPORTED GLOBAL)
# 	set_property(TARGET inishape
# 		PROPERTY IMPORTED_LOCATION ${inishape})
# ENDIF()
