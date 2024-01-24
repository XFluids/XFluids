find_library(timer NAMES libtimer.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${timer}" STREQUAL "timer-NOTFOUND")
	add_library(timer "")
	target_sources(timer
		PRIVATE
		timer.cpp
	)

	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET timer)
	ENDIF()
ELSE()
	message(STATUS "${timer} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(timer STATIC IMPORTED GLOBAL)
	set_property(TARGET timer
		PROPERTY IMPORTED_LOCATION ${timer})
ENDIF()
