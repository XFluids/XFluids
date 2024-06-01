# # add sample lib
find_library(solver_BCs NAMES libsolver_BCs.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${solver_BCs}" STREQUAL "solver_BCs-NOTFOUND")
	add_library(solver_BCs
		STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/BCs_block.cpp
	)

	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET solver_BCs)
	ENDIF()

ELSE()
	message(STATUS "${solver_BCs} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(solver_BCs STATIC IMPORTED GLOBAL)
	set_property(TARGET solver_BCs
		PROPERTY IMPORTED_LOCATION ${solver_BCs})
ENDIF()