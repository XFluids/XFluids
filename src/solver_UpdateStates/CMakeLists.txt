# # add sample lib
find_library(solver_Update NAMES libsolver_Update.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${solver_Update}" STREQUAL "solver_Update-NOTFOUND")
	add_library(solver_Update
		STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/UpdateStates_block.cpp
	)
	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET solver_Update)
	ENDIF()

ELSE()
	message(STATUS "${solver_Update} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(solver_Update STATIC IMPORTED GLOBAL)
	set_property(TARGET solver_Update
		PROPERTY IMPORTED_LOCATION ${solver_Update})
ENDIF()
