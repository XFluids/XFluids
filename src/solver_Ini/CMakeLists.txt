# # add sample lib
find_library(solver_Ini NAMES libsolver_Ini.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${solver_Ini}" STREQUAL "solver_Ini-NOTFOUND")
	add_library(solver_Ini
		STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/Ini_block.cpp
	)
	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET solver_Ini)
	ENDIF()

ELSE()
	message(STATUS "${solver_Ini} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(solver_Ini STATIC IMPORTED GLOBAL)
	set_property(TARGET solver_Ini
		PROPERTY IMPORTED_LOCATION ${solver_Ini})
ENDIF()
