include_directories("${CMAKE_SOURCE_DIR}/src/solver_Ini")

# find_library(cantera_interface NAMES libcantera_interface.a HINTS "${LIBRARY_OUTPUT_PATH}")
# IF("${cantera_interface}" STREQUAL "cantera_interface-NOTFOUND")
	add_library(cantera_interface "")
	target_sources(cantera_interface
		PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/cantera_interface.h
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/cantera_interface.cpp
	)

# ELSE()
# 	message(STATUS "${cantera_interface} is located at ${LIBRARY_OUTPUT_PATH}")
# 	add_library(cantera_interface STATIC IMPORTED GLOBAL)
# 	set_property(TARGET cantera_interface
# 		PROPERTY IMPORTED_LOCATION ${cantera_interface})
# ENDIF()
target_link_libraries(cantera_interface 
	INTERFACE 
		${cantera})

# find_library(xthermal NAMES libxthermal.a HINTS "${LIBRARY_OUTPUT_PATH}")
# IF("${xthermal}" STREQUAL "xthermal-NOTFOUND")
	add_library(xthermal "")
	target_sources(xthermal
		PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/XThermal.h
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/XThermal.cpp
	)

	IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
		add_sycl_to_target(TARGET xthermal)
	ENDIF()

# ELSE()
# 	message(STATUS "${xthermal} is located at ${LIBRARY_OUTPUT_PATH}")
# 	add_library(xthermal STATIC IMPORTED GLOBAL)
# 	set_property(TARGET xthermal
# 		PROPERTY IMPORTED_LOCATION ${xthermal})
# ENDIF()
target_link_libraries(cantera_interface 
	INTERFACE 
		xthermal)
