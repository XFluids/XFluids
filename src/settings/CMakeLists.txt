find_library(settings NAMES libsettings.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${settings}" STREQUAL "settings-NOTFOUND")
	add_library(settings "")
	target_sources(settings
		PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/read_json.h
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/read_json.cpp
	)
	target_include_directories(settings
		PUBLIC
		${CMAKE_SOURCE_DIR}/src/include
	)

	IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
		add_sycl_to_target(TARGET settings)
	ENDIF()
ELSE()
	message(STATUS "${settings} is located at ${LIBRARY_OUTPUT_PATH}")
	add_library(settings STATIC IMPORTED GLOBAL)
	set_property(TARGET settings
		PROPERTY IMPORTED_LOCATION ${settings})
ENDIF()