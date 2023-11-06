IF(USE_MPI)
	find_library(mpiPacks NAMES libmpiPacks.a HINTS "${LIBRARY_OUTPUT_PATH}")

	IF("${mpiPacks}" STREQUAL "mpiPacks-NOTFOUND")
		add_library(mpiPacks "")
		target_include_directories(mpiPacks
			PUBLIC
			${CMAKE_SOURCE_DIR}/src/include)
		target_sources(mpiPacks
			PRIVATE
			mpiPacks.cpp
		)

		target_link_libraries(mpiPacks
			PUBLIC
			mpiUtils
		)

		IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
			add_sycl_to_target(TARGET mpiPacks)
		ENDIF()
	ELSE()
		message(STATUS "${mpiPacks} is located at ${LIBRARY_OUTPUT_PATH}")
		add_library(mpiPacks STATIC IMPORTED GLOBAL)
		set_property(TARGET mpiPacks
			PROPERTY IMPORTED_LOCATION ${mpiPacks})
		target_link_libraries(mpiPacks
			INTERFACE
			mpiUtils
		)
	ENDIF()
ENDIF()
