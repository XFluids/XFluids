IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(SelectDv "omp") # define which platform and devices for compile options: host, nvidia, amd, intel
ENDIF()

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
	# // =======================================================
	IF(SelectDv STREQUAL "cuda")
		set(ARCH "cc${ARCH}")
		set(SelectDv "cuda-nvcxx")
	ELSEIF(SelectDv STREQUAL "hip")
		set(ARCH "gfx${ARCH}")
	ENDIF()

	add_compile_options(-DDEFINED_OPENSYCL)
	set(CMAKE_CXX_COMPILER "syclcc") # for OpenSYCL syclcc compiling system
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=set_but_not_used,declared_but_not_referenced,used_before_set --opensycl-targets='${SelectDv}:${ARCH}'") # get samples from syclcc --help

# // =======================================================
ELSEIF(SYCL_COMPILE_SYSTEM STREQUAL "oneAPI")
	# // =======================================================
	add_compile_options(-DDEFINED_ONEAPI)
	set(CMAKE_CXX_COMPILER "clang++") # for Intel oneAPI compiling system
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	IF(SelectDv STREQUAL "omp")
		set(SelectDv "host")
	ENDIF()

	include(oneAPIdevSelect/init_${SelectDv})
	message(STATUS "  Compile for ARCH: ${ARCH}")
ENDIF()

add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
