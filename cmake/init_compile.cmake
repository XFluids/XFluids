IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(SelectDv "omp") # define which platform and devices for compile options: host, nvidia, amd, intel
ENDIF()

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
	# // =======================================================
	set(AdaptiveCpp_DIR "/home/ljl/Apps/OpenSYCL/lib/cmake/OpenSYCL")
	find_package(AdaptiveCpp CONFIG REQUIRED)

	IF(SelectDv STREQUAL "cuda")
		set(ARCH "cc${ARCH}")
		set(SelectDv "cuda-nvcxx")
	ELSEIF(SelectDv STREQUAL "hip")
		set(ARCH "gfx${ARCH}")
	ENDIF()

	add_compile_options(-DDEFINED_OPENSYCL)

	# set(CMAKE_CXX_COMPILER "syclcc") # for OpenSYCL syclcc compiling system
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=set_but_not_used,declared_but_not_referenced,used_before_set")

	IF((${CMAKE_BUILD_TYPE} STREQUAL "Debug") OR(SelectDv STREQUAL "omp"))
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets='omp'") # get samples from syclcc --help
	ELSE()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-targets='${SelectDv}:${ARCH}'") # get samples from syclcc --help
	ENDIF()

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

# // =======================================================
# #### about device select
# // =======================================================
message(STATUS "CMAKE STATUS:")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
