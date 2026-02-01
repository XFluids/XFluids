IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
ENDIF()

# // =======================================================
# #### about external libs: boost
SET(EXTERNAL_BOOST_ROOT ${CMAKE_SOURCE_DIR}/external/install/boost)
#SET(EXTERNAL_CANTERA_ROOT ${CMAKE_SOURCE_DIR}/external/install/cantera)
# // =======================================================
IF(NOT BOOST_ROOT)
	set(BOOST_ROOT "$ENV{BOOST_ROOT}")
ENDIF()
find_library(boost_fiber NAMES libboost_fiber.so HINTS "${BOOST_ROOT}/lib" "${EXTERNAL_BOOST_ROOT}/lib")
find_library(boost_context NAMES libboost_context.so HINTS "${BOOST_ROOT}/lib" "${EXTERNAL_BOOST_ROOT}/lib")
find_library(boost_filesystem NAMES libboost_filesystem.so HINTS "${BOOST_ROOT}/lib" "${EXTERNAL_BOOST_ROOT}/lib")
IF(("${boost_fiber}" STREQUAL "boost_fiber-NOTFOUND") OR
	("${boost_context}" STREQUAL "boost_context-NOTFOUND") OR
	("${boost_filesystem}" STREQUAL "boost_filesystem-NOTFOUND"))
	EXECUTE_PROCESS(COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_boost.sh ${EXTERNAL_BOOST_ROOT})
	set(BOOST_ROOT ${EXTERNAL_BOOST_ROOT})
	message(STATUS "build boost libs located: ${BOOST_ROOT}")
	find_library(boost_fiber NAMES libboost_fiber.so HINTS "${BOOST_ROOT}/lib")
	find_library(boost_context NAMES libboost_context.so HINTS "${BOOST_ROOT}/lib")
	find_library(boost_filesystem NAMES libboost_filesystem.so HINTS "${BOOST_ROOT}/lib")
ELSE()
	string(REGEX REPLACE "/lib/libboost_fiber.so" "/" BOOST_ROOT "${boost_fiber}")
	message(STATUS "Find boost libs located: ${BOOST_ROOT}")
ENDIF()
	set(BOOST_CXX "ON") # use external boost
	
# // =======================================================
# Modified by gpi:
# Force the script to use the CANTERA_ROOT logic by pre-setting the result to NOTFOUND
#set(cantera "cantera-NOTFOUND" CACHE FILEPATH "Force cantera lookup to use CANTERA_ROOT")
# find_library(cantera NAMES libcantera.so) # <-- Disabled
#IF(("${cantera}" STREQUAL "cantera-NOTFOUND"))
#	IF(NOT CANTERA_ROOT)
#		set(CANTERA_ROOT "$ENV{CANTERA_ROOT}")
#	ENDIF()
#	find_library(cantera NAMES libcantera_shared.so HINTS "${CANTERA_ROOT}/lib" "${EXTERNAL_CANTERA_ROOT}/${CMAKE_BUILD_TYPE}/lib")
#	IF(("${cantera}" STREQUAL "cantera-NOTFOUND"))
#		set(CONDA_PATH "$ENV{CONDA_PREFIX}")
#		IF(NOT CONDA_PATH)
#			message(FATAL_ERROR "Compiling Package \"cantera\" error without conda environment, please activate conda environment")
#		ENDIF()
#		message(STATUS "build cantera libs located: ${CANTERA_ROOT}")
#		EXECUTE_PROCESS(COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_cantera.sh ${CONDA_PATH} ${CMAKE_BUILD_TYPE})
#		set(CANTERA_ROOT ${EXTERNAL_CANTERA_ROOT}/${CMAKE_BUILD_TYPE})
#		message(STATUS "build cantera libs located: ${CANTERA_ROOT}")
#		find_library(cantera NAMES libcantera_shared.so HINTS "${CANTERA_ROOT}/lib")
#		set(SUNDIALS_FOUND ON)
#	ENDIF()
#ELSE()
#	set(CANTERA_ROOT "/usr")
#	find_package(SUNDIALS)
# remove 	find_package(fmt)
#ENDIF()
# add 	find_package(fmt)
#find_package(fmt)

#include_directories(
#	"${CANTERA_ROOT}/include"
#	"${CANTERA_ROOT}/include/cantera/ext")
# string(REGEX REPLACE "/lib/libcantera_shared.so" "/" CANTERA_ROOT "${cantera}")
#message(STATUS "Find cantera headers located: ${CANTERA_ROOT}/include/cantera")
#message(STATUS "Find cantera libs: ${cantera}")
#if(SUNDIALS_FOUND)
#	message(STATUS "Find Package \"sundials\": ${SUNDIALS_DIR}")
#else()
#	message(WARNING "May occur errors without sundials against with cantera")
#endif()
#if(fmt_FOUND)
#	message(STATUS "Find Package \"fmt\": ${SUNDIALS_DIR}")
#else()
#	message(WARNING "May occur errors without fmt against with cantera")
#endif()

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
	IF(SelectDv STREQUAL "cuda-nvcxx")
		set(TARGET "cuda-nvcxx:cc${ARCH}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=set_but_not_used,declared_but_not_referenced,used_before_set,code_is_unreachable,unsigned_compare_with_zero")
	ELSE()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed -Wno-format")
		IF((SelectDv STREQUAL "generic"))
			set(TARGET "generic")
		ELSEIF(SelectDv STREQUAL "cuda")
			set(ARCH "sm_${ARCH}")
			set(TARGET "cuda:${ARCH}")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-cuda-version")
			if(VENDOR_SUBMIT)
				add_compile_options(-D__VENDOR_SUBMIT__)
			endif()
		ELSEIF(SelectDv STREQUAL "hip")
			set(ARCH "gfx${ARCH}")
			set(TARGET "hip:${ARCH}")
			if(VENDOR_SUBMIT)
				add_compile_options(-D__VENDOR_SUBMIT__)
			endif()
		ELSEIF((SelectDv STREQUAL "omp") OR (SelectDv STREQUAL "host"))
			set(TARGET "generic")
			set(SelectDv "generic")
		ENDIF()
	ENDIF()

	set(ENV{ACPP_TARGETS} "${TARGET}") 

	# // =======================================================
	# #### about external libs: ApdativeCpp
	IF(NOT ACPP_PATH)
		set(ACPP_PATH "$ENV{ACPP_PATH}")
	ENDIF()
	SET(EXTERNAL_ADAPTIVECPP_ROOT ${CMAKE_SOURCE_DIR}/external/install/AdaptiveCpp_${SelectDv})
	IF(NOT COMPILER_PATH)
		set(COMPILER_PATH "$ENV{COMPILER_PATH}")
	ENDIF()
	find_package(AdaptiveCpp HINTS ${ACPP_PATH}/lib/cmake/AdaptiveCpp ${EXTERNAL_ADAPTIVECPP_ROOT}/lib/cmake/AdaptiveCpp)
	if(AdaptiveCpp_FOUND)
		message(STATUS "Find Installed Package \"AdaptiveCpp\"")
	else()
		message(STATUS "NO Finding installed Package \"AdaptiveCpp\"")
		message(STATUS "Try Compiling external Package \"AdaptiveCpp\"")
		EXECUTE_PROCESS(
			COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_adaptivecpp.sh ${BOOST_ROOT} ${EXTERNAL_ADAPTIVECPP_ROOT} ${COMPILER_PATH} ${SelectDv} ${ARCH}
			RESULT_VARIABLE _RE)
		IF(_RE)
			message(FATAL_ERROR "Compiling Package \"AdaptiveCpp\" error, please check environment \"COMPILER_PATH\" or other settings")
		ENDIF()
		SET(ACPP_PATH "${EXTERNAL_ADAPTIVECPP_ROOT}")
		find_package(AdaptiveCpp HINTS ${ACPP_PATH}/lib/cmake/AdaptiveCpp CONFIG REQUIRED)
		if(AdaptiveCpp_FOUND)
			message(STATUS "Find Package \"AdaptiveCpp\": ${ACPP_PATH}")
		else()
			message(FATAL_ERROR "Compiling Package \"AdaptiveCpp\" error at: ${ACPP_PATH}")
		endif()
	endif()

# // =======================================================
ELSEIF(SYCL_COMPILE_SYSTEM STREQUAL "oneAPI")
	# // =======================================================
	add_compile_options(-DDEFINED_ONEAPI)
    set(BOOST_CXX "OFF") # use boost c++ library or std internal library
	set(CMAKE_CXX_COMPILER "clang++") #clang++ # for Intel oneAPI compiling system
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    add_compile_options(-DFMT_LOCALE=0)
    
    if(USE_REVERSE_NDRANGE)
        add_compile_definitions(__REVERSE_NDRANGE__)
        message(STATUS "Build with Reverse nd_range (Z-Y-X) enabled.")
    else()
        message(STATUS "Build with Standard nd_range (X-Y-Z) enabled.")
    endif()

    if(USE_MPI_TIMER)
        add_compile_definitions(USE_MPI_TIMER)
        message(STATUS "Build with MPI Timer enabled.")
    else()
        message(STATUS "Build with Chrono Timer enabled.")
    endif()

    IF(ENABLE_HYBRID)
        add_compile_options(-DHYBRID_CALC)
        add_compile_options(-DUSE_MPI)
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda,spir64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvidia_gpu_sm_86,x86_64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-cuda-version")

        # delete warnings about VLA extension and disable fmt locale support
        add_compile_options(-Wno-vla-extension)

        if(NOT TEST_CASE STREQUAL "0")
            add_compile_options(-DTEST_CASE=${TEST_CASE})
        endif()
    ELSE()
  	    IF((SelectDv STREQUAL "omp") OR(SelectDv STREQUAL "host"))
	    	set(SelectDv "host")
	    ENDIF()

        include(oneAPIdevSelect/init_${SelectDv})
    ENDIF()
ENDIF()

# // =======================================================
# #### about device select
# // =======================================================
add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
message(STATUS "CMAKE STATUS:")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
