IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
ENDIF()

# // =======================================================
# #### about external libs: boost
SET(EXTERNAL_BOOST_ROOT ${CMAKE_SOURCE_DIR}/external/install/boost)
SET(EXTERNAL_CANTERA_ROOT ${CMAKE_SOURCE_DIR}/external/install/cantera)
SET(EXTERNAL_ADAPTIVECPP_ROOT ${CMAKE_SOURCE_DIR}/external/install/AdaptiveCpp)
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
set(CONDA_PATH "$ENV{CONDA_PREFIX}")
IF(NOT CONDA_PATH)
	message(FATAL_ERROR "Compiling Package \"cantera\" error without conda environment, please activate conda environment")
ENDIF()
IF(NOT CANTERA_ROOT)
	set(CANTERA_ROOT "$ENV{CANTERA_ROOT}")
ENDIF()

find_library(cantera NAMES libcantera.so)
IF(("${cantera}" STREQUAL "cantera-NOTFOUND"))
	message(STATUS "linux cantera libs : ${CANTERA_ROOT}")
	find_library(cantera NAMES libcantera_shared.so HINTS "${CANTERA_ROOT}/lib" "${EXTERNAL_CANTERA_ROOT}/${CMAKE_BUILD_TYPE}/lib")
	IF(("${cantera}" STREQUAL "cantera-NOTFOUND"))
		message(STATUS "build cantera libs located: ${CANTERA_ROOT}")
		EXECUTE_PROCESS(COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_cantera.sh ${CONDA_PATH} ${CMAKE_BUILD_TYPE})
		set(CANTERA_ROOT ${EXTERNAL_CANTERA_ROOT}/${CMAKE_BUILD_TYPE})
		message(STATUS "build cantera libs located: ${CANTERA_ROOT}")
		find_library(cantera NAMES libcantera_shared.so HINTS "${CANTERA_ROOT}/lib")
		set(SUNDIALS_FOUND ON)
	ENDIF()
ELSE()
	set(CANTERA_ROOT "/usr")
	find_package(SUNDIALS)
	find_package(fmt)
ENDIF()
include_directories(
	"${CANTERA_ROOT}/include"
	"${CANTERA_ROOT}/include/cantera/ext")
# string(REGEX REPLACE "/lib/libcantera_shared.so" "/" CANTERA_ROOT "${cantera}")
message(STATUS "Find cantera headers located: ${CANTERA_ROOT}/include/cantera")
message(STATUS "Find cantera libs: ${cantera}")
if(SUNDIALS_FOUND)
	message(STATUS "Find Package \"sundials\": ${SUNDIALS_DIR}")
else()
	message(WARNING "May occur errors without sundials against with cantera")
endif()
if(fmt_FOUND)
	message(STATUS "Find Package \"fmt\": ${SUNDIALS_DIR}")
else()
	message(WARNING "May occur errors without fmt against with cantera")
endif()

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
	add_compile_options(-DDEFINED_OPENSYCL)
	IF(SelectDv STREQUAL "cuda-nvcxx")
		set(TARGET "cuda-nvcxx:cc${ARCH}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --diag_suppress=set_but_not_used,declared_but_not_referenced,used_before_set,code_is_unreachable,unsigned_compare_with_zero")
	ELSE()
		if(VENDOR_SUBMIT)
			add_compile_options(-D__VENDOR_SUBMIT__)
		endif()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed -Wno-format")
		IF((SelectDv STREQUAL "generic"))
			set(TARGET "generic")
		ELSEIF(SelectDv STREQUAL "cuda")
			set(TARGET "cuda:sm_${ARCH}")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-cuda-version")
		ELSEIF(SelectDv STREQUAL "hip")
			set(TARGET "hip:gfx${ARCH}")
		ELSEIF((SelectDv STREQUAL "omp") OR (SelectDv STREQUAL "host"))
			set(TARGET "omp")
		ENDIF()
	ENDIF()

	set(ENV{ACPP_TARGETS} "${TARGET}") 

	# // =======================================================
	# #### about external libs: ApdativeCpp
	IF(NOT AdaptiveCpp_DIR)
	set(AdaptiveCpp_DIR "$ENV{AdaptiveCpp_DIR}")
	ENDIF()
	find_package(AdaptiveCpp HINTS ${AdaptiveCpp_DIR} ${EXTERNAL_ADAPTIVECPP_ROOT}/lib/cmake/AdaptiveCpp)
	if(AdaptiveCpp_FOUND)
		message(STATUS "Find Package \"AdaptiveCpp\": ${AdaptiveCpp_DIR}")
	else()
		message(STATUS "NO Finding installed Package \"AdaptiveCpp\"")
		message(STATUS "Try Compiling external Package \"AdaptiveCpp\"")
		EXECUTE_PROCESS(COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/build_adaptivecpp.sh ${BOOST_ROOT} ${EXTERNAL_ADAPTIVECPP_ROOT})
		SET(AdaptiveCpp_DIR "${CMAKE_SOURCE_DIR}/external/install/AdaptiveCpp/lib/cmake/AdaptiveCpp")
		find_package(AdaptiveCpp HINTS ${EXTERNAL_ADAPTIVECPP_ROOT}/lib/cmake/AdaptiveCpp CONFIG REQUIRED)
		if(AdaptiveCpp_FOUND)
			message(STATUS "Find Package \"AdaptiveCpp\": ${AdaptiveCpp_DIR}")
		else()
			message(FATAL_ERROR "Compiling Package \"AdaptiveCpp\" error at: ${AdaptiveCpp_DIR}")
		endif()
	endif()

# // =======================================================
ELSEIF(SYCL_COMPILE_SYSTEM STREQUAL "oneAPI")
	# // =======================================================
	add_compile_options(-DDEFINED_ONEAPI)
	set(CMAKE_CXX_COMPILER "clang++") # for Intel oneAPI compiling system
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	IF((SelectDv STREQUAL "omp") OR(SelectDv STREQUAL "host"))
		set(SelectDv "host")
	ENDIF()

	include(oneAPIdevSelect/init_${SelectDv})
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
