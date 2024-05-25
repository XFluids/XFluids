# // =======================================================
# #### cmake features init
# // =======================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

#set(CMAKE_AR "${CMAKE_CXX_COMPILER_AR}" CACHE STRING "Choose the path of CMAKE_AR." FORCE)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -save-temps=obj")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")
