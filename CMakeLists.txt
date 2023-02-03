cmake_minimum_required(VERSION 3.20)
PROJECT(EulerSYCL LANGUAGES CXX)
##default local cmake macro repository
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
##cmake features init
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "clang++")#use clang++ from oneapi/llvm
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
#some options in CMAKE && Kokkos need CACHE STRING FORCE set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

option(USE_DOUBLE "if double" ON)
option(DIM_X "if enable dim_x" ON)
option(DIM_Y "if enable dim_y" ON)
option(DIM_Z "if enable dim_z" OFF)
option(Multi "if enable multi , OFF for component" OFF)
option(COP "if enable compoent, based on Multi=OFF" ON)
option (React "if enable chemical recation, based on COP=ON" ON)
set(ReaType "0")#define which reaction to caculate, 0 for H2+O2
set (DeSelect "nvidia") # "define which platform and devices for compile options:host,nvidia,amd,intel,host" 
option(ENABLE_OPENMP "if need openmp" OFF)
option(Comp_Vis "if enable vis for component" OFF)
option(USE_MPI "if enable mpi" OFF)

##cmake modules
include(init_options)
# #add main exec
add_subdirectory(src)