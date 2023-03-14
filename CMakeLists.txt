cmake_minimum_required(VERSION 3.10)
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
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
endif()

option(USE_DOUBLE "if double" ON)
option(DIM_X "if enable dim_x" OFF)
option(DIM_Y "if enable dim_y" ON)
option(DIM_Z "if enable dim_z" OFF)
option(Multi "if enable multi , OFF for component" OFF)
option(COP "if enable compoent, based on Multi=OFF" ON)
option(React "if enable chemical recation, based on COP=ON" ON)
option(Visc "if add physic viscidity flux" OFF)
option(Heat "if add heat transfer flux" OFF)
option(Diffu "if add mass diffusion" OFF)
set(ReaType "2") # define which reaction to caculate, 0 for H2+O2

# about platform and device
set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
set(ARCH "sm_75") # cuda: sm_86, sm_75 # hip: gfx906, gfx1030
set(Pform_id "1") # first device id in sycl-ls list #1 for host, 2 for GPU
option(USE_MPI "if enable mpi" OFF)
option(AWARE_MPI "if enable aware-mpi" OFF)
set(num_GPUs "1") # number of mpi devices in sycl-ls list; # 1 for host, 2 for 3080Tix2

# cmake modules
include(init_options) # add main exec
add_subdirectory(src)