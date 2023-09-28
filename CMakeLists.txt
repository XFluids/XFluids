cmake_minimum_required(VERSION 3.10)
PROJECT(LAMNSS LANGUAGES CXX)

# #default local cmake macro repository
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(init_cmake) # cmake compile options

# // =======================================================
# #### compile system:
# ####  1. oneAPI:       Intel oneAPI compiling system
# ####  2. OpenSYCL:     OpenSYCL syclcc compiling system
# // =======================================================
set(SYCL_COMPILE_SYSTEM "oneAPI") # # oneAPI or OpenSYCL
# // = about oneAPI platform and device ==================
set(SelectDv "cuda-nvcxx") # which platform for compile options: omp(for host), cuda-nvcxx(for nvidia), hip(for amd)
set(ARCH "75") # omp: no ARCH needed;# cuda/cuda-nvcxx: 75 86;# hip: 906, 1030

# // =======================================================
IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
  # // =======================================================
  IF(SelectDv MATCHES "cuda")
    set(ARCH "cc${ARCH}")
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

  # // =======================================================
ENDIF()

# // =======================================================
# #### function option
# // =======================================================
option(USE_DOUBLE "if double" ON)
option(DIM_X "if enable dim_x" ON)
option(DIM_Y "if enable dim_y" OFF)
option(DIM_Z "if enable dim_z" OFF)
option(COP "if enable compoent" ON)
option(Visc "if add physic viscidity flux" ON)
option(COP_CHEME "if enable chemical reaction" OFF)
option(ESTIM_NAN "estimate if rho is nan or <0 or inf" ON)
option(ERROR_PATCH_YI "if patch negative yi based Roe-Solver" OFF)
option(ERROR_PATCH_PRI "if patch rho/T/P errors while capured" OFF)
option(ERROR_PATCH_YII "if patch negative yi assigning a specific value" ON)
option(POSITIVITY_PRESERVING "if enable positive preserving based Dr.Hu" ON)
option(COP_CHEME_TEST "if enable 0D reaction solver test" OFF)
option(Visc_Heat "if add heat transfer flux" ON)
option(Visc_Diffu "if add mass diffusion" ON) # depends on COP=ON

# // =======================================================
# #### mpi settings
# // =======================================================
# #### EXPLICIT_ALLOC: Method of mpi buffer allocation and transfer:
# # ON: allocate each device buffer and transfer
# # OFF: allocate struct ptr in device
option(USE_MPI "if enable mpi" ON)
option(AWARE_MPI "if enable aware-mpi" OFF) # only be valid while EXPLICIT_ALLOC is set "ON"
option(EXPLICIT_ALLOC "if enable explict mpi buffer allocate" ON)

# // =======================================================
# #### init sample
# // =======================================================
set(THERMAL "JANAF") # NASA or JANAF fit
set(WENO_ORDER "6") # 5, 6 or 7: WENO5, WENOCU6 or WENO7 for FluxWall reconstruction
set(EIGEN_ALLOC "OROC") # #OROC# #RGIF# #AIGE
set(ARTIFICIAL_VISC_TYPE "GLF") # ROE for Roe_type, LLF for local Lax-Friedrichs eigen max, GLF for global Lax-Friedrichs eigen max of all points in Domain
set(INIT_SAMPLE "2d-shock-bubble")
set(COP_SPECIES "Insert-SBI") # name of the folder under ./runtime.dat/
set(REACTION_MODEL "RSBI-19REA") # H2O_19_reaction # RSBI-18REA # Be valid depending on COP_CHEME "ON", will rewrite ${COP_SPECIES} in /cmake/init_options.cmake
set(CHEME_SOLVER "Q2") # "Q2", only Q2 now, depends on COP_CHEME "ON"
set(CHEME_SPLITTING "Strang") # "Strang": React(dt/2)->Flow(dt)->React(dt/2); "Lie": Flow(dt)->React(dt)

# #### Set INIT_SAMPLE for method of eigen_l, eigen_r allocation:
# # OROC: calculate one row and column once in for loop(eigen_lr[Emax])
# # RGIF: allocate eigen matrix in regesters of kernel function(eigen_l[Emax][Emax], eigen_r[Emax][Emax]),which makes regesters spills out as Emax increases
# # AIGE: allocate eigen matrix in global memory (cudaMalloc(&eigen_l,Emax*Emax*Xmax*Ymax*Zmax*sizeof(real_t)))
# #### Set INIT_SAMPLE for Initializing settings:
# # # for-debug, guass-wave, sharp-interface ## util for 1D, 2D, 3D
# #  0d-H2-O2-freely-flame
# # # 1d samples are forced out with plt
# # 1d-insert-st ## automatically rewrite COP_CHEME, Visc to "OFF"
# # 1d-reactive-st ## automatically rewrite COP_CHEME to "ON", Visc to "OFF", path to "reaction_list.dat" depends on REACTION_MODEL by default: H2O_19_reaction
# # 1d-diffusion 1d-diffusion-reverse ## automatically rewrite Visc, Visc_Heat, Visc_Diffu to "ON", COP_CHEME to "ON"
# # 2d-shock-bubble, 2d-shock-bubble-without-fuel, 2d-mixing-layer, 2d-under-expanded-jet ## only XY DIR support
# # 3d-shock-bubble, 3d-shock-bubble-without-fuel, 3d-under-expanded-jet
# #### Set COP_SPECIES for location of species_list.dat read in project
# # /runtime.dat/1d-mc-insert-shock-tube  # # /runtime.dat/2d-under-expanded-jet
# # /runtime.dat/Reaction/2species        # # /runtime.dat/Reaction/CJ_mechanism  # # /runtime.dat/Reaction/inert_2species
# #### Set REACTION_MODEL for reaction model using:
# # 2species             # # CJ_mechanism     # # inert_2species
# # H2O_18_new_reaction  # # H2O_18_reaction  # # H2O_19_reaction     # # H2O_23_reaction
# # RSBI-18REA           # # RSBI-19REA

# cmake modules
include(init_options) # add main exec

# MPI libs
IF(USE_MPI)
  include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(mpiUtils)
ENDIF() # sources

add_subdirectory(src)