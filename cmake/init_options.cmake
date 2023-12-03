set(THERMAL "NASA") # NASA or JANAF Thermal Fit
set(EIGEN_ALLOC "OROC") # Eigen memory allocate method used in FDM method

# # add cmake files
include(init_sample)

# # OROC: calculate one row and column once in registers "for" loop(eigen_lr[Emax])
# # RGIF: allocate eigen matrix in registers of kernel function(eigen_l[Emax][Emax], eigen_r[Emax][Emax]), which makes regesters spills out as Emax increases
# # AIGE: allocate eigen matrix in global memory (cudaMalloc(&eigen_l, Emax*Emax*Xmax*Ymax*Zmax*sizeof(real_t))) with low performance

# file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
# file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output/cal)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output/cal)

# // =======================================================
# #### about compile definitions
# // =======================================================
add_compile_options(-DSCHEME_ORDER=${WENO_ORDER})

IF(EIGEN_ALLOC STREQUAL "OROC")
  add_compile_options(-DEIGEN_ALLOC=0)
ELSEIF(EIGEN_ALLOC STREQUAL "RGIF")
  add_compile_options(-DEIGEN_ALLOC=1)
ELSEIF(EIGEN_ALLOC STREQUAL "AIGE")
  add_compile_options(-DEIGEN_ALLOC=2)
ENDIF()

IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

IF(ESTIM_NAN)
  add_compile_options(-DESTIM_NAN)

  IF(ERROR_PATCH_PRI)
    add_compile_options(-DERROR_PATCH)
  ENDIF()

  IF(ERROR_PATCH_YII)
    add_compile_options(-DERROR_PATCH_YII)
  ENDIF()

  IF(ERROR_PATCH_YI)
    add_compile_options(-DERROR_PATCH_YI)
  ENDIF()
ENDIF(ESTIM_NAN)

IF(POSITIVITY_PRESERVING)
  add_compile_options(-DPositivityPreserving)
ENDIF(POSITIVITY_PRESERVING)

# define Thermo 1				  // 1 for NASA and 0 for JANAF
IF(THERMAL STREQUAL "NASA")
  add_compile_options(-DThermo=1)
ELSEIF(THERMAL STREQUAL "JANAF")
  add_compile_options(-DThermo=0)
ENDIF()

# ROE for Roe_type, LLF for local Lax-Friedrichs eigen max, GLF for global Lax-Friedrichs eigen max of all points in Domain
IF(ARTIFICIAL_VISC_TYPE STREQUAL "ROE")
  add_compile_options(-DArtificial_type=1)
ELSEIF(ARTIFICIAL_VISC_TYPE STREQUAL "LLF")
  add_compile_options(-DArtificial_type=2)
ELSEIF(ARTIFICIAL_VISC_TYPE STREQUAL "GLF")
  add_compile_options(-DArtificial_type=3)
ENDIF()

IF(DIM_X)
  add_compile_options(-DDIM_X=1)
ELSE(DIM_X)
  add_compile_options(-DDIM_X=0)
ENDIF(DIM_X)

IF(DIM_Y)
  add_compile_options(-DDIM_Y=1)
ELSE(DIM_Y)
  add_compile_options(-DDIM_Y=0)
ENDIF(DIM_Y)

IF(DIM_Z)
  add_compile_options(-DDIM_Z=1)
ELSE(DIM_Z)
  add_compile_options(-DDIM_Z=0)
ENDIF(DIM_Z)

IF(Visc)
  add_compile_options(-DVisc)

  IF(Visc_Heat)
    add_compile_options(-DVisc_Heat)
  ENDIF(Visc_Heat)

  IF(Visc_Diffu)
    add_compile_options(-DVisc_Diffu)
  ENDIF(Visc_Diffu)
ENDIF(Visc)

# MPI libs
IF(USE_MPI)
  include(init_mpi)
  include_directories(AFTER ${CMAKE_SOURCE_DIR})
  add_subdirectory(mpiUtils)
ENDIF() # sources
