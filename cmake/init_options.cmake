set(THERMAL "NASA") # NASA or JANAF Thermal Fit
set(EIGEN_ALLOC "OROC") # Eigen memory allocate method used in FDM method
# # OROC: calculate one row and column once in registers "for" loop(eigen_lr[Emax])
# # RGIF: allocate eigen matrix in registers of kernel function(eigen_l[Emax][Emax], eigen_r[Emax][Emax]), which makes regesters spills out as Emax increases
option(COP "if enable compoent" ON)
option(EXPLICIT_ALLOC "if enable explict mpi buffer allocate" ON) # ON: allocate device buffer and transfer. OFF: allocate struct ptr on host
option(ESTIM_NAN "estimate if primitive variable(rho,yi,P,T) is nan or <0 or inf." ON)
option(ERROR_OUT "if out intermediate variables for Flux ((b1,b3,zi)[convention],Di[visc],...)." OFF)
option(ERROR_PATCH_PRI "if patch primitive varibales using Roe average method, destruct physic fluid flow." OFF)

# # add cmake files
include(init_compile) # SYCL compile system
include(init_sample)

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
ENDIF()

IF(ASYNC_SUBMIT)
  add_compile_options(-D__SYNC_TIMER_=0)
ELSE()
  add_compile_options(-D__SYNC_TIMER_=1)
ENDIF()

IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

IF(ESTIM_NAN)
  add_compile_options(-DESTIM_NAN=1)

  IF(ERROR_OUT)
    add_compile_options(-DESTIM_OUT=1)
  ELSE()
    add_compile_options(-DESTIM_OUT=0)
  ENDIF(ERROR_OUT)

  IF(ERROR_PATCH_PRI)
    add_compile_options(-DERROR_PATCH)
  ENDIF()

  IF(ERROR_PATCH_YI)
    add_compile_options(-DERROR_PATCH_YI)
  ENDIF()

  IF(ERROR_PATCH_YII)
    add_compile_options(-DERROR_PATCH_YII)
  ENDIF()
ELSE()
  add_compile_options(-DESTIM_NAN=0)
ENDIF(ESTIM_NAN)

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

IF(Visc)
  add_compile_options(-DVisc=1)

  IF(Visc_Heat)
    add_compile_options(-DVisc_Heat=1)
  ENDIF(Visc_Heat)

  IF(Visc_Diffu)
    add_compile_options(-DVisc_Diffu=1)
  ENDIF(Visc_Diffu)
ENDIF(Visc)

# MPI libs
IF(USE_MPI)
  include(init_mpi)
  add_subdirectory(mpiUtils)
ENDIF() # sources
