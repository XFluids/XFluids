file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# // =======================================================
# #### about device select
# // =======================================================
IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
  set(Pform_id "1") # first device id in sycl-ls list #1 for host, 2 for GPU
ENDIF()

add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
add_compile_options(-DPform_id=${Pform_id}) # first device id in sycl-ls list

IF(SelectDv STREQUAL "host")
  include(init_host)
ELSEIF(SelectDv STREQUAL "nvidia")
  include(init_cuda)
  message(STATUS "  Compile for ARCH: ${ARCH}")
ELSEIF(SelectDv STREQUAL "amd")
  include(init_hip)
  message(STATUS "  Compile for ARCH: ${ARCH}")
ELSEIF(SelectDv STREQUAL "intel")
  include(init_intel)
  message(STATUS "  Compile for ARCH: ${ARCH}")
ENDIF(SelectDv STREQUAL "host")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
message(STATUS "CMAKE STATUS:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

include(init_sample)

IF(USE_MPI)
  include(init_mpi)
ENDIF(USE_MPI)

# // =======================================================
# #### about compile definitions
# // =======================================================
add_compile_options(-DNumFluid=1)
add_compile_options(-DMIDDLE_SYCL_ENABLED)
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

IF(OUT_PLT)
  add_compile_options(-DOUT_PLT)
ENDIF(OUT_PLT)

IF(OUT_VTI)
  add_compile_options(-DOUT_VTI)
ENDIF(OUT_VTI)

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

