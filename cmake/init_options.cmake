message(STATUS "${COP_SAMPLE_PATH}")
include_directories(
  BEFORE
  "${COP_SAMPLE_PATH}"
) # 依据算例文件中的"case_setup.h"头文件自动设置NUM_SPECIES && NUM_REACTIONS

add_compile_options(-DNumFluid=1)

IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

IF(USE_PLT)
  add_compile_options(-DOUT_PLT) # 将参数从cmakelist传入程序中
ENDIF(USE_PLT)

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
ENDIF(Visc)

IF(Heat)
  add_compile_options(-DHeat)
ENDIF(Heat)

IF(Diffu)
  add_compile_options(-DDiffu)
ENDIF(Diffu)

add_compile_options(-DRPath="${COP_THERMAL_PATH}")
add_compile_options(-DRFile="${COP_SAMPLE_PATH}")

IF(COP)
  add_compile_options(-DCOP)

  IF(COP_CHEME)
    add_compile_options(-DCOP_CHEME)

    IF(${CHEME_SOLVER} MATCHES "Q2")
      add_compile_options(-DCHEME_SOLVER=0)
    ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
      add_compile_options(-DCHEME_SOLVER=1)
    ELSE()
    ENDIF()
  ENDIF(COP_CHEME)
ELSE(COP)
  add_compile_options(-DNUM_SPECIES=1)
  add_compile_options(-DNUM_REA=0)
ENDIF(COP)

IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(SelectDv "host") # define which platform and devices for compile options: host, nvidia, amd, intel
  set(Pform_id "1") # first device id in sycl-ls list #1 for host, 2 for GPU
ENDIF()

add_compile_options(-DSelectDv="${SelectDv}") # device, add marco as string-value
add_compile_options(-DPform_id=${Pform_id}) # first device id in sycl-ls list
add_compile_options(-Dnum_GPUs=${num_GPUs}) # number of mpi devices in sycl-ls list
message(STATUS "Compile settings: ")
message(STATUS "  Compile for platform: ${SelectDv}")

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

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
message(STATUS "CMAKE STATUS:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

IF(USE_MPI)
  include(init_mpi)
  message(STATUS "  Number of ${SelectDv} for MPI: ${num_GPUs}")
ENDIF(USE_MPI)
