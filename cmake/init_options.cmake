add_compile_options(-DNumFluid=1)

IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

IF(Visc)
  add_compile_options(-DVisc)
ENDIF(Visc)

IF(Heat)
  add_compile_options(-DHeat)
ENDIF(Heat)

IF(Diffu)
  add_compile_options(-DDiffu)
ENDIF(Diffu)

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

IF(COP)
  add_compile_options(-DCOP)
  add_compile_options(-DReact) # COP ON with React OFF may has error
  add_compile_options(-DReaType=${ReaType})

    IF(COP_CHEME)
    add_compile_options(-DCOP_CHEME)

    IF(${CHEME_SOLVER} MATCHES "Q2")
      add_compile_options(-DCHEME_SOLVER=0)
    ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
      add_compile_options(-DCHEME_SOLVER=1)
    ELSE()
    ENDIF()
  ENDIF(COP_CHEME)
    
  IF(Diffu)
    add_compile_options(-DDiffu)
  ENDIF(Diffu)
ENDIF(COP)

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

IF(SelectDv STREQUAL "host")
ENDIF(SelectDv STREQUAL "host")

IF(USE_MPI)
  include(init_mpi)
  message(STATUS "  Number of ${SelectDv} for MPI: ${num_GPUs}")
ENDIF(USE_MPI)
