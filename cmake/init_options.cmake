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

  IF(React)
    add_compile_options(-DReact)
    add_compile_options(-DReaType=${ReaType})
  ENDIF(React)

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
  message(STATUS "  Compile for ARCH: ${ARCH}")
  include(init_intel)
ENDIF(SelectDv STREQUAL "host")

IF(SelectDv STREQUAL "host")
ENDIF(SelectDv STREQUAL "host")

IF(USE_MPI)
  include(init_mpi)
  message(STATUS "  Number of GPUs for MPI: ${num_GPUs}")
ENDIF(USE_MPI)
