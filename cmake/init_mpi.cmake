##MPI inc
IF(USE_MPI)
  include_directories($ENV{MPI_PATH}/include)
  add_compile_options(-DUSE_MPI)
  IF(AWARE_MPI)
  add_compile_options(-DAWARE_MPI)
  ENDIF(AWARE_MPI)
ENDIF(USE_MPI)
