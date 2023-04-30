add_compile_options(-DUSE_MPI)

include_directories($ENV{MPI_PATH}/include)
find_library(MPI_CXX NAMES libmpi.so HINTS "$ENV{MPI_PATH}/lib")
message(STATUS "MPI settings: ")

IF(AWARE_MPI)
  add_compile_options(-DAWARE_MPI)
  message(STATUS "  AWARE_MPI: ON")
ELSE()
  message(STATUS "  AWARE_MPI: OFF")
ENDIF(AWARE_MPI)

message(STATUS "  MPI_HOME: $ENV{MPI_PATH}")
message(STATUS "  MPI_INC: $ENV{MPI_PATH}/include added")
message(STATUS "  MPI_CXX lib located: ${MPI_CXX} found")
message(STATUS "  Number of ${SelectDv} for MPI: ${num_GPUs}")
