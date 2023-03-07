add_compile_options(-DUSE_MPI)

IF(AWARE_MPI)
  add_compile_options(-DAWARE_MPI)
ENDIF(AWARE_MPI)

include_directories($ENV{MPI_PATH}/include)
find_library(MPI_CXX NAMES libmpi.so HINTS "$ENV{MPI_PATH}/lib")
message(STATUS "MPI settings: ")
message(STATUS "  MPI_HOME: $ENV{MPI_PATH}")
message(STATUS "  MPI_INC: $ENV{MPI_PATH}/include added")
message(STATUS "  MPI_CXX lib located: ${MPI_CXX} found")
