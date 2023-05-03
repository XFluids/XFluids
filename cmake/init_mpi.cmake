add_compile_options(-DUSE_MPI)
# set(USE_PLT "OFF")

IF(EXPLICIT_ALLOC)
  add_compile_options(-DEXPLICIT_ALLOC)
ENDIF()

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