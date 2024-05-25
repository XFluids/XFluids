add_compile_options(-DUSE_MPI)

include_directories(
  "${CMAKE_SOURCE_DIR}"
  "$ENV{MPI_PATH}/include")
find_library(MPI_CXX NAMES libmpi.so libmpicxx.so HINTS "$ENV{MPI_PATH}/lib" "$ENV{MPI_PATH}/lib64")

IF(NOT "$ENV{MPI_PATH}" STREQUAL "")

message(STATUS "MPI settings: ")

IF(EXPLICIT_ALLOC)
  add_compile_options(-DEXPLICIT_ALLOC)
  IF(AWARE_MPI)
      add_compile_options(-DAWARE_MPI)
    ENDIF(AWARE_MPI)
  message(STATUS "  AWARE_MPI: ${AWARE_MPI}")
  message(STATUS "  MPI buffer allocate method: explicit")
ELSE()
  message(STATUS "  MPI buffer allocate method: implicit")
ENDIF()

message(STATUS "  MPI_HOME: $ENV{MPI_PATH}")
message(STATUS "  MPI_INC: $ENV{MPI_PATH}/include added")
message(STATUS "  MPI_CXX lib located: ${MPI_CXX} found")

ELSE()
  message(FATAL_ERROR "Fail to find MPI_CXX library, Please set SYSTEM environment variable MPI_PATH")
ENDIF()
