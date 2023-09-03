include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})
add_library(mpiUtils "")

target_sources(mpiUtils
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/TestForException.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiEnums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.h
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.h
) 
# add_library(mpiUtils
# STATIC
# error.h
# mpiEnums.h
# MpiTransComm.h
# MpiTransComm.cpp
# )
target_link_libraries(mpiUtils
  PUBLIC
  ${MPI_CXX}
)