add_executable(${PROJECT_NAME} "main.cpp")
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})

# # add sources of basic-solvers
add_library(cxxsrc
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Fluids.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LAMNSS.cpp
)

# # add basic components
add_subdirectory(read_ini)

IF(USE_MPI)
add_subdirectory(mpiPacks)
ENDIF()
add_subdirectory(include)
add_subdirectory(sycl_blocks)
add_subdirectory(sycl_kernels)
add_subdirectory(sycl_devices)

IF(USE_MPI)
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
    cxxsrc
    readini
    mpiPacks
  )
ELSE()
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
    cxxsrc
    readini
  )
ENDIF()