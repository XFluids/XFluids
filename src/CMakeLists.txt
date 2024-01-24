# # add external submodule
add_subdirectory(timer)
add_subdirectory(utils)
add_subdirectory(read_grid)
add_subdirectory(read_ini)
add_subdirectory(mpiPacks)
add_subdirectory(solver_Ini)
add_subdirectory(solver_BCs)
add_subdirectory(solver_UpdateStates)

# # add sample lib
find_library(${INIT_SAMPLE} NAMES lib${INIT_SAMPLE}.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${${INIT_SAMPLE}}" STREQUAL "${INIT_SAMPLE}-NOTFOUND")
  add_library(${INIT_SAMPLE}
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Fluids.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/XFLUIDS.cpp
  )

  # # add program sub-solvers
  add_subdirectory(solver_GetDt)
  add_subdirectory(solver_Reaction)
  add_subdirectory(solver_Reconstruction)
  target_include_directories(${INIT_SAMPLE}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/include
  )
  target_link_libraries(${INIT_SAMPLE}
    PUBLIC
    timer
    utils
    solver_Ini
    solver_BCs
    solver_Update
  )

  IF(USE_MPI)
    target_link_libraries(${INIT_SAMPLE}
      PUBLIC
      mpiPacks
    )
  ENDIF()

  IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET ${INIT_SAMPLE})
  ENDIF()

ELSE()
  message(STATUS "${${INIT_SAMPLE}} is located at ${LIBRARY_OUTPUT_PATH}")
  add_library(${INIT_SAMPLE} STATIC IMPORTED GLOBAL)
  set_property(TARGET ${INIT_SAMPLE}
    PROPERTY IMPORTED_LOCATION ${${INIT_SAMPLE}})
  target_link_libraries(${INIT_SAMPLE}
    INTERFACE
    timer
    utils
    solver_Ini
    solver_BCs
    solver_Update
  )
ENDIF()

# # add main implementation
add_executable(${PROJECT_NAME} "main.cpp")
target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/include
)

IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
  add_sycl_to_target(TARGET ${PROJECT_NAME})
ENDIF()

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  readini
  ${INIT_SAMPLE}
)

IF(USE_MPI)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    mpiPacks
  )
ENDIF()