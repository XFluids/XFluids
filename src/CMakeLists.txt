add_executable(${PROJECT_NAME} "main.cpp")

# # add sources of basic-solvers
add_library(cxxsrc
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Fluids.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LAMNSS.cpp
)
target_include_directories(cxxsrc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})

# # add basic components
add_subdirectory(read_ini)

IF(USE_MPI)
  add_subdirectory(mpiPacks)
ENDIF()

add_subdirectory(include)
add_subdirectory(Schemes)
add_subdirectory(solver_Ini)
add_subdirectory(solver_BCs)
add_subdirectory(solver_GetDt)
add_subdirectory(solver_Reconstruction)
add_subdirectory(solver_Viscosity)
add_subdirectory(solver_Reaction)
add_subdirectory(solver_UpdateStates)

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