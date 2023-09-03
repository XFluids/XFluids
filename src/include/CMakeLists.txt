target_include_directories(cxxsrc
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(readini
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

IF(USE_MPI)
target_include_directories(mpiPacks
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()