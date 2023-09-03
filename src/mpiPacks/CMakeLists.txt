target_include_directories(cxxsrc
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(readini
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

add_library(mpiPacks "")
target_sources(mpiPacks
	PRIVATE
	mpiPacks.cpp
)
target_link_libraries(mpiPacks
	PUBLIC
	mpiUtils
)