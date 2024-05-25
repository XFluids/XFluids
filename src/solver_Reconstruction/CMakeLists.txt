add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${DISCRETIZATION_METHOD}_Method)

target_include_directories(${INIT_SAMPLE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})