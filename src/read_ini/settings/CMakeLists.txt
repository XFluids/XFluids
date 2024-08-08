add_library(settings "")
target_sources(settings
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/read_json.h
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/read_json.cpp
)

IF(SYCL_COMPILE_SYSTEM STREQUAL "ACPP")
	add_sycl_to_target(TARGET settings)
ENDIF()
