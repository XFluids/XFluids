# add_library(solver_viscosity "")
# target_sources(solver_viscosity
# PRIVATE
# PUBLIC
# ${CMAKE_CURRENT_SOURCE_DIR}/Util_block.hpp
# ${CMAKE_CURRENT_SOURCE_DIR}/Visc_block.hpp
# ${CMAKE_CURRENT_SOURCE_DIR}/Visc_device.hpp
# ${CMAKE_CURRENT_SOURCE_DIR}/Visc_kernels.hpp)

target_include_directories(${INIT_SAMPLE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${INIT_SAMPLE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/${VISCOSITY_ORDER}_Order)
