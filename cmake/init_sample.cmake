message(STATUS "Sample init include settings: ")

IF(INI_SAMPLE STREQUAL "for-debug")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/for-debug")
ELSEIF(INI_SAMPLE STREQUAL "1d-guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/guass-wave")
ELSEIF(INI_SAMPLE STREQUAL "1d-insert-st")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ELSEIF(INI_SAMPLE STREQUAL "1d-reactive-st")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/reactive-st")
ELSEIF(INI_SAMPLE STREQUAL "2d-guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ELSEIF(INI_SAMPLE STREQUAL "2d-shock-bubble")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ELSEIF(INI_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ELSEIF(INI_SAMPLE STREQUAL "3d-shock-bubble")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ELSEIF(INI_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
ENDIF()

message(STATUS "  Sample init sample path: ${INI_SAMPLE_PATH}")
message(STATUS "  Sample COP  header path: ${COP_SAMPLE_PATH}")

add_compile_options(-DRPath="${COP_THERMAL_PATH}")
add_compile_options(-DRFile="${COP_SAMPLE_PATH}")

include_directories(
    BEFORE
    "${COP_SAMPLE_PATH}"
    "${INI_SAMPLE_PATH}"
)
# ${COP_SAMPLE_PATH}: 依据算例文件中的"case_setup.h"头文件自动设置NUM_SPECIES && NUM_REACTIONS #
# ${INI_SAMPLE_PATH}: 依据sample文件夹中的"ini_sample.hpp"文件选择 #