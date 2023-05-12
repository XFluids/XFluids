math(EXPR DIM_NUM "0" OUTPUT_FORMAT DECIMAL)

IF(DIM_X)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_X)

IF(DIM_Y)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_Y)

IF(DIM_Z)
    math(EXPR DIM_NUM "${DIM_NUM} + 1" OUTPUT_FORMAT DECIMAL)
ENDIF(DIM_Z)

IF(USE_MPI)
    set(APPEND "-mpi.ini")
ELSE()
    set(APPEND ".ini")
ENDIF()

message(STATUS "Sample init include settings: ")

# message(STATUS "DIM_NUM: ${DIM_NUM}")
set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/1d-mc-insert-shock-tube") # Be invalid while option COP_CHEME "ON"

# // =======================================================
# #### util sample
# // =======================================================
IF(INIT_SAMPLE STREQUAL "for-debug")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/for-debug")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-debug${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "guass-wave")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/guass-wave")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-guass-wave${APPEND}")

# // =======================================================
# #### 1d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "1d-insert-st")
    IF(DIM_NUM STREQUAL "1")
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/insert-st")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-1d-shock-tube${APPEND}")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-reactive-st")
    IF(DIM_NUM STREQUAL "1")
        set(COP_CHEME "ON")
        set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/1D-X-Y-Z/reactive-st")
        set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-1d-reactive-st${APPEND}")
    ELSEIF()
        message(FATAL_ERROR: "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

# // =======================================================
# #### 2d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "2d-shock-bubble")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(THERMAL "NASA") # NASA fit of Xe
    message(STATUS "  Only NASA fit for Xe used in RSBI sample.")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/shock-bubble-intera")
    set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Reaction/RSBI-18REA")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-expanded-jet${APPEND}")

# // =======================================================
# #### 3d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "3d-shock-bubble")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(THERMAL "NASA") # NASA fit of Xe
    message(STATUS "  Only NASA fit for Xe used in RSBI sample.")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/shock-bubble-intera")
    set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Reaction/RSBI-18REA")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-shock-bubble${APPEND}")

ELSEIF(INIT_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(INI_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/src/sample/under-expanded-jet")
    set(INI_FILE "${CMAKE_SOURCE_DIR}/sa-expanded-jet${APPEND}")
ELSE()
    message(FATAL_ERROR "ini sample isn't given.")
ENDIF()

IF(COP)
    add_compile_options(-DCOP)

    IF(COP_CHEME)
        add_compile_options(-DCOP_CHEME)
        set(COP_SPECIES "${CMAKE_SOURCE_DIR}/runtime.dat/Reaction/${REACTION_MODEL}") # where to read species including reactions

        IF(${CHEME_SOLVER} MATCHES "Q2")
            add_compile_options(-DCHEME_SOLVER=0)
        ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
            add_compile_options(-DCHEME_SOLVER=1)
        ELSE()
        ENDIF()
    ENDIF(COP_CHEME)
ELSE(COP)
    add_compile_options(-DNUM_SPECIES=1)
    add_compile_options(-DNUM_REA=0)
ENDIF(COP)

add_compile_options(-DINI_SAMPLE="${INIT_SAMPLE}")
message(STATUS "  Sample select: ${INIT_SAMPLE}")
message(STATUS "  Sample init sample path: ${INI_SAMPLE_PATH}")
message(STATUS "  Sample COP  header path: ${COP_SPECIES}")
message(STATUS "  Sample ini  file   path: ${INI_FILE}")

set(COP_THERMAL_PATH "${CMAKE_SOURCE_DIR}/runtime.dat") # where to read .dat about charactersics of compoent gas
add_compile_options(-DIniFile="${INI_FILE}")
add_compile_options(-DRFile="${COP_SPECIES}")
add_compile_options(-DRPath="${COP_THERMAL_PATH}")

include_directories(
    BEFORE
    "${COP_SPECIES}"
    "${INI_SAMPLE_PATH}"
)

# ${COP_SPECIES}: 依据算例文件中的"case_setup.h"头文件自动设置NUM_SPECIES && NUM_REACTIONS #
# ${INI_SAMPLE_PATH}: 依据sample文件夹中的"ini_sample.hpp"文件选择 #