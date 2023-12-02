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

# // =======================================================
# #### util sample
# // =======================================================
IF(INIT_SAMPLE MATCHES "read_grid/") # read grid
    set(INIT_SAMPLE "src/${INIT_SAMPLE}")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/read-grid")
    set(INI_FILE "${INIT_SAMPLE}.json")

ELSEIF(INIT_SAMPLE STREQUAL "guass-wave")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/guass-wave")
    set(INI_FILE "settings/guass-wave.json")

# // =======================================================
# #### 1d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "sharp-interface")
    set(DIM_X "ON")
    set(COP_CHEME "OFF")
    set(WENO_ORDER "6") # WENOCU6 has the larggest unrubost at Riemann separation
    set(COP_SPECIES "Insert-SBI")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/sharp-interface")
    set(INI_FILE "settings/1d-shock-tube.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-insert-st")
    IF(DIM_NUM STREQUAL "1")
        set(COP "ON")
        set(Visc "OFF")
        set(COP_CHEME "OFF")
        set(ESTIM_NAN "OFF")
        set(POSITIVITY_PRESERVING "OFF")
        set(COP_SPECIES "1d-mc-insert-shock-tube")
        set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/insert-st")
        set(INI_FILE "settings/1d-shock-tube.json")
    ELSEIF()
        message(FATAL_ERROR "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-reactive-st")
    IF(DIM_NUM STREQUAL "1")
        set(COP "ON")
        set(Visc "OFF")
        set(COP_CHEME "ON")
        set(COP_CHEME_TEST "ON")
        set(ESTIM_NAN "OFF")
        set(POSITIVITY_PRESERVING "OFF")
        set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/reactive-st")
        set(INI_FILE "settings/1d-reactive-st.json")
    ELSEIF()
        message(FATAL_ERROR "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion")
    IF(DIM_NUM STREQUAL "1")
        set(COP "ON")
        set(Visc "ON")
        set(Visc_Heat "ON")
        set(Visc_Diffu "ON")
        set(COP_CHEME "OFF")
        set(ESTIM_NAN "OFF")
        set(POSITIVITY_PRESERVING "OFF")
        set(COP_SPECIES "1d-mc-diffusion")
        set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/diffusion")
        set(INI_FILE "settings/1d-diffusion.json")
    ELSEIF()
        message(FATAL_ERROR "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion-reverse")
    IF(DIM_NUM STREQUAL "1")
        set(COP "ON")
        set(Visc "ON")
        set(Visc_Heat "ON")
        set(Visc_Diffu "ON")
        set(COP_CHEME "OFF")
        set(ESTIM_NAN "OFF")
        set(POSITIVITY_PRESERVING "OFF")
        add_compile_options(-DDiffuReverse)
        set(COP_SPECIES "1d-mc-diffusion-reverse")
        set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/diffusion")
        set(INI_FILE "settings/1d-diffusion-reverse.json")
    ELSEIF()
        message(FATAL_ERROR "More DIM opened than needed: checkout option DIM_X, DIM_Y, DIM_Z")
    ENDIF()

# // =======================================================
# #### 2d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-shocks") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(COP "OFF")
    set(Visc "OFF")
    set(ESTIM_NAN "OFF")
    set(ERROR_PATCH_YII "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(COP_SPECIES "NO-COP")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/shocks-interaction")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-shock-interruption") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(COP "OFF")
    set(Visc "OFF")
    set(ESTIM_NAN "OFF")
    set(ERROR_PATCH_YII "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(COP_SPECIES "NO-COP")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/shock-interruption")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-interruptions-plus") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(COP "OFF")
    set(Visc "OFF")
    set(ESTIM_NAN "OFF")
    set(ERROR_PATCH_YII "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(COP_SPECIES "NO-COP")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/interruptions-plus")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-interruptions-reduce") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(COP "OFF")
    set(Visc "OFF")
    set(ESTIM_NAN "OFF")
    set(ERROR_PATCH_YII "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(COP_SPECIES "NO-COP")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/interruptions-reduce")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-detonation")
    set(COP "ON")
    set(Visc "OFF")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(COP_CHEME "ON")
    set(ESTIM_NAN "ON")
    set(THERMAL "NASA") # NASA fit of Xe
    set(ERROR_PATCH_YI "OFF")
    set(ERROR_PATCH_YII "OFF")
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-detonation")
    set(INI_FILE "settings/2d-detonation.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-shock-bubble-without-fuel")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(THERMAL "NASA") # NASA fit of Xe
    set(COP_CHEME "OFF")
    set(ESTIM_NAN "ON")
    set(ERROR_PATCH_YII "ON")
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts)
    add_compile_options(-DSBI_WITHOUT_FUEL)
    set(COP_SPECIES "Insert-SBI-without-fuel")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-shock-bubble")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(THERMAL "NASA") # NASA fit of Xe
    set(ESTIM_NAN "ON")
    set(ERROR_PATCH_YII "ON")
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts)

    if(${COP_CHEME})
        if((REACTION_MODEL STREQUAL "RSBI-18REA") OR(REACTION_MODEL STREQUAL "RSBI-19REA"))
        else()
            message(FATAL_ERROR " Not suitable REACTION_MODEL opened: checkout option REACTION_MODEL for RSBI: RSBI-18REA, RSBI-19REA")
        endif()
    endif()

    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/under-expanded-jet")
    set(INI_FILE "settings/expanded-jet.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-mixing-layer")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(REACTION_MODEL "H2O_21_reaction")
    set(COP_SPECIES "Reaction/H2O_21_reaction")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/mixing-layer")
    set(INI_FILE "settings/2d-mixing-layer.json")

# // =======================================================
# #### 3d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "3d-shock-bubble-without-fuel")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(THERMAL "NASA") # NASA fit of Xe
    set(COP_CHEME "OFF")
    set(ESTIM_NAN "ON")
    set(ERROR_PATCH_YII "ON")
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts)
    add_compile_options(-DSBI_WITHOUT_FUEL)
    set(COP_SPECIES "Insert-SBI-without-fuel")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "3d-shock-bubble")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(THERMAL "NASA") # NASA fit of Xe
    set(ESTIM_NAN "ON")
    set(ERROR_PATCH_YII "ON")
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts)

    if(${COP_CHEME})
        if((REACTION_MODEL STREQUAL "RSBI-18REA") OR(REACTION_MODEL STREQUAL "RSBI-19REA"))
        else()
            message(FATAL_ERROR " Not suitable REACTION_MODEL opened: checkout option REACTION_MODEL for RSBI: RSBI-18REA, RSBI-19REA")
        endif()
    endif()

    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(COP "ON")
    set(DIM_X "ON")
    set(DIM_Y "ON")
    set(DIM_Z "ON")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/under-expanded-jet")
    set(INI_FILE "settings/expanded-jet.json")
ELSE()
    message(FATAL_ERROR "ini sample isn't given.")
ENDIF()

set(COP_SPECIES "/runtime.dat/${COP_SPECIES}") # Be invalid while option COP_CHEME "ON"

IF(COP)
    add_compile_options(-DCOP)

    IF(COP_CHEME)
        add_compile_options(-DCOP_CHEME)
        set(COP_SPECIES "/runtime.dat/Reaction/${REACTION_MODEL}") # where to read species including reactions

        IF(COP_CHEME_TEST)
            add_compile_options(-DODESolverTest)
        ENDIF()

        IF(${CHEME_SOLVER} MATCHES "Q2")
            add_compile_options(-DCHEME_SOLVER=0)
        ELSEIF(${CHEME_SOLVER} MATCHES "CVODE")
            add_compile_options(-DCHEME_SOLVER=1)
        ELSE()
        ENDIF()

        IF(${CHEME_SPLITTING} MATCHES "Lie")
            add_compile_options(-DCHEME_SPLITTING=1)
        ELSEIF(${CHEME_SPLITTING} MATCHES "Strang")
            add_compile_options(-DCHEME_SPLITTING=2)
        ELSE()
            add_compile_options(-DCHEME_SPLITTING=1) # Lie splitting by default
        ENDIF()
    ENDIF(COP_CHEME)

ELSE(COP)
    set(Gamma "1.4")
    set(COP_SPECIES "NO-COP")
    add_compile_options(-DNUM_REA=0)
    add_compile_options(-DNUM_SPECIES=1)
    add_compile_options(-DNCOP_Gamma=${Gamma})
ENDIF(COP)

message(STATUS "Solvers' settings: ")

if(USE_DOUBLE)
    message(STATUS "  Double Precision on: ${SYCL_COMPILE_SYSTEM}: ${SelectDv}${ARCH}")
else()
    message(STATUS "  Float Precision on: ${SYCL_COMPILE_SYSTEM}: ${SelectDv}${ARCH}")
endif()
message(STATUS "  Multi-Component: ${COP}")
message(STATUS "  Species' Thermo Fit: ${THERMAL}")
message(STATUS "  DIM_X: ${DIM_X}, DIM_Y: ${DIM_Y}, DIM_Z: ${DIM_Z}")
message(STATUS "  Convention term scheme: WENO${WENO_ORDER}")
message(STATUS "    Convention discretization method: ${DISCRETIZATION_METHOD}")
message(STATUS "    Reconstruction artificial viscosity type: ${ARTIFICIAL_VISC_TYPE}")
message(STATUS "  Positivity_Preserving: ${POSITIVITY_PRESERVING}")
message(STATUS "    Capture unexpected errors: ${ESTIM_NAN}")
message(STATUS "    Fix nan primitive variables(rho,p,T): ${ERROR_PATCH_PRI}")
message(STATUS "    Fix negative mass fraction(Roe_Yi/SetValue_Yii): ${ERROR_PATCH_YI}/${ERROR_PATCH_YII}, Not damage equations but decrease the accuracy of solution")
message(STATUS "  Viscous Flux term: ${Visc}")

IF(Visc)
    message(STATUS "    Viscous Visc_Heat term: ${Visc_Heat}")
    message(STATUS "    Viscous Diffusion term: ${Visc_Diffu}")
    message(STATUS "    Viscous Flux discretization order: ${VISCOSITY_ORDER}")
ENDIF()

message(STATUS "  Reaction term: ${COP_CHEME}")

IF(COP_CHEME)
    message(STATUS "    Reaction term Solver: ${CHEME_SOLVER}")
    message(STATUS "    Reacting ZeroDimensional TEST: ${COP_CHEME_TEST}")
ENDIF()

message(STATUS "Sample select: ${INIT_SAMPLE}")
message(STATUS "  Sample COP  header path: ${COP_SPECIES}")
message(STATUS "  Sample init sample path: ${INI_SAMPLE_PATH}")
message(STATUS "  Sample ini  file   path: ${CMAKE_SOURCE_DIR}/${INI_FILE}")

add_compile_options(-DINI_SAMPLE="${INIT_SAMPLE}")
add_compile_options(-DRFile="${COP_SPECIES}")
add_compile_options(-DIniFile="${CMAKE_SOURCE_DIR}/${INI_FILE}")
add_compile_options(-DRPath="/runtime.dat") # where to read .dat about charactersics of compoent gas

include_directories(
    BEFORE
    "${CMAKE_SOURCE_DIR}/${COP_SPECIES}"
    "${CMAKE_SOURCE_DIR}/${INI_SAMPLE_PATH}"
    "${CMAKE_SOURCE_DIR}/src/solver_Reconstruction/viscosity/${VISCOSITY_ORDER}_Order"
)

IF(${DISCRETIZATION_METHOD} STREQUAL "FDM")
    include_directories(
        BEFORE
        "${CMAKE_SOURCE_DIR}/src/solver_Reconstruction/${DISCRETIZATION_METHOD}_Method"
        "${CMAKE_SOURCE_DIR}/src/solver_Reconstruction/${DISCRETIZATION_METHOD}_Method/${EIGEN_SYSTEM}_eigen"
    )
ENDIF()

IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib${INIT_SAMPLE})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib${INIT_SAMPLE})
ELSE()
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/${INIT_SAMPLE})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/libs/${INIT_SAMPLE})
ENDIF()

IF(INIT_SAMPLE MATCHES "read_grid/") # read grid
    set(INIT_SAMPLE "read_grid")
ENDIF()
