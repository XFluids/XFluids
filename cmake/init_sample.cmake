# // =======================================================
# #### util sample
# // =======================================================
IF(MIXTURE_MODEL MATCHES "Reaction")
    set(COP_CHEME ON)
ENDIF()

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
    set(COP_CHEME "OFF")
    set(WENO_ORDER "6") # WENOCU6 has the larggest unrubost at Riemann separation
    set(MIXTURE_MODEL "Insert-SBI")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/sharp-interface")
    set(INI_FILE "settings/1d-shock-tube.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-insert-st")
    set(COP "ON")
    set(Visc "OFF")
    set(COP_CHEME "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(MIXTURE_MODEL "1d-mc-insert-shock-tube")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/insert-st")
    set(INI_FILE "settings/1d-shock-tube.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-eigen-st")
    set(COP "ON")
    set(Visc "OFF")
    set(COP_CHEME "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(MIXTURE_MODEL "1d-mc-eigen-shock-tube")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/inert-st")
    set(INI_FILE "settings/1d-shock-tube.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-reactive-st")
    set(COP "ON")
    set(Visc "OFF")
    set(COP_CHEME "ON")
    set(POSITIVITY_PRESERVING "OFF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/reactive-st")
    set(INI_FILE "settings/1d-reactive-st.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion")
    set(COP "ON")
    set(Visc "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON")
    set(COP_CHEME "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/diffusion")
    set(INI_FILE "settings/1d-diffusion.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion-reverse")
    set(COP "ON")
    set(Visc "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON")
    set(COP_CHEME "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    add_compile_options(-DDiffuReverse)
    set(MIXTURE_MODEL "1d-mc-diffusion-reverse")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/diffusion")
    set(INI_FILE "settings/1d-diffusion.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-laminar-flame")
    set(COP "ON")
    set(Visc "OFF")
    set(COP_CHEME "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON")
    set(POSITIVITY_PRESERVING "OFF")
    set(MIXTURE_MODEL "Reaction/H2O-N2_21_reaction")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/laminar-flame")
    set(INI_FILE "settings/1d-laminar-flame.json")

ELSEIF(INIT_SAMPLE STREQUAL "1d-diffusion-layer")
    set(COP "ON")
    set(Visc "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON")
    set(COP_CHEME "ON")
    set(POSITIVITY_PRESERVING "OFF")
    add_compile_options(-DConstantFurrier)
    set(MIXTURE_MODEL "Reaction/H2O-N2_21_reaction")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/1D-X-Y-Z/diffusion-layer")
    set(INI_FILE "settings/1d-diffusion-layer.json")

# // =======================================================
# #### 2d sample
# // =======================================================
# actually EulerVortex case
ELSEIF(INIT_SAMPLE STREQUAL "2d-euler-vortex")
    set(COP "OFF")
    set(Visc "OFF")
    set(Visc_Heat "OFF")
    set(Visc_Diffu "OFF")
    set(COP_CHEME "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-EulerVortex")
    set(INI_FILE "settings/2d-euler-vortex.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-shocks") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(COP "OFF")
    set(Visc "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/shocks-interaction")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-shock-interruption") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(COP "OFF")
    set(Visc "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/shock-interruption")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-interruptions-plus") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(COP "OFF")
    set(Visc "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/interruptions-plus")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-riemann-interruptions-reduce") # 2d-riemann(-shocks/-shock-interruption/-interruptions-plus/-interruptions-reduce)
    set(COP "OFF")
    set(Visc "OFF")
    set(POSITIVITY_PRESERVING "OFF")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-Riemann/interruptions-reduce")
    set(INI_FILE "settings/2d-riemann.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-detonation")
    set(COP "ON")
    set(Visc "OFF")
    set(THERMAL "NASA") # NASA fit of Xe
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    set(MIXTURE_MODEL "Reaction/H2O-N2_19_reaction")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/2D-detonation")
    set(INI_FILE "settings/2d-detonation.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-mixing-layer")
    set(COP "ON")
    set(POSITIVITY_PRESERVING "ON")
    set(MIXTURE_MODEL "Reaction/H2O-N2_21_reaction")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/mixing-layer")
    set(INI_FILE "settings/2d-mixing-layer.json")

ELSEIF(INIT_SAMPLE STREQUAL "2d-under-expanded-jet")
    set(COP "ON")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/under-expanded-jet")
    set(INI_FILE "settings/expanded-jet.json")

# // =======================================================
# #### 3d sample
# // =======================================================
ELSEIF(INIT_SAMPLE STREQUAL "shock-bubble-without-fuel")
    set(COP "ON")
    set(Visc "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON") # depends on COP=ON
    set(THERMAL "NASA") # NASA fit of Xe
    set(WENO_ORDER "6") # 5, 6 or 7: WENO5, WENOCU6 or WENO7 for High-Order FluxWall reconstruction
    set(COP_CHEME "OFF")
    set(VISCOSITY_ORDER "Fourth") # Fourth, Second order viscosity discretization method, 2rd-order used both in FDM and FVM, 4th only used in FDM.
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts=1)
    set(MIXTURE_MODEL "Insert-SBI-without-fuel")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "shock-bubble")
    set(COP "ON")
    set(Visc "ON")
    set(Visc_Heat "ON")
    set(Visc_Diffu "ON") # depends on COP=ON
    set(THERMAL "NASA") # NASA fit of Xe
    set(WENO_ORDER "6") # 5, 6 or 7: WENO5, WENOCU6 or WENO7 for High-Order FluxWall reconstruction
    set(VISCOSITY_ORDER "Fourth") # Fourth, Second order viscosity discretization method, 2rd-order used both in FDM and FVM, 4th only used in FDM.
    set(POSITIVITY_PRESERVING "ON")
    set(ARTIFICIAL_VISC_TYPE "GLF")
    add_compile_options(-DSBICounts=1)

    if((MIXTURE_MODEL STREQUAL "Reaction/RSBI-18REA") OR(MIXTURE_MODEL STREQUAL "Reaction/RSBI-19REA"))
        set(COP_CHEME ON)
    elseif(MIXTURE_MODEL STREQUAL "Insert-SBI")
        set(COP_CHEME OFF)
    else()
        message(FATAL_ERROR " Not suitable MIXTURE_MODEL opened: checkout option MIXTURE_MODEL for RSBI: Reaction/RSBI-18REA, Reaction/RSBI-19REA")
    endif()

    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/shock-bubble-intera")
    set(INI_FILE "settings/shock-bubble.json")

ELSEIF(INIT_SAMPLE STREQUAL "3d-under-expanded-jet")
    set(COP "ON")
    set(INI_SAMPLE_PATH "/src/solver_Ini/sample/under-expanded-jet")
    set(INI_FILE "settings/expanded-jet.json")
ELSE()
    message(FATAL_ERROR "ini sample isn't given.")
ENDIF()

IF(COP)
    add_compile_options(-DCOP)
    IF(POSITIVITY_PRESERVING)
        add_compile_options(-DPOSP=1)
    ELSE()
        add_compile_options(-DPOSP=0)
    ENDIF()
    IF(COP_CHEME)
        add_compile_options(-DCOP_CHEME=1)
    ELSE()
        add_compile_options(-DCOP_CHEME=0)
    ENDIF(COP_CHEME)

ELSE(COP)
    set(Gamma "1.4")
    set(MIXTURE_MODEL "NO-COP")
    add_compile_options(-DPOSP=0)
    add_compile_options(-DNUM_REA=1)
    add_compile_options(-DNUM_COP=0)
    add_compile_options(-DCOP_CHEME=0)
    add_compile_options(-DNUM_SPECIES=1)
    add_compile_options(-DNCOP_Gamma=${Gamma})
ENDIF(COP)

set(MIXTURE_MODEL "/runtime.dat/${MIXTURE_MODEL}")
message(STATUS "Solvers' settings: ")

if(USE_DOUBLE)
    message(STATUS "  Double Precision running")
else()
    message(STATUS "  Float  Precision running")
endif()

message(STATUS "  Multi-Component: ${COP}")
message(STATUS "    Species' Thermo Fit: ${THERMAL}")
message(STATUS "  Capture unexpected errors: ${ESTIM_NAN}")

if(ERROR_OUT)
    message(STATUS "    Out intermediate variables while unexpected error captured, ")
    message(STATUS "    Highly grow up device memory usage.")
endif()

message(STATUS "  Convention term scheme: WENO${WENO_ORDER}")
message(STATUS "    Discretization method: ${DISCRETIZATION_METHOD}")
message(STATUS "    Artificial  viscosity: ${ARTIFICIAL_VISC_TYPE}")
message(STATUS "    Positivity Preserving: ${POSITIVITY_PRESERVING}")

# message(STATUS "    Fix nan primitive variables(rho,p,T): ${ERROR_PATCH_PRI}")
# message(STATUS "    Fix negative mass fraction(Roe_Yi/SetValue_Yii): ${ERROR_PATCH_YI}/${ERROR_PATCH_YII})
# #Not damage equations but decrease the accuracy of solution")
message(STATUS "  Viscous Flux term: ${Visc}")

IF(Visc)
    message(STATUS "    Viscous Visc_Heat term: ${Visc_Heat}")
    message(STATUS "    Viscous Diffusion term: ${Visc_Diffu}")
    message(STATUS "    Viscous Flux discretization order: ${VISCOSITY_ORDER}")
ENDIF()

message(STATUS "Sample select: ${INIT_SAMPLE}")
message(STATUS "  Sample COP  header path: ${MIXTURE_MODEL}")
message(STATUS "  Sample init sample path: ${INI_SAMPLE_PATH}")
message(STATUS "  Sample ini  file   path: ${CMAKE_SOURCE_DIR}/${INI_FILE}")

add_compile_options(-DINI_SAMPLE="${INIT_SAMPLE}")
add_compile_options(-DRFile="${MIXTURE_MODEL}")
add_compile_options(-DIniFile="${CMAKE_SOURCE_DIR}/${INI_FILE}")
add_compile_options(-DRPath="/runtime.dat") # where to read .dat about charactersics of compoent gas

include_directories(
    BEFORE
    "${CMAKE_SOURCE_DIR}/${MIXTURE_MODEL}"
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

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/libs/${INIT_SAMPLE}_${SYCL_COMPILE_SYSTEM}_${SelectDv}_${ARCH}_${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/libs/${INIT_SAMPLE}_${SYCL_COMPILE_SYSTEM}_${SelectDv}_${ARCH}_${CMAKE_BUILD_TYPE})

IF(INIT_SAMPLE MATCHES "read_grid/") # read grid
    set(INIT_SAMPLE "read_grid")
ENDIF()
