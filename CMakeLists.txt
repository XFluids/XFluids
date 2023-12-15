cmake_minimum_required(VERSION 3.10)
PROJECT(XFLUIDS LANGUAGES CXX)

# // == Compile system =====================================
# ####  1. oneAPI: Intel oneAPI compiling system  2. OpenSYCL: OpenSYCL syclcc compiling system
set(SYCL_COMPILE_SYSTEM "OpenSYCL") # # oneAPI or OpenSYCL compile system,
set(SelectDv "cuda") # which platform: host(for cpu), cuda(for llvm-cuda), cuda-nvcxx(for nvhpc), hip(for amd)
set(ARCH "86") # host: no ARCH needed; # cuda/cuda-nvcxx: 75 86;# hip: 906, 1030

# // == MPI settings =======================================
option(USE_MPI "if enable mpi" ON)
option(AWARE_MPI "if enable aware-mpi" OFF) # only be valid while EXPLICIT_ALLOC is set "ON"

# // == Functionality option ===============================
# # // // == Ini sample selection ===========================
# # # # (0d and 1d samples are forced out with plt)
# # read_grid/${GRID_NAME}("read_grid/rae2822.xyz"): using Plot3D grid: ./src/read_grid/${GRID_NAME} to initialize flow field
# # 1d, 2d, 3d util: for-debug, guass-wave, sharp-interface
# # 0d-H2-O2-freely-flame, depends on both COP_CHEME and COP_CHEME_TEST are set "ON"
# # 1d-insert-st ## automatically rewrite COP_CHEME, Visc to "OFF"
# # 1d-reactive-st ## automatically rewrite COP_CHEME to "ON", Visc to "OFF", MIXTURE_MODEL, by default: H2O_19_reaction
# # 1d-diffusion 1d-diffusion-reverse ## automatically rewrite Visc, Visc_Heat, Visc_Diffu to "ON", COP_CHEME to "OFF"
# # 2d-detonation ## automatically rewrite COP_CHEME to "ON", Visc to "OFF"
# # 2d-riemann(-shocks/-interruptions-reduce/-interruptions-plus/-shock-interruption) ## only XY DIR support
# # 2d-shock-bubble, 2d-shock-bubble-without-fuel, 2d-mixing-layer, 2d-under-expanded-jet ## only XY DIR support
# # 3d-shock-bubble, 3d-shock-bubble-without-fuel, 3d-under-expanded-jet
# # // // == MIXTURE_MODEL settings =========================================
# # // // // == insert MIXURE_MODEL settings ================================
# #	NO-COP					# # 1d-mc-insert-shock-tube
# # 1d-mc-diffusion			# # 1d-mc-diffusion-reverse # # 2d-under-expanded-jet
# # Insert-SBI 				# # Insert-SBI-without-fuel
# # // // // == reacting MIXURE_MODEL settings ==============================
# # 2species				# # CJ_mechanism			# # inert_2species
# # H2O_18_reaction			# # H2O_19_reaction			# # H2O_21_reaction			# # H2O_23_reaction
# # RSBI-18REA				# # RSBI-19REA				# # for shock-bubble-interaction sample
set(INIT_SAMPLE "3d-shock-bubble-without-fuel") # set INIT_SAMPLE for Initializing settings
set(MIXTURE_MODEL "1d-mc-diffusion") # location of species_list.dat read in project listed under ./runtime.dat/ folder

# # // // == Convention Term ================================
option(USE_DOUBLE "if double" ON)
set(DISCRETIZATION_METHOD "FDM") # FVM or FDM ME FluxWall reconstruction
set(WENO_ORDER "5") # 5, 6 or 7: WENO5, WENOCU6 or WENO7 for High-Order FluxWall reconstruction
set(RIEMANN_SOLVER "HLLC") # HLLC or AUSM Riemann solver, used in FVM method
set(EIGEN_SYSTEM "positive-definite") # positive-definite or overdetermined, used in FDM method
set(ARTIFICIAL_VISC_TYPE "GLF") # ROE for Roe_type, LLF for local Lax-Friedrichs eigen max, GLF for global Lax-Friedrichs eigen max of all points in Domain

# # // // == Viscosity Term =================================
option(Visc "if add physic viscidity flux" OFF)
option(Visc_Heat "if add heat transfer term" ON)
option(Visc_Diffu "if add mass diffusion term" ON) # depends on COP=ON
set(VISCOSITY_ORDER "Second") # Fourth, Second order viscosity discretization method, 2rd-order used both in FDM and FVM, 4th only used in FDM.

# // == CMake modules ======================================
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake") # #default local cmake macro repository
include(init_cmake) # cmake compile options
include(init_options) # add program functionality

# // == ADD source dir =====================================
add_subdirectory(src) # add main program source dir
