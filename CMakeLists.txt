cmake_minimum_required(VERSION 3.10)
PROJECT(XFLUIDS LANGUAGES CXX)

# // == Compile system =====================================
# ####  1. oneAPI: Intel oneAPI compiling system  2. OpenSYCL: OpenSYCL syclcc compiling system
set(SYCL_COMPILE_SYSTEM "OpenSYCL") # # oneAPI or OpenSYCL compile system,
set(SelectDv "generic") # which platform: generic(for sscp), host(for cpu), cuda(for llvm-cuda), cuda-nvcxx(for nvhpc), hip(for amd)
set(ARCH "86") # host: no ARCH needed; # cuda/cuda-nvcxx: 75 86;# hip: 906, 1030

# // == MPI settings =======================================
option(USE_MPI "if enable mpi" OFF)
option(AWARE_MPI "if enable aware-mpi" OFF) # only be valid while EXPLICIT_ALLOC is set "ON"

# // == RUN settings =======================================
option(ASYNC_SUBMIT "if use asynchronous submmission" OFF)
option(VENDOR_SUBMIT "if use CUDA or HIP original parallel submmision" OFF)

# // == Functionality option ===============================
# # // // == Ini sample selection ===========================
# # # # (0d and 1d samples are forced out with plt)
# # read_grid/${GRID_NAME}("read_grid/rae2822.xyz"): using Plot3D grid: ./src/read_grid/${GRID_NAME} to initialize flow field
# # # # 1d, 2d, 3d util: for-debug, guass-wave, sharp-interface
# # 1d-insert-st			# # 1d-reactive-st		 			## shock tube without viscisity and positivity-preserving
# # 1d-eigen-st				# # 1d-reactive-st		 			## shock tube without viscisity and positivity-preserving
# # 1d-diffusion 			# # 1d-diffusion-reverse 			## mass and heat diffusion without convention
# # 1d-laminar-flame		# # 2d-detonation 					## multi-dimensional reactive laminar flame
# # 2d-riemann(-shocks/-interruptions-reduce/-interruptions-plus/-shock-interruption) 		## no-cop sample
# # 1d-diffusion-layer		# # 2d-mixing-layer
# # 2d-under-expanded-jet	# # 3d-under-expanded-jet
# # shock-bubble			# # shock-bubble-without-fuel
# # // // == MIXTURE_MODEL settings =========================================
# # // // // == insert MIXURE_MODEL settings ================================
# #	NO-COP
# # 1d-mc-insert-shock-tube		# # 1d-mc-diffusion 			# # 1d-mc-diffusion-reverse
# # 2d-under-expanded-jet		# # Insert-SBI 					# # Insert-SBI-without-fuel
# # // // // == reacting MIXURE_MODEL settings ==============================
# # Reaction/Insert-Air			# # Reaction/Inert_H2+N2
# # Reaction/2H+O2=H+HO2		# # Reaction/N2_dissociation
# # Reaction/H2O_18_reaction	# # Reaction/H2O_19_reaction	# # Reaction/H2O_23_reaction
# # Reaction/H2O-N2_8_reaction	# # Reaction/H2O-N2_19_reaction	# # Reaction/H2O-N2_21_reaction
# # Reaction/RSBI-18REA			# # Reaction/RSBI-19REA			# # modified for RSBI samples
set(INIT_SAMPLE "1d-reactive-st") # set INIT_SAMPLE for Initializing settings
set(MIXTURE_MODEL "Reaction/H2O_18_reaction") # location of species_list.dat read in project listed under ./runtime.dat/ folder

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
set(VISCOSITY_ORDER "Fourth") # Fourth, Second order viscosity discretization method, 2rd-order used both in FDM and FVM, 4th only used in FDM.

# // == CMake modules ======================================
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake") # #default local cmake macro repository
include(init_cmake) # cmake compile options
include(init_options) # add program functionality# // == ADD source dir =====================================
add_subdirectory(src) # add main program source dir
