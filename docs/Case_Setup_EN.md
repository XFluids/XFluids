# XFluids Configuration & Validation Guide

This document aims to explain the compilation options and runtime parameters of the XFluids project, and records the specific configurations of the validation cases mentioned in the paper.

## 1. Project Options

### 1.1 CMake Build Options
These options are used to configure the build process, determining the physical models, numerical schemes, and hardware backends of the solver.

#### General Settings
| Option | Default | Description |
| :--- | :--- | :--- |
| `CMAKE_BUILD_TYPE` | `Release` | Build type: `Release` (Optimized), `Debug` (Debugging). |
| `SYCL_COMPILE_SYSTEM`| `oneAPI` | Compilation model: `oneAPI` (Intel oneAPI) or `ACPP` (AdaptiveCpp). |
| `VENDOR_SUBMIT`| `OFF` | Uses vendor-native compilation models, e.g., CUDA for NVIDIA devices. |
| `SelectDv` | `cuda` | Target device architecture: For oneAPI, select `host` (CPU), `cuda` (NVIDIA GPU), `hip` (AMD GPU), `intel` (Intel GPU); For ACPP, select `generic`. |
| `ARCH` | `86` | Compute Capability number of the target device. |
| `USE_DOUBLE` | `ON` | Precision control: `ON` uses double precision, `OFF` uses single precision. |
| `EXPLICIT_ALLOC` | `ON` | MPI memory management: `ON` explicitly allocates device buffers and transfers data; `OFF` allocates structure pointers on the host. |
| `USE_MPI` | `OFF` | Parallel computing: Enables MPI multi-device/multi-node parallelism. |
| `AWARE_MPI` | `OFF` | Enables CUDA-Aware MPI (only valid when `EXPLICIT_ALLOC` is `ON`). |
| `INIT_SAMPLE` | `1d-insert-st` | Case selection: Specifies the initialization logic and default configuration file (see `init_sample.cmake` for defaults). |
| `MIXTURE_MODEL` | `1d-mc-insert-shock-tube` | Mixture model path: Points to the species properties and chemical reaction mechanism files under `./runtime.dat/`. |
| `ESTIM_NAN` | `OFF` | Catches errors such as NaN, Infinity, and negative density. |
| `ERROR_OUT` | `OFF` | Outputs error messages to the screen. |

#### Numerical Methods (Numerics)

| Option | Default | Description |
| :--- | :--- | :--- |
| `DISCRETIZATION_METHOD`| `FDM` | Discretization method: `FDM` (Finite Difference Method) or `FVM` (Finite Volume Method). |
| `WENO_ORDER` | `5` | Reconstruction accuracy for inviscid flux: `5` (WENO5), `6` (WENO-CU6), `7` (WENO7). |
| `VISCOSITY_ORDER` | `Fourth` | Discretization order for viscous flux. |
| `RIEMANN_SOLVER` | `HLLC` | Riemann solver (FVM only): `HLLC`, `AUSM`. |
| `EIGEN_SYSTEM` | `positive-definite`| Eigen system solving strategy (FDM only): <br>• `positive-definite`: Solves **N-1** species equations (normalization constraint) to ensure positive definiteness.<br>• `overdetermined`: Solves all **N** species equations to handle overdetermined systems. |
| `EIGEN_ALLOC` | `OROC` | Eigen matrix memory strategy: `OROC` (One Row One Column, register optimization), `RGIF` (Register Global Interface). |
| `ARTIFICIAL_VISC_TYPE`| `GLF` | Artificial viscosity type (Shock capturing): <br>• `ROE`: Roe type.<br>• `LLF`: Local Lax-Friedrichs.<br>• `GLF`: Global Lax-Friedrichs. |
| `POSITIVITY_PRESERVING`| `OFF` | Positivity preserving: Ensures density and pressure remain positive (for strong shocks/vacuum problems). |

#### Physical Models
| Option | Default | Description |
| :--- | :--- | :--- |
| `COP` | `ON` | Multicomponent model (Component): Enables multicomponent flow. |
| `COP_CHEME` | `OFF` | Chemical reactions: Enables chemical reaction source terms (`OFF` for inert gas). |
| `Visc` | `OFF` | Physical viscosity: Enables Navier-Stokes viscous terms. |
| `Visc_Heat` | `OFF` | Heat conduction: Enables heat flux terms (requires `Visc`). |
| `Visc_Diffu` | `OFF` | Species diffusion: Enables mass diffusion terms (requires `Visc` and `COP`). |
| `THERMAL` | `NASA` | Thermodynamic data fitting format: `NASA` or `JANAF`. |

---

### 1.2 JSON Runtime Configuration
JSON configuration files control the grid, time-stepping, I/O, and initialization conditions at runtime.

#### `run` (Time Stepping & Output)
| Parameter | Description |
| :--- | :--- |
| `DtBlockSize` | Thread block size (Kernel Block Size) for calculating the time step (dt). |
| `blockSize_[x,y,z]` | Thread block dimensions for grid calculations. |
| `CFLnumber` | CFL number, controls time-stepping stability. |
| `nStepMax` | Maximum number of calculation steps. |
| `OutBoundary` | Output control: `1` includes Ghost Cells (boundary grid), `0` outputs internal domain only. |
| `OutTimeStamps` | Specified output time stamp list. Format: <br>`"Time: {-C=SliceOpt;-V=Vars;-P=Filter}"`<br>• **-C (Cut/Slice)**: Slice or line probe. e.g., `X,0.0,0.0` means a slice normal to X passing through the origin.<br>• **-V (Variables)**: Output variable list, such as `rho,P,T,yi[H2]`.<br>• **-P (Predicate)**: Predicate filter, e.g., `yi[Xe]>0.01` (outputs only points meeting the condition). |
| `OutTimeArrays` | Defines uniform output time intervals. |

#### `mpi` (Parallelization)
| Parameter | Description |
| :--- | :--- |
| `mx`, `my`, `mz` | Number of MPI domain decompositions in the x, y, and z directions. |
| `DeviceSelect` | Device selection vector `[Count, PlatformID, DeviceID]`: <br>• Number of devices used.<br>• Platform ID.<br>• Starting Device ID. |

#### `mesh` (Grid & Boundary)
| Parameter | Description |
| :--- | :--- |
| `DOMAIN_Size` | Physical size of the domain `[Lx, Ly, Lz]`. |
| `Resolution` | Grid resolution `[Nx, Ny, Nz]`. |
| `Boundarys` | Boundary condition type IDs for the 6 faces. <br>Mapping: <br>• `0`: Inflow<br>• `1`: Outflow<br>• `2`: Symmetry<br>• `3`: Periodic<br>• `4`: nslipWall (No-slip Wall)<br>• `5`: viscWall (Viscous Wall)<br>• `6`: slipWall (Slip Wall) |

#### `init` (Initial Conditions)
| Parameter | Description |
| :--- | :--- |
| `blast_mach` | Initial shock Mach number. |
| `cop_type` | Geometric shape of species distribution: <br>• `0`: 1D Set (1D regional distribution).<br>• `1`: Bubble (Spherical bubble region). |
| `blast_type` | Geometric shape of shock initialization: <br>• `0`: 1D Planar Shock.<br>• `1`: Circular/Spherical Shock. |
| `blast_center` | Coordinates of the shock center. |

---

## 2. Numerical Validation Cases Setup

This section details the configuration parameters for the cases in **Section 4: Numerical validations** of the paper. Option values in Sec 1.1 and 1.2 will adopt default values if not specified below.

### 2.1 2D Euler vortex

|           | Option / Parameter | Value / Setting     | Notes              |
| :-------- | :----------------- | :------------------ | :----------------- |
| **CMake** | `INIT_SAMPLE`      | **2d-euler-vortex** |                    |
|           | `MIXTURE_MODEL`    | **NO-COP**          |                    |
|           | `COP`              | **OFF**             | Single-component   |
|           | `COP_CHEME`        | **OFF**             | Inert gas          |
|           | `Visc`             | **OFF**             | Inviscid           |
|           | `WENO_ORDER`       | 5                   | WENO5 scheme       |
| **JSON**  | `mesh.Resolution`  | `[256, 256, 0]`     |                    |
|           | `mesh.DOMAIN_Size` | `[10.0, 10.0, 0.0]` |                    |
|           | `mesh.Boundarys`   | `[3, 3, 3, 3, ...]` | Periodic boundaries|

### 2.2 Multicomponent Inert Shock Tube
Validates multicomponent convection terms and high-order reconstruction schemes (WENO5/CU6/7).

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-insert-st** |  |
| | `MIXTURE_MODEL` | **1d-mc-insert-shock-tube** |  |
| | `COP` | **ON** | Multicomponent |
| | `COP_CHEME` | **OFF** | Inert gas |
| | `Visc` | **OFF** | Inviscid |
| | `WENO_ORDER` | 5, 6, 7 | Compares three schemes |
| | `EIGEN_SYSTEM` | `positive-definite` | **N-1** species |
| **JSON** | `mesh.Resolution` | `[400, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.1, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[0, 1, ...]` | Left: Inflow, Right: Outflow |

### 2.3 Multicomponent Diffusion
Validates molecular transport and heat conduction terms.

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-diffusion** |  |
| | `MIXTURE_MODEL` | **1d-mc-diffusion** |  |
| | `COP` | **ON** | Multicomponent |
| | `Visc` | **ON** | Enables viscous flux |
| | `Visc_Heat` | **ON** | Enables heat conduction |
| | `Visc_Diffu` | **ON** | Enables species diffusion |
| | `COP_CHEME` | **OFF** | No reaction |
| **JSON** | `mesh.Resolution` | `[200, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[3, 3, ...]` | Periodic boundaries |

### 2.4 Zero-dimensional Constant-pressure Autoignition
Validates the chemical reaction source term (ODE Solver).

Actually a 1D setup; before the 1D calculation starts, the zero-dimensional case is calculated and data files are output.

|  | Option / Parameter | Value / Setting              | Notes                                |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE`      | **1d-reactive-st**           |                                      |
|           | `MIXTURE_MODEL`    | **Reaction/H2O_18_reaction** |                                      |
|           | `COP`              | **ON**                       | Multicomponent                       |
|           | `COP_CHEME`        | **ON**                       | Reaction enabled                     |
|           | `Visc`             | **OFF**                      | Inviscid                             |
| **JSON**  | `mesh.Resolution`  | `[200, 0, 0]`                |                                      |
|           | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]`           |                                      |
|           | `mesh.Boundarys`   | `[4, 1, ...]`                | Left: No-slip Wall, Right: Outflow   |

### 2.5 Reactive Shock Tube
Validates the coupling of reaction source terms and flow.

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-reactive-st** |  |
| | `MIXTURE_MODEL` | **Reaction/H2O_18_reaction** |  |
| | `COP` | **ON** | Multicomponent |
| | `COP_CHEME` | **ON** | Reaction enabled |
| | `Visc` | **OFF** | Inviscid |
| **JSON** | `mesh.Resolution` | `[200, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[4, 1, ...]` | Left: No-slip Wall, Right: Outflow |

### 2.6 Planar Steady Detonation Propagation
Validates rigid detonation wave propagation.

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **2d-detonation** |  |
| | `MIXTURE_MODEL` | **Reaction/H2O_18_reaction** |  |
| | `COP` | **ON** | Multicomponent |
| | `COP_CHEME` | **ON** | Reaction enabled |
| | `POSITIVITY_PRESERVING`| **ON** | Positivity algorithm (for vacuum/shocks) |
| | `Visc` | **OFF** | |
| **JSON** | `mesh.Resolution` | `[100000, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.5, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[0, 1, ...]` | Left: Inflow, Right: Outflow |

### 2.7 Shock-Bubble Interactions (SBI)
Comprehensive validation: Multi-dimensional, multicomponent, shock wave, interface deformation, and (optional) reaction.

#### 2.7.1 Inert Shock-Bubble Interactions (ISBI)

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **shock-bubble** |  |
| | `MIXTURE_MODEL` | **Inert-SBI** |  |
| | `COP` | **ON** | Multicomponent |
| | `Visc` | **ON** | Viscosity enabled |
| | `Visc_Heat` | **ON** | Heat conduction enabled |
| | `Visc_Diffu` | **ON** | Diffusion enabled |
| | `COP_CHEME` | **OFF** | Inert |
| | `POSITIVITY_PRESERVING`| **ON** | Flux positivity method enabled |
| | `THERMAL` | `NASA` | Thermal fitting: `NASA` |
| **JSON** | `mesh.Resolution` | 160 ppr | Points Per Radius |
| | `mesh.Boundarys` | `[0, 1, 2, 1, 2, 1]` |  |
| | `init.blast_type` | `1` | Circular/Spherical |
| | `init.cop_type` | `1` | Bubble |
| | `init.blast_mach` | `2.83` |  |

#### 2.7.2 Reactive Shock-Bubble Interactions (RSBI)

|  | Option / Parameter      | Value / Setting         | Notes                                 |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE`           | **shock-bubble**        |                                       |
|           | `MIXTURE_MODEL`         | **Reaction/RSBI-18REA** |                                       |
|           | `COP`                   | **ON**                  | Multicomponent                        |
|           | `Visc`                  | **ON**                  | Viscosity enabled                     |
|           | `Visc_Heat`             | **ON**                  | Heat conduction enabled               |
|           | `Visc_Diffu`            | **ON**                  | Diffusion enabled                     |
|           | `COP_CHEME`             | **ON**                  | Reaction enabled                      |
|           | `POSITIVITY_PRESERVING` | **ON**                  | Flux positivity method enabled        |
|           | `THERMAL`               | `NASA`                  | Thermal fitting: `NASA`               |
| **JSON**  | `mesh.Resolution`       | 160 ppr                 | Points Per Radius                     |
|           | `mesh.Boundarys`        | `[0, 1, 2, 1, 2, 1]`    |                                       |
|           | `init.blast_type`       | `1`                     | Circular/Spherical                    |
|           | `init.cop_type`         | `1`                     | Bubble                                |
|           | `init.blast_mach`       | `2.83`                  |                                       |