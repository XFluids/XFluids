# Euler-SYCL

## 1. Dependencies

- ### [intel oneapi version &gt;= 2023.0.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline)

- ### [codeplay Solutions for Nvidia and AMD backends](https://codeplay.com/solutions/oneapi/)

- ### environment set for oneAPI appended codeplay sultion libs

    ````bash
    source /opt/intel/oneapi/setvars.sh  --force --include-intel-llvm
    ````

    or you can use the script files before cmake,

    ````bash
    source ./script/oneapi_xx.sh
    ````

## 2. Compile and usage of this project

### 2.1. Read $/CMakeLists.txt

- CMAKE_BUILD_TYPE is set to "Release" by default, SYCL code would target to host while ${CMAKE_BUILD_TYPE}==Debug
- set INIT_SAMPLE as the problem being tested, path to "species_list.dat" should be given to COP_SPECIES
- if COP_CHEME is set to "ON", path to "species_list.dat" and "reaction_list.dat" would be rewriten by the given value of REACTION_MODEL
- value of SelectDv must match with the value of Pform_id, details referenced in [4-device-discovery](#4-device-discovery)
- MPI and AWARE-MPI support added in project, AWARE_MPI need specific GPU-ENABLED mpi version, details referenced in [5-mpi-libs](#5-mpi-libs)
- tempreture(T) approximately beside and below 200 may cause NAN errors: T must be enough high

### 2.2. BUILD

  ````bash
    cd ./EulerSYCL
    mkdir build && cd ./build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j
  ````

### 2.3. RUN

- EulerSYCL automatically read ${workspaceFolder}/*.ini file depending on INIT_SAMPLE setting, you can still append other specific .ini file to EulerSYCL in cmd

  ````bash
    $./EulerSYCL ./setup.ini
  ````

## 3. Compiler and compile options for backends

- ### host and intel backends

    ````cmake
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    ````

- ### cuda backends,$ like sm_75,sm_86 tested

    ````cmake
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-gpu-arch=${ARCH}")
    ````

- ### amd backends,$ like gfx906 tested

    ````cmake
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=${ARCH}")
    ````

## 4. Device discovery

### 4.1. exec "sycl-ls" in cmd for device counting

  ```cmd
    $sycl-ls
    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2022.15.12.0.01_081451]
    [opencl:cpu:1] Intel(R) OpenCL, AMD Ryzen 7 5800X 8-Core Processor              3.0 [2022.15.12.0.01_081451]
    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA T600 0.0 [CUDA 11.5]
  ```

### 4.2. select target device in SYCL project

- set device_id=1 for targetting host and throwing mission to AMD Ryzen 7 5800X 8-Core Processor
- set device_id=2 for targetting nvidia GPU and throwing mission to NVIDIA T600

    ````C++
    auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
    sycl::queue q(device);
    ````

## 5. MPI libs

### 5.1. set MPI_PATH browsed by cmake before build

- cmake system of this project browse libmpi.so automatically in path of ${MPI_PATH}/lib, please export MPI_PATH to the mpi you want:

    ````cmd
    export MPI_PATH=/home/ompi
    ````

### 5.2. the value of MPI_HOME, MPI_INC, path of MPI_CXX(libmpi.so) output on screen while libmpi.so is found

  ````cmake
    -- MPI settings:
    --   MPI_HOME:/home/ompi
    --   MPI_INC: /home/ompi/include added
    --   MPI_CXX lib located: /home/ompi/lib/libmpi.so found
  ````

## 6. .ini file arguments

### 6.1. [run] parameters

  | name of parameters |                             function                             |  type  | default value  |
  | :----------------- | :--------------------------------------------------------------: | :----: | :------------- |
  | StartTime          |                   begin time of the caculation                   | float | 0.0f           |
  | OutputDir          |                   where to output result file                    | string | "./"           |
  | OutBoundary        |                     if output boundary piont                     |  bool  | flase          |
  | OutDIRX            |                     if output XDIR piont                         |  bool  | bool(DIM_X)          |
  | OutDIRY            |                     if output YDIR piont                         |  bool  | bool(DIM_Y)          |
  | OutDIRZ            |                     if output ZDIR piont                         |  bool  | bool(DIM_Z)          |
  | nStepMax           |              max number of steps for evolution loop              |  int  | 10             |
  | nOutMax            |         max number of files outputted for evolution loop         |  int  | 0              |
  | OutInterval        |             interval number of steps for once output             |  int  | nStepMax       |
  | OutTimeBeginning   |                  the time stamp first output                     | float | 0.0            |
  | nOutTimeStamps     |                     number of time interval                      |  int  | 1              |
  | OutTimeInterval    |                   time interval of once ouptut                   | float | 0.0            |
  | OutTimeMethod      | 0 for reading time stamps in .ini, 1 for in ./runtime.dat/time_stamps.dat | int | 1            |
  | DtBlockSize        |             1D local_ndrange parameter used in GetDt             |  int  | 4              |
  | blockSize_x        | X direction local_ndrange parameter used in SYCL lambda function |  int  | BlSz.BlockSize |
  | blockSize_y        | Y direction local_ndrange parameter used in SYCL lambda function |  int  | BlSz.BlockSize |
  | blockSize_z        | Z direction local_ndrange parameter used in SYCL lambda function |  int  | BlSz.BlockSize |

### 6.2. [mpi] parameters

  | name of parameters |                          function                          | type | default value |
  | :----------------- | :---------------------------------------------------------: | :--: | :------------ |
  | NUM                |            number of MPI devices can be selected            | int | 1             |
  | mx                 | number of MPI threads at X direction in MPI Cartesian space | int | 1             |
  | my                 | number of MPI threads at Y direction in MPI Cartesian space | int | 1             |
  | mz                 | number of MPI threads at Z direction in MPI Cartesian space | int | 1             |

### 6.3. [mesh] parameters

  | name of parameters |                                 function                                 | type | default value |
  | :----------------- | :-----------------------------------------------------------------------: | :---: | :------------ |
  | DOMAIN_length      |                    size of the XDIR edge of the domain                    | float | 1.0           |
  | DOMAIN_width       |                    size of the YDIR edge of the domain                    | float | 1.0           |
  | DOMAIN_height      |                    size of the ZDIR edge of the domain                    | float | 1.0           |
  | xmin               |             starting coordinate at X direction of the domain             | float | 0.0           |
  | ymin               |             starting coordinate at Y direction of the domain             | float | 0.0           |
  | zmin               |             starting coordinate at Z direction of the domain             | float | 0.0           |
  | X_inner            |              resolution setting at X direction of the domain              |  int  | 1             |
  | Y_inner            |              resolution setting at Y direction of the domain              |  int  | 1             |
  | Z_inner            |              resolution setting at Z direction of the domain              |  int  | 1             |
  | Bwidth_X           |         number of ghost cells at X direction's edge of the domain         |  int  | 4             |
  | Bwidth_Y           |         number of ghost cells at Y direction's edge of the domain         |  int  | 4             |
  | Bwidth_Z           |         number of ghost cells at Z direction's edge of the domain         |  int  | 4             |
  | CFLnumber          |                     CFL number for advancing in time                     | float | 0.6           |
  | boundary_xmin      | type of Boundary at xmin edge of the domain,influce values of ghost cells |  int  | 2             |
  | boundary_xmax      | type of Boundary at xmax edge of the domain,influce values of ghost cells |  int  | 2             |
  | boundary_ymin      | type of Boundary at ymin edge of the domain,influce values of ghost cells |  int  | 2             |
  | boundary_ymax      | type of Boundary at ymax edge of the domain,influce values of ghost cells |  int  | 2             |
  | boundary_zmin      | type of Boundary at zmin edge of the domain,influce values of ghost cells |  int  | 2             |
  | boundary_zmax      | type of Boundary at zmax edge of the domain,influce values of ghost cells |  int  | 2             |

### 6.4. [init] parameters

- #### Some arguments may be invalid due to samples settings kernel function inside /src/sample/

    | name of parameters |                         function                         | type | default value      |
    | :----------------- | :------------------------------------------------------: | :---: | :----------------- |
    | blast_type         |                 type of blast in domain                 |  int  | 0                  |
    | blast_center_x     |      position of the blast at X direction in domain      | float | 0.0                |
    | blast_center_y     |      position of the blast at Y direction in domain      | float | 0.0                |
    | blast_center_z     |      position of the blast at Z direction in domain      | float | 0.0                |
    | blast_radius       |     radius ratio of shortest edge of domain of blast     | float | 0.0                |
    | blast_mach         |     ini blast by shock-bubble-interaction theroy while blast_mach is set > 1.0 | float | 0.0 |
    | blast_density_in   |           rho of the fluid upstream the blast           | float | 0.0                |
    | blast_density_out  |          rho of the fluid downstream the blast          | float | 0.0                |
    | blast_pressure_in  |            P of the fluid upstream the blast            | float | 0.0                |
    | blast_pressure_out |           P of the fluid downstream the blast           | float | 0.0                |
    | blast_tempreture_in|            T of the fluid upstream the blast            | float | 0.0                |
    | blast_tempreture_out|           T of the fluid downstream the blast           | float | 0.0                |
    | blast_u_in         |            u of the fluid upstream the blast            | float | 0.0                |
    | blast_v_in         |            v of the fluid upstream the blast            | float | 0.0                |
    | blast_w_in         |            v of the fluid upstream the blast            | float | 0.0                |
    | blast_u_out        |           u of the fluid downstream the blast           | float | 0.0                |
    | blast_v_out        |           v of the fluid downstream the blast           | float | 0.0                |
    | blast_w_out        |           w of the fluid downstream the blast           | float | 0.0                |
    | cop_type           |             type of compoent area in domain             |  int  | 0                  |
    | cop_center_x       |    position of compoent area at X direction in domain    | float | 0.0                |
    | cop_center_y       |    position of compoent area at X direction in domain    | float | 0.0                |
    | cop_center_z       |    position of compoent area at X direction in domain    | float | 0.0                |
    | cop_radius         | radius ratio of shortest edge of domain of compoent area | float | 0.0                |
    | bubble_boundary_cells  | number of cells of bubble boundary | float | 2                |
    | bubble_boundary_width  | bubble boundary coffent | float | BlSz.mx * BlSz.X_inner * bubble_boundary                |
    | bubble_shape_x     | reshape bubble to ellipse at X DIR: x*x/bubble_shape_x*bubble_shape_x | float | 0.4*min(Domain_length,Domain_heigh,Domain_width) |
    | bubble_shape_ratioy         | bubble_shape_y=bubble_shape_x/bubble_shape_ratioy | float | 1.0                |
    | bubble_shape_ratioz         | bubble_shape_z=bubble_shape_x/bubble_shape_ratioz | float | 1.0                |
    | bubble_shape_y         | reshape bubble to ellipse at Y DIR: x*x/bubble_shape_y*bubble_shape_y |  float | bubble_shape_x/bubble_shape_ratioy                |
    | bubble_shape_z         | reshape bubble to ellipse at Z DIR: x*x/bubble_shape_y*bubble_shape_y |  float | bubble_shape_x/bubble_shape_ratioz                |
    | bubble_boundary         | number of cells for bubble boundary | int | 3                |
    | bubble_C           | another bubble boundary method argu| float | Domain_length *BlSz.X_inner* 5.0               |
    | cop_density_in     |            rho of the fluid in compoent area            | float | blast_density_out  |
    | cop_pressure_in    |             P of the fluid in compoent area             | float | blast_pressure_out |
    | cop_tempreture_in  |            T of the fluid in compoent area            | float | blast_tempreture_in |

## 7. output data format

- #### set option "OUT_PLT" or "OUT_VTI" ON, mpi support added both of them, both of them can be output

### 7.1. tecplot file

- import .plt files of all ranks of one Step for visualization, points overlapped between boundarys of ranks  

### 7.2. vtk file

- use `paraview` to open `*.pvti` files for visualization;
- use `paraview`  to save data to `Xdmf Data file (*.xmf)`, then `tec360` can open `*.h5` files in the same output directory, select all datasets (each corresponds to a variables, e.g. x, y, z, rho ...)