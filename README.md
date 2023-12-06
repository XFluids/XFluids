# XFLUIDS

## 1. Dependencies before cmake

### 1.1. IF USE Intel oneAPI

- #### [libboost_filesystem](https://www.boost.org/users/history/version_1_83_0.html)
- #### [intel oneapi version &gt;= 2023.0.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline)
- #### [codeplay Solutions for NVIDIA and AMD backends](https://codeplay.com/solutions/oneapi/)
- #### activate environment for oneAPI appended codeplay sultion libs

  ````bash
  source /opt/intel/oneapi/setvars.sh  --force --include-intel-llvm
  # or you can use the script files(only basic environments are included)
  source ./scripts/opeAPI/oneapi_base.sh
  ````
- #### Device discovery: exec "sycl-ls" in cmd for device counting

```cmd
    $sycl-ls
    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2022.15.12.0.01_081451]
    [opencl:cpu:1] Intel(R) OpenCL, AMD Ryzen 7 5800X 8-Core Processor              3.0 [2022.15.12.0.01_081451]
    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA T600 0.0 [CUDA 11.5]
```

### 1.2. IF USE AdaptiveCpp(known as OpenSYCL/hipSYCL)

- #### [boost-version-1.83](https://www.boost.org/users/history/version_1_83_0.html)
- #### [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), [different backends need different dependencies](https://github.com/jiuxiaocloud/uconfig/blob/master/3.7-opensycl(based%20boost).md)
- #### add libs and includes of boost-version-1.83, AdaptiveCpp and dependencies to ENV PATHs
- #### XFLUIDS use find_package(AdaptiveCpp) to find AdaptiveCpp and use app sycl compile system, set cmake option AdaptiveCpp_DIR

  ````cmake
  cmake -DAdaptiveCpp_DIR=/path/to/AdaptiveCpp/lib/cmake/AdaptiveCpp ..
  ````
- #### Device discovery: exec "acpp-info" in cmd for device counting

## 2. Select target device in SYCL project

- set integer platform_id and device_id for targetting different backends

  ````C++
  auto device = sycl::platform::get_platforms()[platform_id].get_devices()[device_id];
  sycl::queue q(device);
  ````

## 3. Compile and usage of this project

### 3.1. Read $/CMakeLists.txt

- CMAKE_BUILD_TYPE is set to "Release" by default, SYCL code would target to host while ${CMAKE_BUILD_TYPE}==Debug
- set INIT_SAMPLE as the problem being tested, path to "species_list.dat" should be given to MIXTURE_MODEL
- MPI and AWARE-MPI support added in project, AWARE_MPI need specific GPU-ENABLED mpi version, details referenced in [4-mpi-libs]("4. MPI libs")
- tempreture(T) approximately beside and below 200 may cause NAN errors: T must be enough high

### 3.2. BUILD

````bash
    cd ./XFLUIDS
    mkdir build && cd ./build && cmake .. && make -j
````

### 3.3. RUN

- XFLUIDS automatically read <${workspaceFolder}/settings/*.json> file depending on INIT_SAMPLE setting in ${workspaceFolder}/CMakeLists.txt

#### 3.3.1. Local machine running

  ````bash
    $./XFLUIDS
  ````
- Append options to XFLUIDS in cmd for another settings, all options are optional, all options are listed in [6. executable file options]()

  ````bash
    $ ./XFLUIDS -dev=1,1,0
    $ mpirun -n mx*my*mz ./XFLUIDS -mpi=mx,my,mz -dev=1,0,0
  ````

#### 3.3.2. Slurm sbatch running on Hygon(KunShan) supercompute center

  ````bash
    $ cd ./XFLUIDS/scripts/KS-DCU
    $ sbatch ./1node.slurm
    $ sbatch ./2node.slurm
  ````

## 4. MPI libs

### 4.1. Set MPI_PATH browsed by cmake before build

- cmake system of this project browse libmpi.so automatically in path of ${MPI_PATH}/lib, please export MPI_PATH to the mpi you want:

  ````cmd
  export MPI_PATH=/home/ompi
  ````

### 4.2. The value of MPI_HOME, MPI_INC, path of MPI_CXX(libmpi.so) output on screen while libmpi.so is found

````cmake
    -- MPI settings:
    --   MPI_HOME:/home/ompi
    --   MPI_INC: /home/ompi/include added
    --   MPI_CXX lib located: /home/ompi/lib/libmpi.so found
````

## 5. .json configure file arguments

- reading commits in src file: ${workspaceFolder}/src/read_ini/settings/read_json.h

## 6. Executable file options

- #### Set "OutDAT", "OutVTI" as 1 in .ini file

  | name of options  |                         function                                                        | type  |
  | :--------------- | :-------------------------------------------------------------------------------------: | :---: |
  | -blk             |  dim_blk_x, dim_blk_y, dim_blk_z,DtBlockSize(if given)                                  |  int  |
  | -mpi             |  mpi cartesian size: mx,my,mz                                                           |  int  |
  | -dev             |  device counting and selecting: device munber,platform,device                           |  int  |
  | -run             |  domain resolution and running steps: X_inner,Y_inner,Z_inner,nStepmax(if given)        |  int  |

## 7. Uutput data format

- #### Set "OutDAT", "OutVTI" as 1 in .ini file

### 7.1. Tecplot file

- import .dat files of all ranks of one Step for visualization, points overlapped between boundarys of ranks(3D parallel tecplot format file visualization is not supportted, using tecplot for 1D visualization is recommended)

### 7.2. VTK file

- use `paraview` to open `*.pvti` files for MPI visualization(1D visualization is not allowed, using paraview for 2/3D visualization is recommended);
