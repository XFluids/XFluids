# XFLUIDS: A SYCL-based unified cross-architecture heterogeneous simulation solver for compressible reacting flows

XFLUIDS is a parallelized SYCL C++ solver for large-scale high-resolution simulations of compressible multi-component reacting flows. It is developed by [Prof. Shucheng Pan&#39;s](https://teacher.nwpu.edu.cn/span.html) group at the School of Aeronautics, Northwestern Polytechincal University.

main developers:

- Jinlong Li (ljl66623@mail.nwpu.edu.cn)
- Shucheng Pan (shucheng.pan@nwpu.edu.cn)

other contributors: 
- Yixuan Lian, Renfei Zhang

## References
If you use XFLUIDS for academic aplications, please cite our paper: 

Jinlong Li, Shucheng Pan (2024). XFLUIDS: A SYCL-based unified cross-architecture heterogeneous simulation solver for compressible reacting flows. arXiv:2403.05910. 
(https://arxiv.org/abs/2403.05910)


## Features
- Support CPU, GPU (integrated & discrete), and FPGA without porting the code
- General for multi-vendor devices (Intel/NVIDIA/AMD/Hygon ... )
- High portability, productivity, and performace
- GPU-aware MPI
- Highly optimized kernels & device functions for multicomponent flows and chemical reaction 
- ongoing work: sharp-interface method, curvilinear mesh, turbulence models ...

## Supported GPUs
The following gpus have been tested:
- NVIDIA
  - Data center GPU: A100, P100
  - Gaming GPS: RTX 4090, RTX 3090/3080/3070/3060TI, T600, RTX 1080
- AMD
  - Data center GPU: MI50
  - Gaming GPS: RX 7900XTX, RX 6800XT, Pro VII
- Intel
  - Gaming GPS: ARC A770/A380
  - Integrated GPUs: UHD P630, UHD 750

## 1. Dependencies before cmake

### 1.1. IF USE AdaptiveCpp(known as OpenSYCL/hipSYCL, recommended)

- #### 1.1.1. install [boost-version-1.83](https://www.boost.org/users/history/version_1_83_0.html)(needed by AdaptiveCpp)
- #### 1.1.2. install [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), for how to install AdaptiveCpp: [different backends need different dependencies](https://github.com/jiuxiaocloud/uconfig/blob/master/3.7-opensycl(based%20boost).md)
- #### 1.1.3. add bin, libs and includes of AdaptiveCpp and dependencies to ENV PATHs
- #### 1.1.4. XFLUIDS use find_package(AdaptiveCpp) targetting AdaptiveCpp compile system, set cmake option AdaptiveCpp_DIR

  ````cmake
  cmake -DAdaptiveCpp_DIR=/path/to/AdaptiveCpp/lib/cmake/AdaptiveCpp ..
  ````
- #### Device discovery: exec "acpp-info" in cmd for device counting

  ```cmd
  $ acpp-info
  =================Backend information===================
  Loaded backend 0: OpenMP
    Found device: hipSYCL OpenMP host device
  Loaded backend 1: CUDA
    Found device: NVIDIA GeForce RTX 3070
  =================Device information===================
  ***************** Devices for backend OpenMP *****************
  Device 0:
  General device information:
    Name: hipSYCL OpenMP host device
    Backend: OpenMP
    Vendor: the hipSYCL project
    Arch: <native-cpu>
    Driver version: 1.2
    Is CPU: 1
    Is GPU: 0
  ***************** Devices for backend CUDA *****************
  Device 0:
  General device information:
    Name: NVIDIA GeForce RTX 3070
    Backend: CUDA
    Vendor: NVIDIA
    Arch: sm_86
    Driver version: 12000
    Is CPU: 0
    Is GPU: 1
  ```

### 1.2. IF USE Intel oneAPI (recommended for Intel GPU platform)

- #### 1.1.1.[intel oneapi version &gt;= 2023.0.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline) as compiler
- #### 1.1.2.[codeplay Solutions for NVIDIA and AMD backends](https://codeplay.com/solutions/oneapi/) if GPU targets are needed
- #### 1.1.3.1.activate environment for oneAPI appended codeplay sultion libs

  ````bash
  source /opt/intel/oneapi/setvars.sh  --force --include-intel-llvm
  ````
- #### 1.2.3.2.or you can use the script files(only basic environments are included)

  ````bash
  source ./scripts/oneAPI/oneapi_base.sh
  ````
- #### 1.1.4.[libboost_filesystem](https://www.boost.org/users/history/version_1_83_0.html) as external lib while gcc internal filesystem is missing

- #### Device discovery: exec "sycl-ls" in cmd for device counting

  ```cmd
  $ sycl-ls
  [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2
  [opencl:cpu:1] Intel(R) OpenCL, AMD Ryzen 7 5800X 8-Core Processor 3.0
  [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA T600 0.0 [CUDA 11.5]
  ```

### 2. Select target device in SYCL project

- set integer platform_id and device_id for targetting different backends("DeviceSelect" in json file or option: -dev)

  ````C++
  auto device = sycl::platform::get_platforms()[platform_id].get_devices()[device_id];
  sycl::queue q(device);
  ````

## 3. Compile and usage of this project

### 3.1. Read root <XFLUIDS/CMakeLists.txt>

- `CMAKE_BUILD_TYPE` is set to "Release" by default, SYCL code would target to host while ${CMAKE_BUILD_TYPE}==Debug
- set `INIT_SAMPLE` as the problem being tested, path to "species_list.dat" and "reaction_list.dat" should be given to MIXTURE_MODEL
- MPI and AWARE-MPI support added in project, AWARE_MPI need specific GPU-ENABLED mpi version, details referenced in [4-mpi-libs]("4. MPI libs")
- `VENDOR_SUBMIT` allows throwing some parallism tuning cuda/hip model to their GPU, only supportted by AdaptiveCpp compile environment

### 3.2. BUILD and RUN

- #### 3.2.1.Build with cmake
- build with cmake

  ````bash
  cd ./XFLUIDS
  mkdir build && cd ./build && cmake .. && make -j
  ````
- #### 3.2.2.Local machine running
- XFLUIDS automatically read <XFLUIDS/settings/*.json> file depending on INIT_SAMPLE setting

  ````bash
  ./XFLUIDS
  ````
- Append options to XFLUIDS in cmd for another settings, all options are optional, all options are listed in [6. executable file options]("6. Executable file options")

  ````bash
  ./XFLUIDS -dev=1,1,0
  mpirun -n mx*my*mz ./XFLUIDS -mpi=mx,my,mz -dev=1,0,0
  ````
- #### 3.2.2.Slurm sbatch running on Hygon(KunShan) supercompute center

  ````bash
  cd ./XFLUIDS/scripts/KS-DCU
  sbatch ./1node.slurm
  sbatch ./2node.slurm
  ````

## 4. MPI libs
### 4.1. Set MPI_PATH browsed by cmake before build

- cmake system of this project browse libmpi.so automatically in path of ${MPI_PATH}/lib, please export MPI_PATH to the mpi you want:

  ````cmd
  export MPI_PATH=/home/ompi
  ````
### 4.2. The value of MPI_HOME, MPI_INC, path of MPI_CXX(libmpi.so/linmpicxx.so) output on screen while it is found

````cmake
  -- MPI settings:
  --   MPI_HOME:/home/ompi
  --   MPI_INC: /home/ompi/include added
  --   MPI_CXX lib located: /home/ompi/lib/libmpi.so found
````

## 5. .json configure file arguments

- reading commits in src file: <${workspaceFolder}/src/read_ini/settings/read_json.cpp>

## 6. Executable file options

| name of options |                                       function                                       |    type    |
| :-------------- | :----------------------------------------------------------------------------------: | :---------: |
| -domain         |              domain size                        : length, width, height              |    float    |
| -run            |   domain resolution and running steps: X_inner,Y_inner,Z_inner,nStepmax(if given)    |     int     |
| -blk            | initial local work-group size, dim_blk_x, dim_blk_y, dim_blk_z,DtBlockSize(if given) |     int     |
| -dev            |             device counting and selecting: device munber,platform,device             |     int     |
| -mpi            |                             mpi cartesian size: mx,my,mz                             |     int     |
| -mpi-s          |                                  "weak" or "strong"                                  | std::string |
| -mpidbg         |         append the option with or without value to open mpi multi-rank debug         | just append |

## 7. Output data format

- ###### Set "OutDAT", "OutVTI" as 1 in .json file
### 7.1. Tecplot file

- import .dat files of all ranks of one Step for visualization, points overlapped between boundarys of ranks(3D parallel tecplot format file visualization is not supportted, using tecplot for 1D visualization is recommended)

### 7.2. VTK file

- use `paraview` to open `*.pvti` files for MPI visualization(1D visualization is not allowed, using paraview for 2/3D visualization is recommended);

## Cite XFLUIDS
```
@misc{li2024xfluids,
      title={XFLUIDS: A SYCL-based unified cross-architecture heterogeneous simulation solver for compressible reacting flows}, 
      author={Jinlong Li and Shucheng Pan},
      year={2024},
      eprint={2403.05910},
      archivePrefix={arXiv}
}
```

## Acknowledgments

XFLUIDS has received financial support from the following fundings:
- The Guanghe foundation (Grant No. ghfund202302016412)
- The National Natural Science Foundation of China (Grant No. 11902271)