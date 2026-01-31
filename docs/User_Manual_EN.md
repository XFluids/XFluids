# XFLUIDS v0.1 User Manual

## 1. Installation and environment configuration

This project provides two SYCL-based implementations: **Intel oneAPI** and **AdaptiveCpp**. To simplify complex dependency configurations, automated shell scripts are provided in the project root to automatically download and deploy the necessary toolchains based on the current hardware environment (NVIDIA/AMD/Intel GPU).

### 1.1 SYCL implementation selection

|  | Intel oneAPI | AdaptiveCpp |
| :--- | :--- | :--- |
| **Install Size** | Large (Approx. 2.3GB) | Small (< 10MB) |
| **Compiler Dependency** | Includes `icpx` / `clang++` | **Requires an LLVM environment** |
| **Adaptability** | Supports isolated installation; good compatibility. | Requires installation via Conda environment if system-level LLVM is unavailable. |
| **GPU Backend Support** | Highly optimized for Intel devices; supports NVIDIA/AMD (via Codeplay plugins). | Supports NVIDIA (CUDA), AMD (ROCm), Intel (SYCL) and **Generic (SSCP)** modes. |
| **Extra Features** | Includes MPI and Intel math libraries (e.g. oneMKL); supports Intel performance analysis tools (e.g. VTune and Advisor). | Requires manual installation of MPI and other math libraries; No performance analysis tools. |

> **Special Note**: On platforms without `sudo` privileges (such as supercomputers), if native LLVM cannot be installed via `apt`, the AdaptiveCpp SYCL implementation must rely on the project-provided Conda environment to deploy the LLVM toolchain.

### 1.2 Download

Clone the XFluids repository from GitHub:

```bash
git clone https://github.com/XFluids/XFluids.git
```

Then, depending on the SYCL implementation selected in Sec. 1.1, execute the installation script (`run_install_oneAPI.sh` or `run_install_AdaptiveCpp.sh`) to download the required dependencies:

*   **Resource acquisition**:
    The installation script automatically downloads specific versions of dependencies from GitHub Releases to the local `external/downloads` directory.
    
    > **Verification mechanism**: Before downloading, the script checks local files and verifies integrity using **SHA256 checksums**. If the file exists and the checksum matches, the download is skipped. This prevents failures of downloading in unstable network environments.

*   **Backend auto-detection**:
    The installation script detects if `nvcc` (or `/usr/local/cuda`) for NVIDIA, or `hipcc` (or `/opt/rocm`) for AMD exists in the system.
    
    *   If an NVIDIA GPU is detected, the CUDA backend is configured.
    *   If an AMD GPU is detected, the ROCm backend is configured.
    *   If neither is detected, the Intel Level Zero/OpenCL backend is configured by default.
    
    > **Special Note**: For hybrid mode (in Sec. 5), users need to manually configure the device backends according to the specific combinations of different vendors in their systems.

#### Implementation A: Intel oneAPI
Based on the official Intel toolchain, supporting NVIDIA/AMD GPUs via Codeplay plugins.

> For relative old devices (e.g. NVIDIA P100), one need to use old  version of oneAPI (2024.0.0) and icpx (instead of clang++)  to compile XFLUIDS, to improve the computational efficiency. 

*   **Installation process**:
    
    1.  The script automatically installs Intel oneAPI Base Toolkit (2025.0).
    2.  **Plugin adaptation**: If a non-Intel GPU is detected, it automatically installs plugins provided by Codeplay (currently supports CUDA 12.0 or ROCm 5.4/6.1).
    3.  Compile and install the Boost 1.83 library using the Intel `clang++` compiler.
*   **Installation command**:
    
    ```bash
    chmod +x run_install_oneAPI.sh
    ./run_install_oneAPI.sh
    ```
    ![oneAPI Installation Process](./Figs/oneAPI/run_install_oneAPI.png)

#### Implementation B: AdaptiveCpp
Achieves isolation via a Conda environment, allowing LLVM deployment without `sudo` privileges.
*   **Installation process**:
    
    1.  **Conda Isolation**: Automatically installs Miniconda and creates an independent environment named `XFLUIDS`.
    2.  **LLVM Deployment**: Installs the LLVM/Clang 16 toolchain within the Conda environment to avoid contaminating the host system.
    3.  **Boost Compilation**: Compiles Boost 1.83 using the Conda-integrated `clang++`.
    4.  **AdaptiveCpp**: Compiles and installs the AdaptiveCpp runtime based on the detected backend.
*   **Installation command**:
    
    ```bash
    chmod +x run_install_AdaptiveCpp.sh
    ./run_install_AdaptiveCpp.sh
    ```
    ![AdaptiveCpp Installation Process](./Figs/AdaptiveCpp/run_install_ACPP.png)

## 2. Project build

After dependency installation is complete, the installation script (`run_install_oneAPI.sh` or `run_install_AdaptiveCpp.sh`) will generate the corresponding environment loading script (`XFLUIDS_oneAPI_setvars.sh` or `XFLUIDS_AdaptiveCpp_setvars.sh`) and the build script (`run_build_oneAPI.sh` or `run_build_AdaptiveCpp.sh`) in the root directory.

### 2.1 Running build scripts

Run the generated build scripts for standard compilation (Release mode):

*   **oneAPI implementation**:
    
    The build script for the oneAPI implementation compiles the CUDA version by default. If you need to compile for ROCm/Intel/CPU, you must manually modify the `SelectDv` value in `CMakeLists.txt` to `hip`, `intel`, or `host`. Additionally, the script defaults `ARCH` to `86`; you need to modify the `ARCH` value in `CMakeLists.txt` according to your specific device.
    
    ```bash
    ./run_build_XFLUIDS_oneAPI.sh
    ```
    ![Start oneAPI Build Script](./Figs/oneAPI/run_build_oneAPI_0.png)
    
    ![oneAPI Build Complete](./Figs/oneAPI/run_build_oneAPI_3.png)
    
*   **AdaptiveCpp implementation**:
    
    The build script for the AdaptiveCpp implementation compiles the `generic` version by default. No manual modification of the `CMakeLists.txt` file is required.
    
    ```bash
    ./run_build_XFLUIDS_AdaptiveCpp.sh
    ```
    ![Start ACPP Build Script](./Figs/AdaptiveCpp/run_build_ACPP_0.png)
    
    ![ACPP Build Complete](./Figs/AdaptiveCpp/run_build_ACPP_3.png)

After successful compilation, the binary file `XFLUIDS` will be located in the `./build` directory.

### 2.2 Loading Environment Variables

The build scripts load environment variables internally. However, for manual compilation or running the executable, one **must** load the corresponding environment variables:

*   **oneAPI Implementation**:
    
    ```bash
    source XFLUIDS_oneAPI_setvars.sh
    ```
    ![Load oneAPI Environment](./Figs/oneAPI/source_oneAPI.png)
    
*   **AdaptiveCpp Implementation**:
    
    ```bash
    source XFLUIDS_AdaptiveCpp_setvars.sh
    ```
    ![Load ACPP Environment](./Figs/AdaptiveCpp/source_ACPP.png)

## 3. Program execution

### 3.1 Running examples
*   **oneAPI Implementation**:
    
    ```bash
    cd build
    source ../XFLUIDS_oneAPI_setvars.sh
    ./XFLUIDS -dev=1,2,0
    ```
    ![oneAPI Run Example](./Figs/oneAPI/run_XFLUIDS_oneAPI.png)

    Monitor GPU load via `watch -n 0 nvidia-smi` to ensure the program is running on the specified GPU.
    ![oneAPI Load Monitor](./Figs/oneAPI/watch_nvidia-smi_oneAPI.png)

*   **AdaptiveCpp Implementation**:
    
    ```bash
    cd build
    source ../XFLUIDS_AdaptiveCpp_setvars.sh
    ./XFLUIDS -dev=1,1,0
    ```
    ![ACPP Run Example](./Figs/AdaptiveCpp/run_XFLUIDS_ACPP.png)

    Monitor GPU load via `watch -n 0 nvidia-smi` to ensure the program is running on the specified GPU.
    ![ACPP Load Monitor](./Figs/AdaptiveCpp/watch_nvidia-smi_ACPP.png)

**CPU Multithreading**:

If the CPU is selected as the running device, both oneAPI and AdaptiveCpp SYCL implementations will use multi-threaded parallelism. As shown below, XFLUIDS runs in multi-threaded mode on an 8-core AMD 5800X (with hyper-threading disabled).
    ![CPU Multi-thread Monitor](./Figs/CPU_Multi_thread.png)

### 3.2 Runtime parameters

*   **Device selection**: Specified via the `-dev` argument.
    
    ```bash
    ./XFLUIDS -dev=1,1,0  # 1st arg: number of devices; 2nd arg: platform ID; 3rd arg: device ID
    ```
*   **Resolution and running steps**:
    
    ```bash
    ./XFLUIDS -run=2048,2048,0,50 # Resolutions for X, Y, Z directions, and number of steps
    ```
*   **MPI multi-device execution**:
    
    ```bash
    mpirun -n mx*my*mz ./XFLUIDS -mpi=mx,my,mz
    ```

> Note: The default values of -dev and -run have been written in json files (in the folder `settings`) for each test cases.

### 3.3 Key feature: Adaptive Range Assignment (ARA)

This project includes the **ARA** feature for heterogeneous hardware.
1.  **First Run (Search Phase)**:
    During the initial run, the solver attempts different nd-range parameters for the first few dozen steps. The terminal will display the parameters being tested.
    ![ARA Example](./Figs/ARA.png)
    *   **Requirement**: It is recommended to run at least **50 steps** until the terminal stops, with the tuning information printed. At this point, the optimal configuration has been saved to a binary cache file.
2.  **Second Run (Calculation Phase)**:
    When running the case with the same resolution again, XFLUIDS will directly read the optimal nd-range configuration cache file, achieving maximum performance.

> **Note**: In hybrid computation mode, although this feature is partially overridden, it is still necessary to run twice. The first run only needs **2 steps** to generate the necessary cache file, after which the second run can be performed.

## 4. Examples
After completing the dependency installation and the initial compilation of XFLUIDS, two case configuration scripts are provided in the `example` directory. Running the case configuration script will modify the case settings in `CMakeLists.txt`. After re-running the build script, users can execute `XFLUIDS` in the `build` directory.

### 4.1 1d-insert-st

```bash
cd example/1d-insert-st
chmod 777 activate_case.sh
./activate_case.sh

cd ../..
./run_build_oneAPI.sh # For oneAPI version
# ./run_build_AdaptiveCpp.sh # For AdaptiveCpp version
cd build/
source ../XFLUIDS_oneAPI_setvars.sh # For oneAPI version
#source ../XFLUIDS_AdaptiveCpp_setvars.sh # For AdaptiveCpp version

./XFLUIDS
```

### 4.2 2d-euler-vortex

```bash
cd example/2d-euler-vortex
chmod 777 activate_case.sh
./activate_case.sh

cd ../..
./run_build_oneAPI.sh # For oneAPI version
# ./run_build_AdaptiveCpp.sh # For AdaptiveCpp version
cd build/
source ../XFLUIDS_oneAPI_setvars.sh # For oneAPI version
#source ../XFLUIDS_AdaptiveCpp_setvars.sh # For AdaptiveCpp version

./XFLUIDS
```

## 5. Hybrid heterogeneous computation

This chapter introduces the CPU (Host) + GPU (Device) heterogeneous fusion computation mode based on **Intel oneAPI**.
> **Note**: To demonstrate this feature, the `hybrid` branch contains handcoded implementations specifically for the **AMD Ryzen 9 9950X + NVIDIA RTX 3080** desktop. To port to other hardware, source code modifications regarding resolution and load partitioning are required.

### 5.1 Device and load distribution

To simplify the setup, in the `hybrid` branch, XFLUIDS partitions the computational domain in the **Y-direction**:
*   **Rank 0 (GPU)**: Responsible for invoking the GPU for parallel computation.
    *   **Host Thread Binding**: Pinned to **Physical Core 0** (Single thread manages the GPU).
*   **Rank 1 (CPU)**: Utilizes the multi-core performance of the CPU for computation.
    *   **Host Thread Binding**: Pinned to **Physical Cores 1-15** (Using 15 threads for calculation).

### 5.2 Dependency download and installation

This is identical to the oneAPI implementation download and installation described above; it can be installed using the script file.

### 5.3 Compiling 

Hybrid mode requires manual CMake configuration to enable the `ENABLE_HYBRID` macro.

```bash
# Load Environment
source XFLUIDS_oneAPI_setvars.sh

# Create Build Directory
mkdir build_hybrid && cd build_hybrid

# Manual Compilation (Modify BOOST_ROOT according to your actual path)
cmake .. \
    -DENABLE_HYBRID=ON \
    -DUSE_MPI=ON \
    -DTEST_CASE=3 \
    -DBOOST_ROOT=/your/path/to/external/install/boost \
    -DCMAKE_EXE_LINKER_FLAGS="-L/your/path/to/external/install/boost/lib -lboost_filesystem -lboost_system"

make -j
```
![Hybrid Example 0](./Figs/Hybrid/[TEST_CASE=3]run_0.png)

![Hybrid Example 1](./Figs/Hybrid/[TEST_CASE=3]run_1.png)

### 5.4 Running and test cases (TEST_CASE)

Coordinate the `TEST_CASE` macro with the `launch.sh` script to switch between three test modes.

*   **CASE 1: Pure CPU Mode (9950X)**
    
    ```bash
    # CMake Argument: -DTEST_CASE=1
    mpirun -n 1 ./XFLUIDS
    ```
    ![Hybrid_TEST_CASE=1 Example](./Figs/Hybrid/[TEST_CASE=1]run.png)

    Monitor CPU load using `top`.
    ![Hybrid_TEST_CASE=1 Monitor](./Figs/Hybrid/[TEST_CASE=1]watch-top.png)

*   **CASE 2: Pure GPU Mode (RTX 3080)**
    
    ```bash
    # CMake Argument: -DTEST_CASE=2
    mpirun -n 1 ./XFLUIDS
    ```
    ![Hybrid_TEST_CASE=2 Example](./Figs/Hybrid/[TEST_CASE=2]run.png)

*   **CASE 3: Hybrid Computation Mode (GPU + CPU)**
    Launch two processes. The script automatically pins Rank 0 to the GPU and Rank 1 to CPU computation.
    
    > Note: For AMD CPUs, oneAPI's `clang++` uses only `num_cores/number_ranks` processes for CPU computation in its MPI multi-process environment. To ensure full CPU utilization, the `launch.sh` script identifies the MPI Rank and performs `taskset` for core binding.
    
    ```bash
    # CMake Argument: -DTEST_CASE=3
    mpirun -n 2 -genv I_MPI_PIN 0 ../launch.sh ./XFLUIDS
    ```
    
    ![Hybrid_TEST_CASE=3 Example](./Figs/Hybrid/[TEST_CASE=3]run_4.png)
    
    ![Hybrid_TEST_CASE=3 Result](./Figs/Hybrid/[TEST_CASE=3]run_9.png)

Monitor CPU/GPU load via `top` and `watch -n 0 nvidia-smi`. 

> Clearly, the first XFLUIDS process is executed on **CPU (with 15 threads)**, and the second XFLUIDS process is executed on **GPU (managed by one CPU thread)**. 

![Hybrid_TEST_CASE=3 Monitor 0](./Figs/Hybrid/[TEST_CASE=3]watch-top.png)

![Hybrid_TEST_CASE=3 Monitor 1](./Figs/Hybrid/[TEST_CASE=3]watch-nvidia-smi.png)

### 5.5 Source modification guide (adapting to other desktops)

If users need to change the device configuration or case resolution, they must modify the following locations:

1.  **Device binding (`src/read_ini/src/constructor.cpp`)**:
    In the `Setup` constructor, modify the index of `get_platforms()[id]` based on `TEST_CASE` to match your devices list (use `sycl-ls`).
2.  **Domain partitioning (`src/read_ini/src/iniset.cpp`)**:
    *   `HybridReWrite` function: Define `BlSz.Y_inner` (subdomain height). For example: Rank 0 (GPU) assigned 1760 rows, Rank 1 (CPU) assigned 800 rows.
    *   `init` function: Overwrite global physical dimensions `global_domain_len_y` and resolution `global_resolution_y`.
3.  **Case settings (`CMakeLists.txt` && `settings/2d-euler-vortex.json`)**:
    One need to modify the `INIT_SAMPLE` field in `CMakeLists.txt`, and simultaneously modify the `Resolution` field in the corresponding case Json file to match the total resolution in the source code (e.g., `[2560, 2560, 0]`).