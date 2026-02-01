# XFluids Configuration & Validation Guide

本文档旨在说明 XFluids 项目的编译选项与运行参数，并记录论文中各验证算例的具体配置。

## 1. Project Options

### 1.1 CMake Build Options
这些选项用于配置构建过程，决定了求解器的物理模型、数值格式和硬件后端。

#### General Settings
| Option | Default | Description |
| :--- | :--- | :--- |
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型：`Release` (优化), `Debug` (调试)。 |
| `SYCL_COMPILE_SYSTEM`| `oneAPI` | 编译模型：`oneAPI` (Intel oneAPI) 或 `ACPP` (AdaptiveCpp)。 |
| `VENDOR_SUBMIT`| `OFF` | 采用厂商原生编译模型，比如对于NVIDIA设备采用CUDA。 |
| `SelectDv` | `cuda` | 目标设备架构：oneAPI选择`host` (CPU), `cuda` (NVIDIA GPU), `hip` (AMD GPU), `intel`(Intel GPU); ACPP选择 `generic`。 |
| `ARCH` | `86` | 目标设备的Compute Capability号。 |
| `USE_DOUBLE` | `ON` | 精度控制：`ON` 使用双精度浮点数 (double)，`OFF` 使用单精度 (float)。 |
| `EXPLICIT_ALLOC` | `ON` | MPI 内存管理：`ON` 显式分配设备缓冲区并传输；`OFF` 在 Host 端分配结构体指针。 |
| `USE_MPI` | `OFF` | 并行计算：是否启用 MPI 多卡/多节点并行。 |
| `AWARE_MPI` | `OFF` | 是否启用 CUDA-Aware MPI (仅在 `EXPLICIT_ALLOC` 为 ON 时有效)。 |
| `INIT_SAMPLE` | `1d-insert-st` | 算例选择：指定初始化逻辑与默认配置文件（默认配置文件见 init_sample.cmake）。 |
| `MIXTURE_MODEL` | `1d-mc-insert-shock-tube` | 组分模型路径：指向 ./runtime.dat/ 下的物质属性与化学反应机理文件。|
| `ESTIM_NAN` | `OFF` | 捕捉NaN、 无穷大、负密度的错误。 |
| `ERROR_OUT` | `OFF` | 屏幕输出错误。 |

#### Numerical Methods (Numerics)

| Option | Default | Description |
| :--- | :--- | :--- |
| `DISCRETIZATION_METHOD`| `FDM` | 离散化方法：`FDM` (有限差分) 或 `FVM` (有限体积)。 |
| `WENO_ORDER` | `5` | 无粘通量的重构精度：`5` (WENO5), `6` (WENO-CU6), `7` (WENO7)。 |
| `VISCOSITY_ORDER` | `Fourth` | 粘性通量的离散阶数 |
| `RIEMANN_SOLVER` | `HLLC` | 黎曼求解器 (仅 FVM)：`HLLC`, `AUSM`。 |
| `EIGEN_SYSTEM` | `positive-definite`| 特征系统求解策略 (仅 FDM)：<br>• `positive-definite`: 求解 **N-1** 个组分方程 (归一化约束)，保证正定性。<br>• `overdetermined`: 求解全 **N** 个组分方程，处理超定系统。 |
| `EIGEN_ALLOC` | `OROC` | 特征矩阵内存策略：`OROC` (One Row One Column, 寄存器优化), `RGIF` (Register Global Interface)。 |
| `ARTIFICIAL_VISC_TYPE`| `GLF` | 人工粘性类型 (激波捕捉)：<br>• `ROE`: Roe 类型。<br>• `LLF`: 局部 Lax-Friedrichs。<br>• `GLF`: 全局 Lax-Friedrichs。 |
| `POSITIVITY_PRESERVING`| `OFF` | 保正性：是否强制保证密度/压力为正值 (针对强激波/真空问题)。 |

#### Physical Models
| Option | Default | Description |
| :--- | :--- | :--- |
| `COP` | `ON` | 多组分模型 (Component)：是否启用多组分流动。 |
| `COP_CHEME` | `OFF` | 化学反应：是否启用化学反应源项 (`OFF` 为惰性气体)。 |
| `Visc` | `OFF` | 物理粘性：是否启用 Navier-Stokes 粘性项。 |
| `Visc_Heat` | `OFF` | 热传导：是否启用热通量项 (需开启 `Visc`)。 |
| `Visc_Diffu` | `OFF` | 组分扩散：是否启用扩散项 (需开启 `Visc` 和 `COP`)。 |
| `THERMAL` | `NASA` | 热力学数据拟合格式：`NASA` 或 `JANAF`。 |

---

### 1.2 JSON Runtime Configuration
JSON 配置文件控制运行时的网格、时间步进、I/O 和初始化条件。

#### `run` (Time Stepping & Output)
| Parameter | Description |
| :--- | :--- |
| `DtBlockSize` | 计算时间步长 (dt) 时的线程块大小 (Kernel Block Size)。 |
| `blockSize_[x,y,z]` | 求解器计算网格时的线程块尺寸配置。 |
| `CFLnumber` | CFL 数，控制时间步长稳定性。 |
| `nStepMax` | 最大计算步数。 |
| `OutBoundary` | 输出控制：`1` 包含 Ghost Cells (边界网格)，`0` 仅输出内部计算域。 |
| `OutTimeStamps` | 指定时间点的输出控制列表。格式如下：<br>`"Time: {-C=SliceOpt;-V=Vars;-P=Filter}"`<br>• **-C (Cut/Slice)**: 切片或线采样。例如 `X,0.0,0.0` 表示法向为 X 过原点的切片。<br>• **-V (Variables)**: 输出变量列表，如 `rho,P,T,yi[H2]`。<br>• **-P (Predicate)**: 过滤器，如 `yi[Xe]>0.01` (仅输出满足条件的点)。 |
| `OutTimeArrays` | 定义等间隔的时间序列输出。 |

#### `mpi` (Parallelization)
| Parameter | Description |
| :--- | :--- |
| `mx`, `my`, `mz` | MPI 在 x, y, z 方向的区域分解数量。 |
| `DeviceSelect` | 设备选择向量 `[Count, PlatformID, DeviceID]`：<br>• 使用的设备数量。<br>• 平台 ID。<br>• 起始设备 ID。 |

#### `mesh` (Grid & Boundary)
| Parameter | Description |
| :--- | :--- |
| `DOMAIN_Size` | 计算域物理尺寸 `[Lx, Ly, Lz]`。 |
| `Resolution` | 网格分辨率 `[Nx, Ny, Nz]`。 |
| `Boundarys` | 6个面的边界条件类型 ID <br>映射关系：<br>• `0`: Inflow (入流)<br>• `1`: Outflow (出流)<br>• `2`: Symmetry (对称)<br>• `3`: Periodic (周期)<br>• `4`: nslipWall (无滑移壁面)<br>• `5`: viscWall (粘性壁面)<br>• `6`: slipWall (滑移壁面) |

#### `init` (Initial Conditions)
| Parameter | Description |
| :--- | :--- |
| `blast_mach` | 初始激波马赫数。 |
| `cop_type` | 组分分布几何形状：<br>• `0`: 1D Set (一维区域分布)。<br>• `1`: Bubble (气泡/球形区域)。 |
| `blast_type` | 激波初始化几何形状：<br>• `0`: 1D Planar Shock (平面激波)。<br>• `1`: Circular/Spherical Shock (圆形/球形激波)。 |
| `blast_center` | 激波中心坐标。 |

---

## 2. Numerical Validation Cases Setup

本节详细列出了论文 **Section 4: Numerical validations** 中各算例对应的配置参数。Sec 1.1和1.2中的option值如果没有在以下算例中设置，将采用默认值。

### 2.1 2D Euler vortex

|           | Option / Parameter | Value / Setting     | Notes              |
| :-------- | :----------------- | :------------------ | :----------------- |
| **CMake** | `INIT_SAMPLE`      | **2d-euler-vortex** |                    |
|           | `MIXTURE_MODEL`    | **NO-COP**          |                    |
|           | `COP`              | **OFF**             | 单组分             |
|           | `COP_CHEME`        | **OFF**             | 惰性气体 (Inert)   |
|           | `Visc`             | **OFF**             | 无粘 (Inviscid)    |
|           | `WENO_ORDER`       | 5                   | 选择WENO5格式      |
| **JSON**  | `mesh.Resolution`  | `[256, 256, 0]`     |                    |
|           | `mesh.DOMAIN_Size` | `[10.0, 10.0, 0.0]` |                    |
|           | `mesh.Boundarys`   | `[3, 3, 3, 3, ...]` | 选择周期性边界条件 |


### 2.2 Multicomponent Inert Shock Tube

*验证多组分对流项与高精度重构格式 (WENO5/CU6/7)。*

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-insert-st** |  |
| | `MIXTURE_MODEL` | **1d-mc-insert-shock-tube** |  |
| | `COP` | **ON** | 多组分 |
| | `COP_CHEME` | **OFF** | 惰性气体 (Inert) |
| | `Visc` | **OFF** | 无粘 (Inviscid) |
| | `WENO_ORDER` | 5, 6, 7 | 对比三种格式 |
| | `EIGEN_SYSTEM` | `positive-definite` | **N-1** 个组分 |
| **JSON** | `mesh.Resolution` | `[400, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.1, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[0, 1, ...]` | 左边界选择入流，右边界选择出流 |

### 2.3 Multicomponent Diffusion
*验证分子输运与热传导项。*

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-diffusion** |  |
| | `MIXTURE_MODEL` | **1d-mc-diffusion** |  |
| | `COP` | **ON** | 多组分 |
| | `Visc` | **ON** | 开启粘性通量 |
| | `Visc_Heat` | **ON** | 开启热传导 |
| | `Visc_Diffu` | **ON** | 开启组分扩散 |
| | `COP_CHEME` | **OFF** | 无反应 |
| **JSON** | `mesh.Resolution` | `[200, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[3, 3, ...]` | 选择周期性边界条件 |

### 2.4 Zero-dimensional Constant-pressure Autoignition
*验证化学反应源项 (ODE Solver)。*

实际是1维设置，在开始计算1维算例之前，会计算零维算例并输出数据文件。

|  | Option / Parameter | Value / Setting              | Notes                                |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE`      | **1d-reactive-st**           |                                      |
|           | `MIXTURE_MODEL`    | **Reaction/H2O_18_reaction** |                                      |
|           | `COP`              | **ON**                       | 多组分                               |
|           | `COP_CHEME`        | **ON**                       | 开启化学反应                         |
|           | `Visc`             | **OFF**                      | 无粘                                 |
| **JSON**  | `mesh.Resolution`  | `[200, 0, 0]`                |                                      |
|           | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]`           |                                      |
|           | `mesh.Boundarys`   | `[4, 1, ...]`                | 左边界选择无滑移壁面，右边界选择出流 |

### 2.5 Reactive Shock Tube
*验证反应源项与流动耦合。*

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **1d-reactive-st** |  |
| | `MIXTURE_MODEL` | **Reaction/H2O_18_reaction** |  |
| | `COP` | **ON** | 多组分 |
| | `COP_CHEME` | **ON** | 开启化学反应 |
| | `Visc` | **OFF** | 无粘 |
| **JSON** | `mesh.Resolution` | `[200, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.05, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[4, 1, ...]` | 左边界选择无滑移壁面，右边界选择出流 |

### 2.6 Planar Steady Detonation Propagation
*验证刚性爆轰波传播。*

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **2d-detonation** |  |
| | `MIXTURE_MODEL` | **Reaction/H2O_18_reaction** |  |
| | `COP` | **ON** | 多组分 |
| | `COP_CHEME` | **ON** | 开启化学反应 |
| | `POSITIVITY_PRESERVING`| **ON** | 开启保正性算法 (处理真空/强间断) |
| | `Visc` | **OFF** | |
| **JSON** | `mesh.Resolution` | `[100000, 0, 0]` |  |
| | `mesh.DOMAIN_Size` | `[0.5, 0.0, 0.0]` |  |
| | `mesh.Boundarys` | `[0, 1, ...]` | 左边界选择入流，右边界选择出流 |

### 2.7 Shock-Bubble Interactions (SBI)
*综合验证：多维、多组分、激波、界面变形、(可选)反应。*

#### 2.7.1 Inert Shock-Bubble Interactions (ISBI)

|  | Option / Parameter | Value / Setting | Notes |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE` | **shock-bubble** |  |
| | `MIXTURE_MODEL` | **Inert-SBI** |  |
| | `COP` | **ON** | 多组分 |
| | `Visc` | **ON** | 开启粘性 |
| | `Visc_Heat` | **ON** | 开启热传导 |
| | `Visc_Diffu` | **ON** | 开启扩散 |
| | `COP_CHEME` | **OFF** | 惰性 |
| | `POSITIVITY_PRESERVING`| **ON** | 开启通量保正方法 |
| | `THERMAL` | `NASA` | 热力学数据拟合格式：`NASA` |
| **JSON** | `mesh.Resolution` | 160 ppr |  |
| | `mesh.Boundarys` | `[0, 1, 2, 1, 2, 1]` |  |
| | `init.blast_type` | `1` |  |
| | `init.cop_type` | `1` | Bubble (气泡) |
| | `init.blast_mach` | `2.83` |  |

#### 2.7.2 Reacitve Shock-Bubble Interactions (RSBI)

|  | Option / Parameter      | Value / Setting         | Notes                                 |
| :--- | :--- | :--- | :--- |
| **CMake** | `INIT_SAMPLE`           | **shock-bubble**        |                                       |
|           | `MIXTURE_MODEL`         | **Reaction/RSBI-18REA** |                                       |
|           | `COP`                   | **ON**                  | 多组分                                |
|           | `Visc`                  | **ON**                  | 开启粘性                              |
|           | `Visc_Heat`             | **ON**                  | 开启热传导                            |
|           | `Visc_Diffu`            | **ON**                  | 开启扩散                              |
|           | `COP_CHEME`             | **ON**                  | 反应                                  |
|           | `POSITIVITY_PRESERVING` | **ON**                  | 开启通量保正方法                      |
|           | `THERMAL`               | `NASA`                  | 热力学数据拟合格式：`NASA`            |
| **JSON**  | `mesh.Resolution`       | 160 ppr                 |     |
|           | `mesh.Boundarys`        | `[0, 1, 2, 1, 2, 1]`    |  |
|           | `init.blast_type`       | `1`       |               |
|           | `init.cop_type`         | `1`                     | Bubble (气泡)                         |
|           | `init.blast_mach`       | `2.83`                  |                              |
