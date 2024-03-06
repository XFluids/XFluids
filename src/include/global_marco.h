#pragma once
// =======================================================
// // repeated code definitions
// =======================================================

// =======================================================
// //    Global sycL_reduction
#if defined(DEFINED_ONEAPI)
#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<>())

#define SYCL_KERNEL SYCL_EXTERNAL
#define SYCL_DEVICE
#define __LBMt 256

#else
#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<real_t>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<real_t>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<real_t>())

// // OpenSYCL HIP Target
#ifdef __HIPSYCL_ENABLE_HIP_TARGET__
#define __LBMt 256 // <=256 for HIP
using vendorError_t = hipError_t;
using vendorDeviceProp = hipDeviceProp_t;
using vendorFuncAttributes = hipFuncAttributes;

#define _VENDOR_KERNEL_ __global__ __launch_bounds__(256, 1)

#define vendorSuccess hipSuccess;
#define vendorSetDevice(A) hipSetDevice(A)
#define vendorGetLastError() hipGetLastError()
#define vendorDeviceSynchronize() hipDeviceSynchronize()
#define vendorFuncGetAttributes(A, B) hipFuncGetAttributes(A, B)
#define vendorGetDeviceProperties(A, B) hipGetDeviceProperties(A, B)

#define CheckGPUErrors(call)                                                             \
    {                                                                                    \
        hipError_t hipStatus = call;                                                     \
        if (hipSuccess != hipStatus)                                                     \
        {                                                                                \
            fprintf(stderr,                                                              \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "           \
                    "with "                                                              \
                    "%s (%d).\n",                                                        \
                    #call, __LINE__, __FILE__, hipGetErrorString(hipStatus), hipStatus); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }                                                                                    \
    while (0)

// // OpenSYCL CUDA Target
#elif defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
#define __LBMt 256 // <=256 for HIP
using vendorError_t = cudaError_t;
using vendorDeviceProp = cudaDeviceProp;
using vendorFuncAttributes = cudaFuncAttributes;

#define _VENDOR_KERNEL_ __global__

#define vendorSuccess cudaSuccess;
#define vendorSetDevice(A) cudaSetDevice(A)
#define vendorGetLastError() cudaGetLastError()
#define vendorDeviceSynchronize() cudaDeviceSynchronize()
#define vendorFuncGetAttributes(A, B) cudaFuncGetAttributes(A, B)
#define vendorGetDeviceProperties(A, B) cudaGetDeviceProperties(A, B)

#define CheckGPUErrors(call)                                                                \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
        {                                                                                   \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }                                                                                       \
    while (0)

#else
#define __LBMt 256 // <=256 for HIP
#define SYCL_KERNEL
#define SYCL_DEVICE

#endif

#if defined(__HIPSYCL_ENABLE_HIP_TARGET__) || (__HIPSYCL_ENABLE_CUDA_TARGET__)
#define SYCL_KERNEL __host__ __device__
#define SYCL_DEVICE __host__ __device__
#define _VENDOR_KERNEL_LB_(A, B) __global__ __launch_bounds__(A, B)
#endif

#endif // end oneAPI or OpenSYCL

// =======================================================
//    Set Domain size
#define MARCO_DOMAIN()        \
    int Xmax = bl.Xmax;       \
    int Ymax = bl.Ymax;       \
    int Zmax = bl.Zmax;       \
    int X_inner = bl.X_inner; \
    int Y_inner = bl.Y_inner; \
    int Z_inner = bl.Z_inner;

#define MARCO_DOMAIN_GHOST()    \
    int Xmax = bl.Xmax;         \
    int Ymax = bl.Ymax;         \
    int Zmax = bl.Zmax;         \
    int X_inner = bl.X_inner;   \
    int Y_inner = bl.Y_inner;   \
    int Z_inner = bl.Z_inner;   \
    int Bwidth_X = bl.Bwidth_X; \
    int Bwidth_Y = bl.Bwidth_Y; \
    int Bwidth_Z = bl.Bwidth_Z;

// =======================================================
//    get Roe values insde Reconstructflux
#define MARCO_ROE()                               \
    real_t D = sycl::sqrt(rho[id_r] / rho[id_l]); \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));        \
    real_t _u = (u[id_l] + D * u[id_r]) * D1;     \
    real_t _v = (v[id_l] + D * v[id_r]) * D1;     \
    real_t _w = (w[id_l] + D * w[id_r]) * D1;     \
    real_t _H = (H[id_l] + D * H[id_r]) * D1;     \
    real_t _P = (p[id_l] + D * p[id_r]) * D1;     \
    real_t _rho = sycl::sqrt(rho[id_r] * rho[id_l]);

// =======================================================
//    Get c2
#ifdef COP
#define MARCO_GETC2() MARCO_COPC2()
// MARCO_COPC2() //MARCO_COPC2_ROB()
#else
#define MARCO_GETC2() MARCO_NOCOPC2()
#endif // end COP

// //    Get error out of c2 arguments
#if ESTIM_OUT

#define MARCO_ERROR_OUT()                   \
    eb1[id_l] = b1;                         \
    eb3[id_l] = b3;                         \
    ec2[id_l] = c2;                         \
    for (size_t nn = 0; nn < NUM_COP; nn++) \
    {                                       \
        ezi[nn + NUM_COP * id_l] = z[nn];   \
    }
#else

#define MARCO_ERROR_OUT() ;

#endif // end ESTIM_OUT

// =======================================================
//    Loop in Output
#define MARCO_OUTLOOP                                        \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)                \
        for (int j = VTI.minY; j < VTI.maxY; j++)            \
            for (int i = VTI.minX; i < VTI.maxX; i++)

#define MARCO_POUTLOOP(BODY)                          \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)         \
        for (int j = VTI.minY; j < VTI.maxY; j++)     \
        {                                             \
            for (int i = VTI.minX; i < VTI.maxX; i++) \
                out << BODY << " ";                   \
            out << "\n";                              \
        }

#define MARCO_COUTLOOP                                       \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = minZ; k < maxZ; k++)                        \
        for (int j = minY; j < maxY; j++)                    \
            for (int i = minX; i < maxX; i++)

// =======================================================
// end repeated code definitions
// =======================================================
