#pragma once

#if defined(__ACPP_ENABLE_CUDA_TARGET__)
#include <cuda_runtime.h>
#endif

#if defined(__ACPP_ENABLE_HIP_TARGET__)
// #define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

// =======================================================
// //    sycL_Kernel attribute
#if defined(DEFINED_ONEAPI)
#define SYCL_KERNEL SYCL_EXTERNAL
#define SYCL_DEVICE
#define __LBMt 256

#elif defined(__ACPP__)
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

#else

#define SYCL_KERNEL
#define SYCL_DEVICE
#define _VENDOR_KERNEL_LB_(A, B)

#endif
// =======================================================
