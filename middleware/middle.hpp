#pragma once
/**
 * This is a simple middleware for CUDA and SYCL
 */
// =======================================================
// include headers
// =======================================================

#if defined(MIDDLE_CUDA_ENABLED) // target middleware to CUDA backend
#include "cuda_backend/middle_cuda.cuh"
#elif defined(MIDDLE_ROCM_ENABLED) // target middleware to ROCm backend
#include "rocm_backend/middle_rocm.hpp"
#else // target middleware to SYCL backend
#include "sycl_backend/middle_sycl.hpp"
#endif
