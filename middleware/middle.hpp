#pragma once
/**
 * This is a simple middleware for CUDA and SYCL
 */
// =======================================================
// include headers
// =======================================================

#ifdef MIDDLE_CUDA_ENABLED // target middleware to CUDA backend
#include "cuda_backend/middle_cuda.cuh"
#endif

#ifdef MIDDLE_ROCM_ENABLED // target middleware to ROCm backend
#include "rocm_backend/middle_rocm.hpp"
#endif

#ifdef MIDDLE_SYCL_ENABLED // target middleware to SYCL backend
#include "sycl_backend/middle_sycl.hpp"
#endif