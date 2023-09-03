#pragma once
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>

#define CheckGPUErrors(call)                                                              \
    {                                                                                     \
        cudaError_t gpuStatus = call;                                                     \
        if (cudaSuccess != gpuStatus)                                                     \
        {                                                                                 \
            fprintf(stderr,                                                               \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with "       \
                    "%s (%d).\n",                                                         \
                    #call, __LINE__, __FILE__, cudaGetErrorString(gpuStatus), gpuStatus); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }                                                                                     \
    while (0)
