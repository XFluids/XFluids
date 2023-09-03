#pragma once

#include <string>
#include "error.h"
#include "../utils/utils.h"

namespace middle
{
    using device_t = int;
    using stream_t = cudaStream_t;
    using range_t = dim3; // Data_type for allocate threads
    enum MemCpy_t
    {
        HtH = int(cudaMemcpyHostToHost),     /**< Host   -> Host */
        HtD = int(cudaMemcpyHostToDevice),   /**< Host   -> Device */
        DtH = int(cudaMemcpyDeviceToHost),   /**< Device -> Host */
        DtD = int(cudaMemcpyDeviceToDevice), /**< Device -> Device */
        DeF = int(cudaMemcpyDefault)         /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
    };

    // =======================================================
    // Sync API
    // =======================================================

    inline std::string DevInfo(device_t &device)
    {
        cudaDeviceProp temp;
        cudaGetDeviceProperties(&temp, device);
        std::string name_ = temp.name;
        // std::string sm_ = std::to_string(float(temp.major) + float(temp.minor / 10.0f));
        int driver_v;
        cudaDriverGetVersion(&driver_v);
        // std::string driver_ = std::to_string(float(driver_v) / 1000.0f);
        return name_ + " with sm: " + std::to_string(temp.major) + "." + std::to_string(temp.minor) +
               " based Driver Version: " + std::to_string(int(driver_v / 1000)) + "." + std::to_string(int(driver_v % 1000));
    }

    // =======================================================
    // Memory API
    // =======================================================

    template <typename T>
    inline T *MallocHost(T *ptr, size_t count, device_t device)
    {
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaMallocHost((void **)(&ptr), count * sizeof(T)));
        return ptr;
    }

    template <typename T>
    inline T *MallocDevice(T *ptr, size_t count, device_t device)
    {
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaMalloc((void **)(&ptr), count * sizeof(T)));
        return ptr;
    }

    template <typename T>
    inline T *MallocShared(T *ptr, size_t count, device_t device)
    {
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaMallocManaged((void **)(&ptr), count * sizeof(T)));
        return ptr;
    }

    template <typename T>
    inline void MemCpy(void *dest, void *src, size_t count, device_t &device, MemCpy_t ctype)
    { // size=count*sizeof(T)
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaMemcpy(dest, src, count * sizeof(T), cudaMemcpyKind(ctype)));
    }

    template <typename T>
    inline void MemCpy(void *dest, void *src, size_t count, device_t device)
    {
        CheckGPUErrors(cudaSetDevice(device));
        cudaPointerAttributes p_dest, p_src;
        CheckGPUErrors(cudaPointerGetAttributes(&p_dest, dest));
        CheckGPUErrors(cudaPointerGetAttributes(&p_src, src));
        MemCpy_t ctype = DeF;
        if (p_dest.type == cudaMemoryTypeHost)
        {
            if (p_src.type == cudaMemoryTypeHost)
            {
                ctype = HtH;
            }
            else if (p_src.type == cudaMemoryTypeDevice)
            {
                ctype = DtH;
            }
        }
        else if (p_dest.type == cudaMemoryTypeDevice)
        {
            if (p_src.type == cudaMemoryTypeHost)
            {
                ctype = HtD;
            }
            else if (p_src.type == cudaMemoryTypeDevice)
            {
                ctype = DtD;
            }
        }
        else
        {
            printf("Direction of the transfer isn't considered and is set to cudaMemcpyDefault: \n");
            printf("  It's not one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice.\n");
        }
        // printf("Direction of the transfer is: %d\n", cudaMemcpyKind(ctype));
        CheckGPUErrors(cudaMemcpy(dest, src, count * sizeof(T), cudaMemcpyKind(ctype)));
    }

    inline void MemCpy(void *dest, void *src, size_t size, device_t device, MemCpy_t ctype)
    {
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaMemcpy(dest, src, size, cudaMemcpyKind(ctype)));
    }

    inline void Free(void *ptr, device_t &device)
    {
        CheckGPUErrors(cudaSetDevice(device));
        CheckGPUErrors(cudaFree(ptr));
    }

    // =======================================================
    // Thread API
    // =======================================================

    inline range_t AllocThd(const int global_range_x, const int global_range_y, const int global_range_z, range_t &block_size)
    {
        range_t grid_size;
        grid_size.x = (global_range_x - 1) / block_size.x + 1;
        grid_size.y = (global_range_y - 1) / block_size.y + 1;
        grid_size.z = (global_range_z - 1) / block_size.z + 1;
        return grid_size; // all blocks inside a grid;
    }

    // =======================================================
    // Sync API
    // =======================================================

    inline void Synchronize(sync_t stype, device_t &device)
    {
        switch (stype)
        {
        case 0: // device
            CheckGPUErrors(cudaDeviceSynchronize());
            break;
        case 1:
            break;
        }
    }

} // namespace name
