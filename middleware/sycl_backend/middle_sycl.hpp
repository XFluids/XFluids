#pragma once

#include <sycl/sycl.hpp>
#include <string>
#include "../utils/utils.h"

namespace middle
{
    using device_t = sycl::queue;
    using stream_t = int; // stream in sycl need to be added
                          // TODO: unify dim3.x/y/z and sycl::range[], may needn't if their member not referenced
    using range_t = sycl::range<3>;
    // typedef struct
    // {
    //     size_t x, y, z;
    //     void operator=(sycl::range<3> range_)
    //     {
    //         x = range_[0];
    //         y = range_[1];
    //         z = range_[2];
    //     }
    // } range_t;

    enum MemCpy_t
    {
        HtH = 0, /**< Host   -> Host */
        HtD = 1, /**< Host   -> Device */
        DtH = 2, /**< Device -> Host */
        DtD = 3, /**< Device -> Device */
        DeF = 4  /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
    };

    // =======================================================
    // Sync API
    // =======================================================

    inline std::string DevInfo(device_t &device)
    {
        return device.get_device().get_info<sycl::info::device::name>() + " based Driver Version: " + device.get_device().get_info<sycl::info::device::version>();
    }

    // =======================================================
    // Memory API
    // =======================================================

    template <typename T>
    inline T *MallocHost(T *ptr, size_t count, device_t &device)
    {
        return ptr = sycl::malloc_host<T>(count, device);
    }

    template <typename T>
    inline T *MallocDevice(T *ptr, size_t count, device_t &device)
    {
        return ptr = sycl::malloc_device<T>(count, device);
    }

    template <typename T>
    inline T *MallocShared(T *ptr, size_t count, device_t &device)
    {
        return ptr = sycl::malloc_shared<T>(count, device);
    }

    template <typename T>
    inline void MemCpy(void *dest, void *src, size_t count, device_t &device, MemCpy_t ctype)
    { // size=count*sizeof(T)
        device.memcpy(dest, src, sizeof(T) * count).wait();
    }

    template <typename T>
    inline void MemCpy(void *dest, void *src, size_t count, device_t &device)
    { // size=count*sizeof(T)
        device.memcpy(dest, src, sizeof(T) * count).wait();
    }

    inline void MemCpy(void *dest, void *src, size_t size, device_t &device, MemCpy_t ctype)
    {
        device.memcpy(dest, src, size).wait();
    }

    inline void Free(void *ptr, device_t &device)
    {
        sycl::free(ptr, device);
    }

    // =======================================================
    // Thread API
    // =======================================================

    inline range_t AllocThd(const size_t global_range_x, const size_t global_range_y, const size_t global_range_z, range_t &block_size)
    {
        range_t grid_size{((global_range_x - 1) / block_size[0] + 1) * block_size[0],
                          ((global_range_y - 1) / block_size[1] + 1) * block_size[1],
                          ((global_range_z - 1) / block_size[2] + 1) * block_size[2]};
        return grid_size; // all threads inside a grid;
    }

    // =======================================================
    // Sync API
    // =======================================================

    inline void Synchronize(sync_t stype, device_t &device)
    {
        switch (stype)
        {
        case 0: // device
            device.wait();
            break;
        case 1:
            break;
        }
    }

} // namespace middle