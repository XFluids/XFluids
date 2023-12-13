/**
 * This file is a simple part of early and original Umidware, a SYCL implementation.
 *
 * Copyright (c) 2023 JinLong Li
 * All rights reserved.
 *
 * some features and API functions may be deprecated or repackaged in UMidWare releases version
 */
#pragma once

#include <string>
#include <sycl/sycl.hpp>

namespace middle
{
    using device_t = sycl::queue;
    using stream_t = int; // stream in sycl need to be added
    using range_t = sycl::range<3>;

    enum sync_t
    {
        device = 0, /**DeviceSynchronize */
        block = 1   /**__syncthreads in a block */
    };

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

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocHost2D(T *ptr, size_t num_rows, size_t num_cols, sycl::queue &queue)
    {
        T **return_ptr = static_cast<T **>(sycl::malloc_host(num_rows * sizeof(T *), queue));
        // T **Bk_return_ptr = return_ptr;
        // std::cout << Bk_return_ptr << " " << return_ptr << std::endl;
        for (size_t i = 0; i < num_rows; i++)
        {
            return_ptr[i] = ptr + i * num_cols;
            // std::cout << return_ptr[i] << " ";
        }
        // std::cout << std::endl;
        // queue.memcpy(return_ptr[0], ptr, num_rows * num_cols * sizeof(T)).wait();
        // sycl::free(Bk_return_ptr, queue);

        return return_ptr;
    }

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocDevice2D(T *ptr, size_t num_rows, size_t num_cols, sycl::queue &queue)
    {
        T **return_ptr = static_cast<T **>(sycl::malloc_device(num_rows * sizeof(T *), queue));
        queue.submit([&](sycl::handler &h) {                                           // PARALLEL;
                 h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
                     for (size_t i = 0; i < num_rows; i++)
                     {
                         return_ptr[i] = ptr + i * num_cols;
                     }
                 });
             })
            .wait();

        return return_ptr;
    }

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocHost2D(T *ptr, size_t num_rows, size_t *num_cols, sycl::queue &queue)
    {
        T **return_ptr = static_cast<T **>(sycl::malloc_host(num_rows * sizeof(T *), queue));
        // std::cout << return_ptr << " " << std::endl;
        T *temp_ptr = ptr;
        for (size_t i = 0; i < num_rows; i++)
        {
            return_ptr[i] = temp_ptr;
            temp_ptr += num_cols[i];
            // std::cout << return_ptr[i] << " ";
        }
        // std::cout << std::endl;

        return return_ptr;
    }

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocDevice2D(T *ptr, size_t num_rows, size_t *num_cols, sycl::queue &queue)
    {
        size_t *dev_num_cols = sycl::malloc_device<size_t>(num_rows, queue);
        queue.memcpy(dev_num_cols, num_cols, num_rows * sizeof(size_t)).wait();
        T **return_ptr = static_cast<T **>(sycl::malloc_device(num_rows * sizeof(T *), queue));
        queue.submit([&](sycl::handler &h) {                                           // PARALLEL;
                 h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
                     T *temp_ptr = ptr;
                     for (size_t i = 0; i < num_rows; i++)
                     {
                         return_ptr[i] = temp_ptr;
                         temp_ptr += dev_num_cols[i];
                     }
                 });
             })
            .wait();

        return return_ptr;
    }

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocHost2D(T *ptr, size_t num_rows, int *num_cols, sycl::queue &queue)
    {
        T **return_ptr = static_cast<T **>(sycl::malloc_host(num_rows * sizeof(T *), queue));
        // std::cout <<  << return_ptr << " " << std::endl;
        T *temp_ptr = ptr;
        for (size_t i = 0; i < num_rows; i++)
        {
            return_ptr[i] = temp_ptr;
            temp_ptr += num_cols[i];
            // std::cout << return_ptr[i] << " ";
        }
        // std::cout << std::endl;

        return return_ptr;
    }

    /**
     * @param ptr: 1d device memory pointer
     * @param num_rows:
     * @param num_cols:
     */
    template <typename T>
    inline T **MallocDevice2D(T *ptr, size_t num_rows, int *num_cols, sycl::queue &queue)
    {
        int *dev_num_cols = sycl::malloc_device<int>(num_rows, queue);
        queue.memcpy(dev_num_cols, num_cols, num_rows * sizeof(int)).wait();
        T **return_ptr = static_cast<T **>(sycl::malloc_device(num_rows * sizeof(T *), queue));
        queue.submit([&](sycl::handler &h) {                                           // PARALLEL;
                 h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
                     T *temp_ptr = ptr;
                     for (size_t i = 0; i < num_rows; i++)
                     {
                         return_ptr[i] = temp_ptr;
                         temp_ptr += dev_num_cols[i];
                     }
                 });
             })
            .wait();

        return return_ptr;
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
