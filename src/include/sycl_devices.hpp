#pragma once

#include "global_setup.h"
#include "../solver_Ini/Mixing_device.h"

using namespace sycl;

// =======================================================
// //    Global sycL_reduction
#if defined(DEFINED_ONEAPI)
#define sycl_plus_op(T) sycl::plus<>()
#define sycl_max_op(T) sycl::maximum<>()
#define sycl_min_op(T) sycl::minimum<>()

#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<>())

#elif defined(__ACPP__)
#define sycl_plus_op(T) sycl::plus<T>()
#define sycl_max_op(T) sycl::maximum<T>()
#define sycl_min_op(T) sycl::minimum<T>()

#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<real_t>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<real_t>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<real_t>())
#endif
// =======================================================
