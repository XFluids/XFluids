#pragma once

#include <string>
#include "compile_sycl.h"

#if defined(__HIPSYCL_ENABLE_HIP_TARGET__) || (__HIPSYCL_ENABLE_CUDA_TARGET__)
void GetKernelAttributes(const void *Func_ptr, std::string Func_name);
#endif