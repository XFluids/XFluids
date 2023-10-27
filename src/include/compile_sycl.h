#pragma once

#if defined(DEFINED_OPENSYCL)
#define __SYCL_EXTERNAL_
#else
#define __SYCL_EXTERNAL_ SYCL_EXTERNAL
#endif