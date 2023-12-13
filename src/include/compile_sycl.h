#pragma once

#if defined(DEFINED_OPENSYCL)
#define __SYCL_EXTERNAL_
#elif defined(DEFINED_ONEAPI)
#define __SYCL_EXTERNAL_ SYCL_EXTERNAL
#endif
