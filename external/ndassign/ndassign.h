#pragma once

#include <sycl/sycl.hpp>

struct Assign
{
	float time;
	char tag[40];
	sycl::range<3> local_nd;

	Assign();
	~Assign() {};
	Assign(sycl::range<3> lnd, std::string T = "undefined block");
	Assign(size_t x, size_t y, size_t z, std::string T = "undefined block");
	sycl::range<3> global_nd(sycl::range<3> gnd);
	sycl::range<3> global_nd(size_t x, size_t y, size_t z);
	sycl::range<3> global_nd(sycl::range<3> gnd, sycl::range<3> lnd);
	sycl::range<3> global_nd(size_t x, size_t y, size_t z, sycl::range<3> lnd);
	Assign Time(float t);
#if defined(__ACPP_ENABLE_HIP_TARGET__) || (__ACPP_ENABLE_CUDA_TARGET__)
	dim3 local_blk;
	dim3 local_block();
	dim3 local_block(sycl::range<3> lnd);
	dim3 local_block(size_t x, size_t y, size_t z);
	dim3 global_gd(sycl::range<3> gnd);
	dim3 global_gd(size_t x, size_t y, size_t z);
	dim3 global_gd(sycl::range<3> gnd, sycl::range<3> lnd);
	dim3 global_gd(size_t x, size_t y, size_t z, sycl::range<3> lnd);
	dim3 global_gd(sycl::range<3> gnd, dim3 lnd);
	dim3 global_gd(size_t x, size_t y, size_t z, dim3 lnd);
#endif
};