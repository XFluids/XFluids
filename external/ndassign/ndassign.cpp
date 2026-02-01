#include "ndassign.h"

Assign::Assign() : local_nd(sycl::range<3>(0, 0, 0))
{
}

Assign::Assign(sycl::range<3> lnd, std::string T) : Assign(lnd[0], lnd[1], lnd[2], T) {}

Assign::Assign(size_t x, size_t y, size_t z, std::string T) : local_nd(sycl::range(x, y, z))
{
	time = 0;
	strcpy(tag, T.c_str());
#if defined(__ACPP_ENABLE_HIP_TARGET__) || (__ACPP_ENABLE_CUDA_TARGET__)
	local_blk = local_block();
#endif
}

sycl::range<3> Assign::global_nd(sycl::range<3> gnd)
{
	return global_nd(gnd, this->local_nd);
}

sycl::range<3> Assign::global_nd(size_t x, size_t y, size_t z)
{
	return global_nd(x, y, z, this->local_nd);
}

sycl::range<3> Assign::global_nd(sycl::range<3> gnd, sycl::range<3> lnd)
{
	sycl::range<3> temp(0, 0, 0);
	temp[0] = (gnd[0] + lnd[0] - 1) / lnd[0] * lnd[0];
	temp[1] = (gnd[1] + lnd[1] - 1) / lnd[1] * lnd[1];
	temp[2] = (gnd[2] + lnd[2] - 1) / lnd[2] * lnd[2];

	return temp;
}

sycl::range<3> Assign::global_nd(size_t x, size_t y, size_t z, sycl::range<3> lnd)
{
	sycl::range<3> temp(0, 0, 0);
	temp[0] = (x + lnd[0] - 1) / lnd[0] * lnd[0];
	temp[1] = (y + lnd[1] - 1) / lnd[1] * lnd[1];
	temp[2] = (z + lnd[2] - 1) / lnd[2] * lnd[2];

	return temp;
}

Assign Assign::Time(float t)
{
	time = t;
	return *this;
}

#if defined(__ACPP_ENABLE_HIP_TARGET__) || (__ACPP_ENABLE_CUDA_TARGET__)
dim3 Assign::local_block()
{
	return local_block(this->local_nd);
}

dim3 Assign::local_block(sycl::range<3> lnd)
{
	dim3 temp;
	temp.x = lnd[0];
	temp.y = lnd[1];
	temp.z = lnd[2];

	return temp;
}

dim3 Assign::local_block(size_t x, size_t y, size_t z)
{
	dim3 temp;
	temp.x = x;
	temp.y = y;
	temp.z = z;

	return temp;
}

dim3 Assign::global_gd(sycl::range<3> gnd)
{
	return global_gd(gnd, this->local_block());
}

dim3 Assign::global_gd(size_t x, size_t y, size_t z)
{
	return global_gd(x, y, z, this->local_block());
}

dim3 Assign::global_gd(sycl::range<3> gnd, sycl::range<3> lnd)
{
	dim3 temp;
	temp.x = (gnd[0] + lnd[0] - 1) / lnd[0];
	temp.y = (gnd[1] + lnd[1] - 1) / lnd[1];
	temp.z = (gnd[2] + lnd[2] - 1) / lnd[2];

	return temp;
}

dim3 Assign::global_gd(size_t x, size_t y, size_t z, sycl::range<3> lnd)
{
	dim3 temp;
	temp.x = (x + lnd[0] - 1) / lnd[0];
	temp.y = (y + lnd[1] - 1) / lnd[1];
	temp.z = (z + lnd[2] - 1) / lnd[2];

	return temp;
}

dim3 Assign::global_gd(sycl::range<3> gnd, dim3 lnd)
{
	dim3 temp;
	temp.x = (gnd[0] + lnd.x - 1) / lnd.x;
	temp.y = (gnd[1] + lnd.y - 1) / lnd.y;
	temp.z = (gnd[2] + lnd.z - 1) / lnd.z;

	return temp;
}

dim3 Assign::global_gd(size_t x, size_t y, size_t z, dim3 lnd)
{
	dim3 temp;
	temp.x = (x + lnd.x - 1) / lnd.x;
	temp.y = (y + lnd.y - 1) / lnd.y;
	temp.z = (z + lnd.z - 1) / lnd.z;

	return temp;
}
#endif