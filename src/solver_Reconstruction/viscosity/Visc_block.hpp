#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"

#include "Visc_kernels.hpp"
#include "Visc_Order_kernels.hpp"

void GetCellCenterDerivative(sycl::queue &q, Block bl, FlowData &fdata, BConditions BC[6])
{
	int range_x = bl.DimX ? bl.X_inner + 4 : 1; // NOTE：这里的4是由求微分的算法确定的,内点网格向两边各延伸两个点
	int range_y = bl.DimY ? bl.Y_inner + 4 : 1;
	int range_z = bl.DimZ ? bl.Z_inner + 4 : 1;
	int offset_x = bl.DimX ? bl.Bwidth_X - 2 : 0; // NOTE: 这是计算第i(j/k)个点右边的那个半点，所以从=(+Bwidth-2) 开始到<(+inner+Bwidth+2)结束
	int offset_y = bl.DimY ? bl.Bwidth_Y - 2 : 0;
	int offset_z = bl.DimZ ? bl.Bwidth_Z - 2 : 0;
	auto local_ndrange_ck = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_ck = range<3>(((range_x - 1) / bl.dim_block_x + 1) * bl.dim_block_x, ((range_y - 1) / bl.dim_block_y + 1) * bl.dim_block_y, ((range_z - 1) / bl.dim_block_z + 1) * bl.dim_block_z);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_ck, local_ndrange_ck), [=](sycl::nd_item<3> index)
							  {
			int i = index.get_global_id(0) + offset_x;
			int j = index.get_global_id(1) + offset_y;
			int k = index.get_global_id(2) + offset_z;
			GetInnerCellCenterDerivativeKernel(i, j, k, bl, fdata.u, fdata.v, fdata.w, fdata.Vde, fdata.vxs, fdata.vx); }); })
		.wait();

	if (bl.DimX)
	{
		auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
		auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

		BConditions BC0 = BC[0], BC1 = BC[1];
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
								  {
    		int i0 = index.get_global_id(0) + 0;
			int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			CenterDerivativeBCKernelX(i0, j, k, bl, BC0, fdata.Vde, 0, bl.Bwidth_X, 1);
			CenterDerivativeBCKernelX(i1, j, k, bl, BC1, fdata.Vde, bl.X_inner, bl.Xmax-bl.Bwidth_X-1, -1); }); }); //.wait()
	}
	if (bl.DimY)
	{
		auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
		auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

		BConditions BC2 = BC[2], BC3 = BC[3];
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
								  {
    		int i = index.get_global_id(0);
			int j0 = index.get_global_id(1) + 0;
			int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
			int k = index.get_global_id(2);

			CenterDerivativeBCKernelY(i, j0, k, bl, BC2, fdata.Vde, 0, bl.Bwidth_Y, 1);
			CenterDerivativeBCKernelY(i, j1, k, bl, BC3, fdata.Vde, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); }); }); //.wait()
	}
	if (bl.DimZ)
	{
		auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
		auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

		BConditions BC4 = BC[4], BC5 = BC[5];
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
								  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
    		int k0 = index.get_global_id(2) + 0;
			int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

			CenterDerivativeBCKernelZ(i, j, k0, bl, BC4, fdata.Vde, 0, bl.Bwidth_Z, 1);
			CenterDerivativeBCKernelZ(i, j, k1, bl, BC5, fdata.Vde, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); }); }); //.wait()
	}

	q.wait();
}
