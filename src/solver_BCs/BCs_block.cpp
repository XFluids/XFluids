#include "BCs_block.h"
#include "BCs_kernels.hpp"

float FluidBoundaryCondition(sycl::queue &q, Setup setup, BConditions BCs[6], real_t *d_UI)
{
	Block bl = setup.BlSz;
#if USE_MPI
	MpiTrans Trans = *(setup.mpiTrans);
// =======================================================
#ifdef EXPLICIT_ALLOC
	// =======================================================
	real_t *ptr_TransBufSend_xmin = Trans.d_mpiData.TransBufSend_xmin;
	real_t *ptr_TransBufSend_xmax = Trans.d_mpiData.TransBufSend_xmax;
	real_t *ptr_TransBufRecv_xmin = Trans.d_mpiData.TransBufRecv_xmin;
	real_t *ptr_TransBufRecv_xmax = Trans.d_mpiData.TransBufRecv_xmax;

	real_t *ptr_TransBufSend_ymin = Trans.d_mpiData.TransBufSend_ymin;
	real_t *ptr_TransBufSend_ymax = Trans.d_mpiData.TransBufSend_ymax;
	real_t *ptr_TransBufRecv_ymin = Trans.d_mpiData.TransBufRecv_ymin;
	real_t *ptr_TransBufRecv_ymax = Trans.d_mpiData.TransBufRecv_ymax;

	real_t *ptr_TransBufSend_zmin = Trans.d_mpiData.TransBufSend_zmin;
	real_t *ptr_TransBufSend_zmax = Trans.d_mpiData.TransBufSend_zmax;
	real_t *ptr_TransBufRecv_zmin = Trans.d_mpiData.TransBufRecv_zmin;
	real_t *ptr_TransBufRecv_zmax = Trans.d_mpiData.TransBufRecv_zmax;
	// =======================================================
#else
	// =======================================================
#define ptr_TransBufSend_xmin Trans.d_mpiData->TransBufSend_xmin
#define ptr_TransBufSend_xmax Trans.d_mpiData->TransBufSend_xmax
#define ptr_TransBufRecv_xmin Trans.d_mpiData->TransBufRecv_xmin
#define ptr_TransBufRecv_xmax Trans.d_mpiData->TransBufRecv_xmax

#define ptr_TransBufSend_ymin Trans.d_mpiData->TransBufSend_ymin
#define ptr_TransBufSend_ymax Trans.d_mpiData->TransBufSend_ymax
#define ptr_TransBufRecv_ymin Trans.d_mpiData->TransBufRecv_ymin
#define ptr_TransBufRecv_ymax Trans.d_mpiData->TransBufRecv_ymax

#define ptr_TransBufSend_zmin Trans.d_mpiData->TransBufSend_zmin
#define ptr_TransBufSend_zmax Trans.d_mpiData->TransBufSend_zmax
#define ptr_TransBufRecv_zmin Trans.d_mpiData->TransBufRecv_zmin
#define ptr_TransBufRecv_zmax Trans.d_mpiData->TransBufRecv_zmax
	// =======================================================
#endif // end EXPLICIT_ALLOC
// =======================================================
#endif // end USE_MPI

	float duration_x = 0.0f, duration_y = 0.0f, duration_z = 0.0f;

	if (bl.DimX)
	{
		auto local_ndrange_x = sycl::range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
		auto global_ndrange_x = sycl::range<3>(bl.Bwidth_X,
											   (bl.Ymax + local_ndrange_x[1] - 1) / local_ndrange_x[1] * local_ndrange_x[1],
											   (bl.Zmax + local_ndrange_x[2] - 1) / local_ndrange_x[2] * local_ndrange_x[2]);

#if USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
								  int i0 = index.get_global_id(0) + 0;
								  int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
								  int j = index.get_global_id(1);
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelX(i0, j, k, bl, ptr_TransBufSend_xmin, d_UI, 0, -bl.Bwidth_X, BorToBuf);					// X_MIN
								  FluidMpiCopyKernelX(i1, j, k, bl, ptr_TransBufSend_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BorToBuf); // X_MAX
							  }); })
		.wait();

	std::chrono::high_resolution_clock::time_point start_time_x = std::chrono::high_resolution_clock::now();

	Trans.MpiTransBuf(q, XDIR);

	std::chrono::high_resolution_clock::time_point end_time_x = std::chrono::high_resolution_clock::now();
	duration_x = std::chrono::duration<float, std::milli>(end_time_x - start_time_x).count();

#endif // USE_MPI
	BConditions BC0 = BCs[0], BC1 = BCs[1];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
				   {
					   int i0 = index.get_global_id(0) + 0;
					   int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
					   int j = index.get_global_id(1);
					   int k = index.get_global_id(2);
#if USE_MPI
					   if (Trans.neighborsBC[XMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
						   FluidMpiCopyKernelX(i0, j, k, bl, ptr_TransBufRecv_xmin, d_UI, 0, -bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i0, j, k, bl, BC0, d_UI, 0, bl.Bwidth_X, 1);
#ifdef USE_MPI
					   if (Trans.neighborsBC[XMAX] == BC_COPY)
						   FluidMpiCopyKernelX(i1, j, k, bl, ptr_TransBufRecv_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i1, j, k, bl, BC1, d_UI, bl.X_inner, bl.Xmax - bl.Bwidth_X - 1, -1); }); })
		.wait();
	}

	if (bl.DimY)
	{
		auto local_ndrange_y = sycl::range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
		auto global_ndrange_y = sycl::range<3>((bl.Xmax + local_ndrange_y[0] - 1) / local_ndrange_y[0] * local_ndrange_y[0],
											   bl.Bwidth_Y,
											   (bl.Zmax + local_ndrange_y[2] - 1) / local_ndrange_y[2] * local_ndrange_y[2]);

#if USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelY(i, j0, k, bl, ptr_TransBufSend_ymin, d_UI, 0, -bl.Bwidth_Y, BorToBuf);					// X_MIN
								  FluidMpiCopyKernelY(i, j1, k, bl, ptr_TransBufSend_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BorToBuf); // X_MAX
							  }); })
		.wait();

	std::chrono::high_resolution_clock::time_point start_time_y = std::chrono::high_resolution_clock::now();

	Trans.MpiTransBuf(q, YDIR);

	std::chrono::high_resolution_clock::time_point end_time_y = std::chrono::high_resolution_clock::now();
	duration_y = std::chrono::duration<float, std::milli>(end_time_y - start_time_y).count();

#endif // USE_MPI
	BConditions BC2 = BCs[2], BC3 = BCs[3];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
				   {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);
#if USE_MPI
								  if (Trans.neighborsBC[YMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
									  FluidMpiCopyKernelY(i, j0, k, bl, ptr_TransBufRecv_ymin, d_UI, 0, -bl.Bwidth_Y, BufToBC);
								  else
#endif // USE_MPI
									  FluidBCKernelY(i, j0, k, bl, BC2, d_UI, 0, bl.Bwidth_Y, 1);
#ifdef USE_MPI
								  if (Trans.neighborsBC[YMAX] == BC_COPY)
									  FluidMpiCopyKernelY(i, j1, k, bl, ptr_TransBufRecv_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BufToBC);
								  else
#endif																													  // USE_MPI
									  FluidBCKernelY(i, j1, k, bl, BC3, d_UI, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); //
				   }); })
		.wait();
	}

	if (bl.DimZ)
	{
		auto local_ndrange_z = sycl::range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
		auto global_ndrange_z = sycl::range<3>((bl.Xmax + local_ndrange_z[0] - 1) / local_ndrange_z[0] * local_ndrange_z[0],
											   (bl.Ymax + local_ndrange_z[1] - 1) / local_ndrange_z[1] * local_ndrange_z[1],
											   bl.Bwidth_Z);

#if USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
				   {
					   int i = index.get_global_id(0);
					   int j = index.get_global_id(1);
					   int k0 = index.get_global_id(2) + 0;
					   int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

					   FluidMpiCopyKernelZ(i, j, k0, bl, ptr_TransBufSend_zmin, d_UI, 0, -bl.Bwidth_Z, BorToBuf);					 // X_MIN
					   FluidMpiCopyKernelZ(i, j, k1, bl, ptr_TransBufSend_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BorToBuf); // X_MAX
				   }); })
		.wait();

	std::chrono::high_resolution_clock::time_point start_time_z = std::chrono::high_resolution_clock::now();

	Trans.MpiTransBuf(q, ZDIR);

	std::chrono::high_resolution_clock::time_point end_time_z = std::chrono::high_resolution_clock::now();
	duration_z = std::chrono::duration<float, std::milli>(end_time_z - start_time_z).count();

#endif // USE_MPI
	BConditions BC4 = BCs[4], BC5 = BCs[5];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
				   {
					   int i = index.get_global_id(0);
					   int j = index.get_global_id(1);
					   int k0 = index.get_global_id(2) + 0;
					   int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;
#if USE_MPI
					   if (Trans.neighborsBC[ZMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
						   FluidMpiCopyKernelZ(i, j, k0, bl, ptr_TransBufRecv_zmin, d_UI, 0, -bl.Bwidth_Z, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelZ(i, j, k0, bl, BC4, d_UI, 0, bl.Bwidth_Z, 1);
#ifdef USE_MPI
					   if (Trans.neighborsBC[ZMAX] == BC_COPY)
						   FluidMpiCopyKernelZ(i, j, k1, bl, ptr_TransBufRecv_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BufToBC);
					   else
#endif
						   FluidBCKernelZ(i, j, k1, bl, BC5, d_UI, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); //
				   }); })
		.wait();
	}

	q.wait();

	return (duration_x + duration_y + duration_z) * 1.0E-3f;
}
