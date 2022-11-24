#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

#include <oneapi/tbb/global_control.h>
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/tick_count.h"
#include "tbb/scalable_allocator.h"
#include "tbb/tick_count.h"

#include "global_class.h"
#include "global_function.h"

using namespace sycl;
using namespace std;

using namespace tbb;

void AllocateMemory(sycl::queue &q, FluidSYCL &fluid);

// void ReconstructFluxX(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real* rho, Real* u, Real* v, 
//                                             Real* w, Real* H, Real dx, Real dy);

// void RoeAverage_x(Real eigen_l[Emax][Emax], Real eigen_r[Emax][Emax], Real const _rho, Real const _u, Real const _v, Real const _w, 
// 	Real const _H, Real const D, Real const D1);
// Real weno5old_P(Real *f, Real delta);
// Real weno5old_M(Real *f, Real delta);

// Real *d_a;

int main2()
{
	oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, NumThread);

	FluidSYCL fluid;

	// Real *d_U, *d_U1, *d_LU;
	// Real *d_eigen_local;

	// Real *d_FluxF, *d_FluxG, *d_FluxH;
	// Real *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;

	// FlowData d_fstate;

	Real *h_U, *h_U1, *h_LU;
	Real *h_eigen_local;

	Real *h_FluxF, *h_FluxG, *h_FluxH;
	Real *h_wallFluxF, *h_wallFluxG, *h_wallFluxH;

	FlowData h_fstate;

	auto device = sycl::platform::get_platforms()[2].get_devices()[0];
	// accelerator_selector device;
	queue q(device, dpc_common::exception_handler);
    
    std::cout << "Device: "<< q.get_device().get_info<sycl::info::device::name>() 
						<< ",  version = "<<q.get_device().get_info<sycl::info::device::version>() << "\n";

	AllocateMemory(q, fluid);
  	// d_U  = static_cast<Real *>(malloc_device(cellbytes, q));
  	// d_U1 = static_cast<Real *>(malloc_device(cellbytes, q));
  	// d_LU = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_eigen_local = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_fstate.rho = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.p = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.c = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.H = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.u = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.v = static_cast<Real *>(malloc_device(bytes, q));
	// d_fstate.w = static_cast<Real *>(malloc_device(bytes, q));
	// d_FluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_FluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_FluxH  = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_wallFluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_wallFluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	// d_wallFluxH  = static_cast<Real *>(malloc_device(cellbytes, q));

  	// 主机内存
  	h_U  = static_cast<Real *>(malloc(cellbytes));
  	h_U1 = static_cast<Real *>(malloc(cellbytes));
  	h_LU = static_cast<Real *>(malloc(cellbytes));
	h_eigen_local = static_cast<Real *>(malloc(cellbytes));
	h_fstate.rho = static_cast<Real *>(malloc(bytes));
	h_fstate.p = static_cast<Real *>(malloc(bytes));
	h_fstate.c = static_cast<Real *>(malloc(bytes));
	h_fstate.H = static_cast<Real *>(malloc(bytes));
	h_fstate.u = static_cast<Real *>(malloc(bytes));
	h_fstate.v = static_cast<Real *>(malloc(bytes));
	h_fstate.w = static_cast<Real *>(malloc(bytes));
	h_FluxF  = static_cast<Real *>(malloc(cellbytes));
	h_FluxG  = static_cast<Real *>(malloc(cellbytes));
	h_FluxH  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxF  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxG  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxH  = static_cast<Real *>(malloc(cellbytes));

	cout << "Memory Usage: " << (Real)((long)10*cellbytes + (long)7*bytes)/(Real)(1024*1024*1024)<< " GB\n";

	bool is_3d = DIM_X*DIM_Y*DIM_Z ? true : false;

	#if DIM_X
    // blocksize = is_3d ?  dim3(1, WarpSize/2, WarpSize) : dim_blk;
    // gridsize = is_3d ? dim3(X_inner/blocksize.x+1, Y_inner/blocksize.y, Z_inner/blocksize.z) : dim3(dim_grid.x + 1, dim_grid.y, dim_grid.z);

  	// local_ndrange：工作组大小为block_size*block_size = 16*16
  	auto local_ndrange = range<3>(dim_block_x, dim_block_y, dim_block_z);
  	// global_ndrange：工作项的大小 = 1024*1024
  	auto global_ndrange = range<3>(X_inner+local_ndrange[0], Y_inner, Z_inner);

  	double duration = 0.0;
  	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	// cpu solver
	static affinity_partitioner ap;
	for(int n=0; n<10; n++){
		parallel_for( blocked_range3d<int>(Bwidth_X-1, X_inner+Bwidth_X, Bwidth_Y, Y_inner+Bwidth_Y, Bwidth_Z, Z_inner+Bwidth_Z),
			[&](const blocked_range3d<int>& r){
			for( int i=r.pages().begin(); i!=r.pages().end(); ++i)
				for( int j=r.rows().begin(); j!=r.rows().end(); ++j)
					for( int k=r.cols().begin(); k!=r.cols().end(); ++k){
						// ReconstructFluxX(i, j, k, h_U, h_FluxF, h_wallFluxF, h_eigen_local, h_fstate.rho, h_fstate.u, h_fstate.v, h_fstate.w, h_fstate.H, 0.01, 0.01);
					}
			}, ap);
	}

  	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
  	duration = std::chrono::duration<float, std::milli>(end_time - start_time).count();

  	cout << "CPU running time: " << duration/1000.0f << " s\n\n";

	start_time = std::chrono::high_resolution_clock::now();

	Real *U = fluid.d_U;

	// d_a  = static_cast<Real *>(malloc_device(sizeof(Real), q));
	// Real *h_a = static_cast<Real *>(malloc(sizeof(Real)));
	// Real *a = d_a;

    for(int n=0; n<10; n++){
    	// 提交命令组以执行
		try{
		q.submit([&](sycl::handler &h){
			// auto out = sycl::stream(20000, 20000, h);
			h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    			// 利用get_global_id获得全局指标
    			int i = index.get_global_id(0) + Bwidth_X - 1;
    			int j = index.get_global_id(1) + Bwidth_Y;
				int k = index.get_global_id(2) + Bwidth_Z;

				// a[0] = 0.7f;

				// ReconstructFluxX(i, j, k, fluid.d_U, fluid.d_FluxF, fluid.d_wallFluxF, fluid.d_eigen_local, fluid.d_fstate.rho, fluid.d_fstate.u, fluid.d_fstate.v, fluid.d_fstate.w, fluid.d_fstate.H, 0.01, 0.01);
			});
		});
		}
		catch (sycl::exception exc) {
    		std::cerr << exc.what() << "Exception caught" << std::endl;
    		std::exit(1);
		}

		q.wait();
    }
	#endif

	// q.memcpy(h_a, d_a, sizeof(Real)).wait();
	// cout << "a = " << h_a[0] << " \n\n";

  	end_time = std::chrono::high_resolution_clock::now();
  	duration = std::chrono::duration<float, std::milli>(end_time - start_time).count();

  	cout << "GPU Offload time: " << duration/1000.0f << " s\n\n";

    // CheckCUDAErrors(cudaDeviceSynchronize());

	// printf("GPU runtime for original flux: %g s\n",  elapsed_time/1.0e3);

    return 0;
}

// // add "sycl::nd_item<3> item" for get_global_id
// // add "stream const s" for output
// void ReconstructFluxX(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real* rho, Real* u, Real* v, 
//                                             Real* w, Real* H, Real dx, Real dy)
// {
//     // // 利用get_global_id获得全局指标
//     // int i = item.get_global_id(0) + Bwidth_X - 1;
//     // int j = item.get_global_id(1) + Bwidth_Y;
// 	// int k = item.get_global_id(2) + Bwidth_Z;

//     int id_l = Xmax*Ymax*k + Xmax*j + i;
//     int id_r = Xmax*Ymax*k + Xmax*j + i + 1;

// 	// cout<<"i,j,k = "<<i<<", "<<j<<", "<<k<<", "<<Emax*(Xmax*j + i-2)+0<<"\n";
// 	// printf("%f", UI[0]);

//     if(i>= X_inner+Bwidth_X)
//         return;

//     Real eigen_l[Emax][Emax], eigen_r[Emax][Emax];

// 	//preparing some interval value for roe average
// 	Real D	=	sqrt(rho[id_r]/rho[id_l]);
// 	#if USE_DP
// 	Real D1	=	1.0 / (D + 1.0);
// 	#else
// 	Real D1	=	1.0f / (D + 1.0f);
// 	#endif
// 	Real _u	=	(u[id_l] + D*u[id_r])*D1;
// 	Real _v	=	(v[id_l] + D*v[id_r])*D1;
// 	Real _w	=	(w[id_l] + D*w[id_r])*D1;
// 	Real _H	=	(H[id_l] + D*H[id_r])*D1;
// 	Real _rho = sqrt(rho[id_r]*rho[id_l]);

//     RoeAverage_x(eigen_l, eigen_r, _rho, _u, _v, _w, _H, D, D1);

//     Real uf[10],  ff[10], pp[10], mm[10];
//     Real f_flux, _p[Emax][Emax];

// 	// construct the right value & the left value scalar equations by characteristic reduction			
// 	// at i+1/2 in x direction
//     // #pragma unroll Emax
// 	for(int n=0; n<Emax; n++){
//         #if USE_DP
//         Real eigen_local_max = 0.0;
//         #else
//         Real eigen_local_max = 0.0f;
//         #endif
//         for(int m=-2; m<=3; m++){
//             int id_local = Xmax*Ymax*k + Xmax*j + i + m;
//             eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
//         }

// 		for(int m=i-3; m<=i+4; m++){	// 3rd oder and can be modified
//             int id_local = Xmax*Ymax*k + Xmax*j + m;
//             #if USE_DP
// 			uf[m-i+3] = 0.0;
// 			ff[m-i+3] = 0.0;
//             #else
// 			uf[m-i+3] = 0.0f;
// 			ff[m-i+3] = 0.0f;
//             #endif

// 			for(int n1=0; n1<Emax; n1++){
// 				uf[m-i+3] = uf[m-i+3] + UI[Emax*id_local+n1]*eigen_l[n][n1];
// 				ff[m-i+3] = ff[m-i+3] + Fx[Emax*id_local+n1]*eigen_l[n][n1];
// 			}
// 			// for local speed
// 			pp[m-i+3] = 0.5f*(ff[m-i+3] + eigen_local_max*uf[m-i+3]);
// 			mm[m-i+3] = 0.5f*(ff[m-i+3] - eigen_local_max*uf[m-i+3]);
//         }

// 		// calculate the scalar numerical flux at x direction
//         #if USE_DP
//         f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0;
//         #else
//         f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0f;
//         #endif

// 		// get Fp
// 		for(int n1=0; n1<Emax; n1++)
// 			_p[n][n1] = f_flux*eigen_r[n1][n];
//     }

// 	// reconstruction the F-flux terms
// 	for(int n=0; n<Emax; n++){
//         #if USE_DP
//         Real fluxx = 0.0;
//         #else
//         Real fluxx = 0.0f;
//         #endif
// 		for(int n1=0; n1<Emax; n1++) {
//             fluxx += _p[n1][n];
// 		}
//         Fxwall[Emax*id_l+n] = fluxx;
// 	}
// }

void AllocateMemory(sycl::queue &q, FluidSYCL &fluid)
{
  	// 采用USM方法分配设备内存
  	fluid.d_U  = static_cast<Real *>(malloc_device(cellbytes, q));
  	fluid.d_U1 = static_cast<Real *>(malloc_device(cellbytes, q));
  	fluid.d_LU = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_eigen_local = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_fstate.rho = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.p = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.c = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.H = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.u = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.v = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_fstate.w = static_cast<Real *>(malloc_device(bytes, q));
	fluid.d_FluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_FluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_FluxH  = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_wallFluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_wallFluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	fluid.d_wallFluxH  = static_cast<Real *>(malloc_device(cellbytes, q));

	// cout << "Memory Usage: " << (float)(10*cellbytes + 7*bytes)/(float)(1024*1024*1024) << " GB\n";
}


// void RoeAverage_x(Real eigen_l[Emax][Emax], Real eigen_r[Emax][Emax], Real const _rho, Real const _u, Real const _v, Real const _w, 
// 	Real const _H, Real const D, Real const D1)
// {
// 	//preparing some interval value
// 	#if USE_DP
// 	Real one_float = 1.0;
// 	Real half_float = 0.5;
// 	Real zero_float = 0.0;
// 	#else
// 	Real one_float = 1.0f;
// 	Real half_float = 0.5f;
// 	Real zero_float = 0.0f;
// 	#endif

// 	Real _Gamma = Gamma - one_float;
// 	Real _rho1 = one_float / _rho;
// 	Real q2 = _u*_u + _v*_v + _w*_w;
// 	Real c2 = _Gamma*(_H - half_float*q2);
// 	Real _c = sqrt(c2);
// 	Real _c1_rho = half_float*_rho / _c;
// 	Real c21_Gamma = _Gamma / c2;
// 	Real _c1_rho1_Gamma = _Gamma*_rho1 / _c;

// 	// left eigen vectors
// 	eigen_l[0][0] = one_float - half_float*c21_Gamma*q2;
// 	eigen_l[0][1] = c21_Gamma*_u;
// 	eigen_l[0][2] = c21_Gamma*_v;
// 	eigen_l[0][3] = c21_Gamma*_w;
// 	eigen_l[0][4] = -c21_Gamma;
	
// 	eigen_l[1][0] = -_w*_rho1;
// 	eigen_l[1][1] = zero_float;
// 	eigen_l[1][2] = zero_float;
// 	eigen_l[1][3] = _rho1;
// 	eigen_l[1][4] = zero_float;

// 	eigen_l[2][0] = _v*_rho1;
// 	eigen_l[2][1] = zero_float;
// 	eigen_l[2][2] = -_rho1;
// 	eigen_l[2][3] = zero_float;
// 	eigen_l[2][4] = zero_float;

// 	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _u*_rho1;
// 	eigen_l[3][1] = -_c1_rho1_Gamma*_u + _rho1;
// 	eigen_l[3][2] = -_c1_rho1_Gamma*_v;
// 	eigen_l[3][3] = -_c1_rho1_Gamma*_w;
// 	eigen_l[3][4] = _c1_rho1_Gamma;

// 	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _u*_rho1;
// 	eigen_l[4][1] = -_c1_rho1_Gamma*_u - _rho1;
// 	eigen_l[4][2] = -_c1_rho1_Gamma*_v;
// 	eigen_l[4][3] = -_c1_rho1_Gamma*_w;
// 	eigen_l[4][4] = _c1_rho1_Gamma;

// 	//right eigen vectors
// 	eigen_r[0][0] = one_float;
// 	eigen_r[0][1] = zero_float;
// 	eigen_r[0][2] = zero_float;
// 	eigen_r[0][3] = _c1_rho;
// 	eigen_r[0][4] = _c1_rho;
	
// 	eigen_r[1][0] = _u;
// 	eigen_r[1][1] = zero_float;
// 	eigen_r[1][2] = zero_float;
// 	eigen_r[1][3] = _c1_rho*(_u + _c);
// 	eigen_r[1][4] = _c1_rho*(_u - _c);

// 	eigen_r[2][0] = _v;
// 	eigen_r[2][1] = zero_float;
// 	eigen_r[2][2] = -_rho;
// 	eigen_r[2][3] = _c1_rho*_v;
// 	eigen_r[2][4] = _c1_rho*_v;

// 	eigen_r[3][0] = _w;
// 	eigen_r[3][1] = _rho;
// 	eigen_r[3][2] = zero_float;
// 	eigen_r[3][3] = _c1_rho*_w;
// 	eigen_r[3][4] = _c1_rho*_w;

// 	eigen_r[4][0] = half_float*q2;
// 	eigen_r[4][1] = _rho*_w;
// 	eigen_r[4][2] = -_rho*_v;
// 	eigen_r[4][3] = _c1_rho*(_H + _u*_c);
// 	eigen_r[4][4] = _c1_rho*(_H - _u*_c);
// }

// Real weno5old_P(Real *f, Real delta)
// {
// 	int k;
// 	Real v1, v2, v3, v4, v5;
// 	Real a1, a2, a3;
// //	Real w1, w2, w3;

// 	//assign value to v1, v2,...
// 	k = 0;
// 	v1 = *(f + k - 2);
// 	v2 = *(f + k - 1);
// 	v3 = *(f + k);
// 	v4 = *(f + k + 1); 
// 	v5 = *(f + k + 2);

// 	//smoothness indicator
// //	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
// //	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
// //	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
// //	   + 3.0*(v2 - v4)*(v2 - v4);
// //	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
// //	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	
//         //weights
// //      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
// //      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
// //      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
// //      Real tw1 = 1.0/(a1 + a2 +a3); 
// //      w1 = a1*tw1;
// //      w2 = a2*tw1;
// //      w3 = a3*tw1;

// //      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
// //      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
// //      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

// //      return (a1+a2+a3)/6.0;

//         //return weighted average
// //      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
//   //              + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
//     //            + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;

// 		#if USE_DP
// 		a1 = v1 - 2.0*v2 + v3;
//         Real s1 = 13.0*a1*a1;
//         a1 = v1 - 4.0*v2 + 3.0*v3;
//         s1 += 3.0*a1*a1;
//         a1 = v2 - 2.0*v3 + v4;
//         Real s2 = 13.0*a1*a1;
//         a1 = v2 - v4;
//         s2 += 3.0*a1*a1;
//         a1 = v3 - 2.0*v4 + v5;
//         Real s3 = 13.0*a1*a1;
//         a1 = 3.0*v3 - 4.0*v4 + v5;
//         s3 += 3.0*a1*a1;
// 		#else
// 		a1 = v1 - 2.0f*v2 + v3;
//         Real s1 = 13.0f*a1*a1;
//         a1 = v1 - 4.0f*v2 + 3.0f*v3;
//         s1 += 3.0f*a1*a1;
//         a1 = v2 - 2.0f*v3 + v4;
//         Real s2 = 13.0f*a1*a1;
//         a1 = v2 - v4;
//         s2 += 3.0f*a1*a1;
//         a1 = v3 - 2.0f*v4 + v5;
//         Real s3 = 13.0f*a1*a1;
//         a1 = 3.0f*v3 - 4.0f*v4 + v5;
//         s3 += 3.0f*a1*a1;
// 		#endif

//     	// a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
//     	// a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
//     	// a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
//     	// Real tw1 = 1.0/(a1 + a2 +a3); 
//     	// a1 = a1*tw1;
//     	// a2 = a2*tw1;
//     	// a3 = a3*tw1;
//         Real tol = 1.0e-6;
// 		#if USE_DP
//         a1 = 0.1*(tol + s2)*(tol + s2)*(tol + s3)*(tol + s3);
//         a2 = 0.2*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
//         a3 = 0.3*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
//         Real tw1 = 1.0/(a1 + a2 +a3);
// 		#else
//         a1 = 0.1f*(tol + s2)*(tol + s2)*(tol + s3)*(tol + s3);
//         a2 = 0.2f*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
//         a3 = 0.3f*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
//         Real tw1 = 1.0f/(a1 + a2 +a3);
// 		#endif

//     	a1 = a1*tw1;
//     	a2 = a2*tw1;
//     	a3 = a3*tw1;

// 		#if USE_DP
//     	s1 = a1*(2.0*v1 - 7.0*v2 + 11.0*v3);
//     	s2 = a2*(-v2 + 5.0*v3 + 2.0*v4);
//     	s3 = a3*(2.0*v3 + 5.0*v4 - v5);
// 		#else
//     	s1 = a1*(2.0f*v1 - 7.0f*v2 + 11.0f*v3);
//     	s2 = a2*(-v2 + 5.0f*v3 + 2.0f*v4);
//     	s3 = a3*(2.0f*v3 + 5.0f*v4 - v5);
// 		#endif

//     	// return (s1+s2+s3)/6.0;
// 		return (s1+s2+s3);

//         // a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
//         // a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
//         // a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
//         // Real tw1 = 6.0*(a1 + a2 + a3);
//         // return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
// }
// Real weno5old_M(Real *f, Real delta)
// {
// 	int k;
// 	Real v1, v2, v3, v4, v5;
// 	Real a1, a2, a3;
// //	Real w1, w2, w3;

// 	//assign value to v1, v2,...
// 	k = 1;
// 	v1 = *(f + k + 2);
// 	v2 = *(f + k + 1);
// 	v3 = *(f + k);
// 	v4 = *(f + k - 1); 
// 	v5 = *(f + k - 2);

// 	//smoothness indicator
// //	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
// //	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
// //	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
// //	   + 3.0*(v2 - v4)*(v2 - v4);
// //	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
// //	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

//         //weights
// //      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
// //      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
// //      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
// //      Real tw1 = 1.0/(a1 + a2 +a3); 
// //      w1 = a1*tw1;
// //      w2 = a2*tw1;
// //      w3 = a3*tw1;

// //      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
// //      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
// //      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

// //      return (a1+a2+a3)/6.0;

//         //return weighted average
// //      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
// //                + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
// //                + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;	

// 		#if USE_DP
//         a1 = v1 - 2.0*v2 + v3;
//         Real s1 = 13.0*a1*a1;
//         a1 = v1 - 4.0*v2 + 3.0*v3;
//         s1 += 3.0*a1*a1;
//         a1 = v2 - 2.0*v3 + v4;
//         Real s2 = 13.0*a1*a1;
//         a1 = v2 - v4;
//         s2 += 3.0*a1*a1;
//         a1 = v3 - 2.0*v4 + v5;
//         Real s3 = 13.0*a1*a1;
//         a1 = 3.0*v3 - 4.0*v4 + v5;
//         s3 += 3.0*a1*a1;
// 		#else
//         a1 = v1 - 2.0f*v2 + v3;
//         Real s1 = 13.0f*a1*a1;
//         a1 = v1 - 4.0f*v2 + 3.0f*v3;
//         s1 += 3.0f*a1*a1;
//         a1 = v2 - 2.0f*v3 + v4;
//         Real s2 = 13.0f*a1*a1;
//         a1 = v2 - v4;
//         s2 += 3.0f*a1*a1;
//         a1 = v3 - 2.0f*v4 + v5;
//         Real s3 = 13.0f*a1*a1;
//         a1 = 3.0f*v3 - 4.0f*v4 + v5;
//         s3 += 3.0f*a1*a1;
// 		#endif

// 		//  a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
// 		//  a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
// 		//  a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
// 		//  Real tw1 = 1.0/(a1 + a2 +a3); 
// 		//  a1 = a1*tw1;
// 		//  a2 = a2*tw1;
// 		//  a3 = a3*tw1;
//         Real tol = 1.0e-6;
// 		#if USE_DP
// 		a1 = 0.1*(tol+ s2)*(tol + s2)*(tol + s3)*(tol + s3);
//         a2 = 0.2*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
//         a3 = 0.3*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
//         Real tw1 = 1.0/(a1 + a2 +a3);
// 		#else
// 		a1 = 0.1f*(tol+ s2)*(tol + s2)*(tol + s3)*(tol + s3);
//         a2 = 0.2f*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
//         a3 = 0.3f*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
//         Real tw1 = 1.0f/(a1 + a2 +a3);
// 		#endif
//          a1 = a1*tw1;
//          a2 = a2*tw1;
//          a3 = a3*tw1;

// 		#if USE_DP
// 		s1 = a1*(2.0*v1 - 7.0*v2 + 11.0*v3);
// 		s2 = a2*(-v2 + 5.0*v3 + 2.0*v4);
// 		s3 = a3*(2.0*v3 + 5.0*v4 - v5);
// 		#else
// 		s1 = a1*(2.0f*v1 - 7.0f*v2 + 11.0f*v3);
// 		s2 = a2*(-v2 + 5.0f*v3 + 2.0f*v4);
// 		s3 = a3*(2.0f*v3 + 5.0f*v4 - v5);
// 		#endif
 
// 		//  return (s1+s2+s3)/6.0;
// 		return (s1+s2+s3);

//         //  a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
//         //  a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
//         //  a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
//         //  Real tw1 = 6.0*(a1 + a2 + a3);
//         //  return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
// }