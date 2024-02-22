#pragma once

#include "Eigen_matrix.hpp"
#include "Utils_device.hpp"
#include "../Recon_device.hpp"
#include "../schemes/schemes_device.hpp"

extern SYCL_KERNEL void ReconstructFluxX(int i, int j, int k, real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										 real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										 real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * k + Xmax * j + i + 1;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

//     // construct the right value & the left value scalar equations by characteristic reduction
//     // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m, j, k);

	// 	real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value, artificial_viscosity;                   
	// for (int n = 0; n < Emax; n++)                                                                                                      
	// {                                                                                                                                   
	// 	real_t eigen_local_max = _DF(0.0);                                                                                              
	// 	RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); 
	// 	for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                     
	// 	{ 
	// 		int _i_1 = i + m, _j_1 = j, _k_1 = k; /* Xmax * Ymax * k + Xmax * j + i + m;*/                                                  
	// 		int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                               
	// 		/* local lax-friedrichs*/                               
	// 		eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n]));    
	// 	}                                                                                                                               
	// 	artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                         
	// 	for (int m = -3; m <= 4; m++)                                                                                                   
	// 	{                                                                            
	// 		/* 3rd oder and can be modified */                                                   
	// 		int _i_2 = i + m, _j_2 = j, _k_2 = k; /* Xmax * Ymax * k + Xmax * j + m + i;*/                
	// 		int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                               
	// 		uf[m + 3] = _DF(0.0);                                                                                                       
	// 		ff[m + 3] = _DF(0.0);                                                                                                       
	// 		for (int n1 = 0; n1 < Emax; n1++)                                                                                           
	// 		{                                                                                                                           
	// 			uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; 
	// 			ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];                                                        
	// 		} /*  for local speed*/                                                                                                     
	// 		pp[m + 3] = _DF(0.5) * (ff[m + 3] + artificial_viscosity * uf[m + 3]);                                                      
	// 		mm[m + 3] = _DF(0.5) * (ff[m + 3] - artificial_viscosity * uf[m + 3]);                                                      
	// 	} /* calculate the scalar numerical flux at x direction*/                                                                       
	// 	f_flux = WENO_GPU;                                                                                                              
	// 	/* WENOCU6_GPU(&pp[3], &mm[3], dl) WENO_GPU WENOCU6_P(&pp[3], dl) + WENOCU6_P(&mm[3], dl);*/                                    
	// 	/*(weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)) / _DF(6.0);*/                                                               
	// 	/* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/                                                       
	// 	/* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/                                                                          
	// 	RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
	// 	for (int n1 = 0; n1 < Emax; n1++)                                                                                               
	// 	{									   /* get Fp */                                                                             
	// 		_p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */                                                                   
	// 	}                                                                                                                               
	// }                                                                                                                                   
	// for (int n = 0; n < Emax; n++)                                                                                                      
	// { /* reconstruction the F-flux terms*/                                                                                              
	// 	real_t fluxl = _DF(0.0);                                                                                                        
	// 	for (int n1 = 0; n1 < Emax; n1++)                                                                                               
	// 	{                                                                                                                               
	// 		fluxl += _p[n1][n];                                                                                                         
	// 	}                                                                                                                               
	// 	Fwall[Emax * id_l + n] = fluxl;                                                                                                 
	// }
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC

	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_X, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_X, i + m, j, k, i + m, j, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	// real_t de_fw[Emax];
	// get_Array(Fwall, de_fw, Emax, id_l);
	// real_t de_fx[Emax];
}

extern SYCL_KERNEL void ReconstructFluxY(int i, int j, int k, real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										 real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										 real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * k + Xmax * (j + 1) + i;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC

	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

extern SYCL_KERNEL void ReconstructFluxZ(int i, int j, int k, real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										 real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										 real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * (k + 1) + Xmax * j + i;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC

	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

extern void UpdateFluidLU(int i, int j, int k, Block bl, real_t *LU, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
{
	MARCO_DOMAIN();
	if (i >= bl.Xmax)
		return;
	if (j >= bl.Ymax)
		return;
	if (k >= bl.Zmax)
		return;
	int id = Xmax * Ymax * k + Xmax * j + i;
	int id_im = Xmax * Ymax * k + Xmax * j + i - 1;
	int id_jm = Xmax * Ymax * k + Xmax * (j - 1) + i;
	int id_km = Xmax * Ymax * (k - 1) + Xmax * j + i;

	for (int n = 0; n < Emax; n++)
	{
		real_t LU0 = _DF(0.0);

		if (bl.DimX)
			LU0 += (FluxFw[Emax * id_im + n] - FluxFw[Emax * id + n]) * bl._dx;

		if (bl.DimY)
			LU0 += (FluxGw[Emax * id_jm + n] - FluxGw[Emax * id + n]) * bl._dy;

		if (bl.DimZ)
			LU0 += (FluxHw[Emax * id_km + n] - FluxHw[Emax * id + n]) * bl._dz;

		LU[Emax * id + n] = LU0;
	}

	// real_t de_LU[Emax];
	// get_Array(LU, de_LU, Emax, id);
	// real_t de_XU[Emax];
}

#if __VENDOR_SUBMIT__
_VENDOR_KERNEL_LB_(__LBMt, 1)
void ReconstructFluxXVendorWrapper(real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
								   real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
								   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + bl.Bwidth_X - 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + bl.Bwidth_Y;
	int k = blockIdx.z * blockDim.z + threadIdx.z + bl.Bwidth_Z;

	ReconstructFluxX(i, j, k, dl, bl, thermal, UI, Fl, Fwall, eigen_local, eigen_lt, eigen_rt, eb1, eb3, ec2, ezi, p, rho, u, v, w, y, T, H, eigen_block);
}

_VENDOR_KERNEL_LB_(__LBMt, 1)
void ReconstructFluxYVendorWrapper(real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
								   real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
								   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + bl.Bwidth_X;
	int j = blockIdx.y * blockDim.y + threadIdx.y + bl.Bwidth_Y - 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + bl.Bwidth_Z;

	ReconstructFluxY(i, j, k, dl, bl, thermal, UI, Fl, Fwall, eigen_local, eigen_lt, eigen_rt, eb1, eb3, ec2, ezi, p, rho, u, v, w, y, T, H, eigen_block);
}

_VENDOR_KERNEL_LB_(__LBMt, 1)
void ReconstructFluxZVendorWrapper(real_t const dl, MeshSize bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
								   real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
								   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + bl.Bwidth_X;
	int j = blockIdx.y * blockDim.y + threadIdx.y + bl.Bwidth_Y;
	int k = blockIdx.z * blockDim.z + threadIdx.z + bl.Bwidth_Z - 1;

	ReconstructFluxZ(i, j, k, dl, bl, thermal, UI, Fl, Fwall, eigen_local, eigen_lt, eigen_rt, eb1, eb3, ec2, ezi, p, rho, u, v, w, y, T, H, eigen_block);
}
#endif
