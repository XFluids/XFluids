#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"

extern void EstimateYiKernel(int i, int j, int k, Block bl, int *error_pos, bool *error_org, bool *error_nan, real_t *UI, real_t *rho, real_t *y)
{
	int Xmax = bl.Xmax;
	int Ymax = bl.Ymax;
	if (i >= Xmax - bl.Bwidth_X)
		return;
	if (j >= Ymax - bl.Bwidth_Y)
		return;
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;

	int id_xm = (Xmax * Ymax * k + Xmax * j + (i - 1));
	int id_xp = (Xmax * Ymax * k + Xmax * j + (i + 1));
	real_t Dx = sycl::sqrt(rho[id_xp] / rho[id_xm]), D1x = _DF(1.0) / (Dx + _DF(1.0));
	int id_ym = (Xmax * Ymax * k + Xmax * (j - 1) + i);
	int id_yp = (Xmax * Ymax * k + Xmax * (j + 1) + i);
	real_t Dy = sycl::sqrt(rho[id_yp] / rho[id_ym]), D1y = _DF(1.0) / (Dy + _DF(1.0));
	int id_zm = (Xmax * Ymax * (k - 1) + Xmax * j + i);
	int id_zp = (Xmax * Ymax * (k + 1) + Xmax * j + i);
	real_t Dz = sycl::sqrt(rho[id_zp] / rho[id_zm]), D1z = _DF(1.0) / (Dz + _DF(1.0));

	real_t theta = _DF(0.0);
	if (bl.DimX)
		theta += _DF(1.0);
	if (bl.DimY)
		theta += _DF(1.0);
	if (bl.DimZ)
		theta += _DF(1.0);

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	bool spc = false, spcnan = false, spcs[NUM_SPECIES], spcnans[NUM_SPECIES];
	bool rhonan = (rho[id] < 0) || sycl::isnan(rho[id]) || sycl::isinf(rho[id]);
	real_t *yi = &(y[NUM_SPECIES * id]), *U = &(UI[Emax * id]), _theta = _DF(1.0) / theta;

	if (rhonan)
		error_pos[NUM_SPECIES + 1] = 1;
	for (size_t n2 = 0; n2 < NUM_SPECIES; n2++) // nan yi
	{
		spcs[n2] = (yi[n2] < _DF(1e-20) || yi[n2] > _DF(1.0));
		spcnans[n2] = (sycl::isnan(yi[n2]) || sycl::isinf(yi[n2]));
		spc = spc || spcs[n2], spcnan = spcnan || spcnans[n2];
		if (spc || spcnan)
			error_pos[n2] = 1;
		// 		if (spcs[n2] || spcnans[n2])
		// 		{
		// // #ifdef ERROR_PATCH_YI
		// // 			yi[n2] = _DF(0.0);
		// // 			yi[n2] += (y[n2 + id_xm * NUM_SPECIES] + Dx * y[n2 + id_xp * NUM_SPECIES]) * D1x;
		// // 			yi[n2] += (y[n2 + id_ym * NUM_SPECIES] + Dy * y[n2 + id_yp * NUM_SPECIES]) * D1y;
		// // 			yi[n2] += (y[n2 + id_zm * NUM_SPECIES] + Dz * y[n2 + id_zp * NUM_SPECIES]) * D1z;
		// // 			yi[n2] *= _theta;
		// // #endif // end ERROR_PATCH_YI

		// // #ifdef ERROR_PATCH_YII
		// // 			yi[n2] = _DF(1.0e-20);
		// // #endif // end ERROR_PATCH_YII
		// 		}
	}
	// 	if (spc || spcnan)
	// 	{
	// #ifdef ERROR_PATCH_YI
	// 		// //         if (spcnan)
	// 		// //         {
	// 		// //             T[id] = 0.0;
	// 		// //             u[id] = (u[id_xm] + Dx * u[id_xp]) * D1x;
	// 		// //             T[id] += (T[id_xm] + Dx * T[id_xp]) * D1x;

	// 		// //             v[id] = (v[id_ym] + Dy * v[id_yp]) * D1y;
	// 		// //             T[id] += (T[id_ym] + Dy * T[id_yp]) * D1y;

	// 		// //             w[id] = (w[id_zm] + Dz * w[id_zp]) * D1z;
	// 		// //             T[id] += (T[id_zm] + Dz * T[id_zp]) * D1z;

	// 		// //             T[id] *= _theta ;
	// 		// //             U[1] = U[0] * u[id];
	// 		// //             U[2] = U[0] * v[id];
	// 		// //             U[3] = U[0] * w[id];
	// 		// //         }
	// #endif // end ERROR_PATCH_YI

	// 		real_t sum = _DF(0.0);
	// 		for (size_t nn = 0; nn < NUM_SPECIES; nn++)
	// 			sum += yi[nn];
	// 		sum = _DF(1.0) / sum;
	// 		for (size_t nn = 0; nn < NUM_SPECIES; nn++)
	// 			yi[nn] *= sum;
	// 		for (size_t n = 0; n < NUM_COP; n++)
	// 			U[n + 5] = rho[id] * yi[n];

	// *error_org = true; //, SumPts += 1;
	// 	}
	if (rhonan || spcnan) // add condition to avoid rewrite by other threads
		*error_nan = true, error_pos[NUM_SPECIES + 2] = i, error_pos[NUM_SPECIES + 3] = j, error_pos[NUM_SPECIES + 4] = k;
	// if (i == 100 && j == 100)
	//     SumPts += 1;
}

extern void EstimatePrimitiveVarKernel(int i, int j, int k, Block bl, Thermal thermal, int *error_pos, bool *error1, bool *error2, real_t *UI, real_t *rho,
													 real_t *u, real_t *v, real_t *w, real_t *p, real_t *T, real_t *y, real_t *H, real_t *e, real_t *gamma, real_t *c)
{ // numPte: number of Vars need be posoitive; numVars: length of *Vars(numbers of all Vars need to be estimed).
	int Xmax = bl.Xmax;
	int Ymax = bl.Ymax;
	if (i >= Xmax - bl.Bwidth_X)
		return;
	if (j >= Ymax - bl.Bwidth_Y)
		return;
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	int id_xm = (Xmax * Ymax * k + Xmax * j + (i - 1));
	int id_xp = (Xmax * Ymax * k + Xmax * j + (i + 1));
	real_t Dx = sycl::sqrt(rho[id_xp] / rho[id_xm]), D1x = _DF(1.0) / (Dx + _DF(1.0));
	int id_ym = (Xmax * Ymax * k + Xmax * (j - 1) + i);
	int id_yp = (Xmax * Ymax * k + Xmax * (j + 1) + i);
	real_t Dy = sycl::sqrt(rho[id_yp] / rho[id_ym]), D1y = _DF(1.0) / (Dy + _DF(1.0));
	int id_zm = (Xmax * Ymax * (k - 1) + Xmax * j + i);
	int id_zp = (Xmax * Ymax * (k + 1) + Xmax * j + i);
	real_t Dz = sycl::sqrt(rho[id_zp] / rho[id_zm]), D1z = _DF(1.0) / (Dz + _DF(1.0));

	real_t theta = _DF(1.0), Theta = _DF(0.0);
	if (bl.DimX)
		theta *= _DF(0.5), Theta += _DF(1.0);
	if (bl.DimY)
		theta *= _DF(0.5), Theta += _DF(1.0);
	if (bl.DimZ)
		theta *= _DF(0.5), Theta += _DF(1.0);

	bool ngatve = false, ngatves[3];
	real_t ngaVs[3] = {rho[id], p[id], T[id]}, *ngaPatch[3] = {rho, p, T}, _Theta = _DF(1.0) / Theta;
	for (size_t n1 = 0; n1 < 3; n1++)
	{
		ngatves[n1] = (ngaVs[n1] < 0) || sycl::isnan(ngaVs[n1]) || sycl::isinf(ngaVs[n1]);
		ngatve = ngatve || ngatves[n1];
		if (ngatves[n1])
		{
			error_pos[n1] = 1;
#ifdef ERROR_PATCH // may cause physical unconservative
			ngaVs[n1] = _DF(0.0);
			if (bl.DimX)
				ngaVs[n1] += (ngaPatch[n1][id_xm] + Dx * ngaPatch[n1][id_xp]) * D1x;

			if (bl.DimY)
				ngaVs[n1] += (ngaPatch[n1][id_ym] + Dy * ngaPatch[n1][id_yp]) * D1y;

			if (bl.DimZ)
				ngaVs[n1] += (ngaPatch[n1][id_zm] + Dz * ngaPatch[n1][id_zp]) * D1z;

			ngaVs[n1] *= Theta;
#endif // end ERROR_PATCH
		}
	}
	// ngatve = true;
	if (ngatve) // add condition to avoid rewrite by other threads
		*error1 = true, error_pos[3 + NUM_SPECIES] = i, error_pos[4 + NUM_SPECIES] = j, error_pos[5 + NUM_SPECIES] = k;
}

extern void EstimateFluidNANKernel(int i, int j, int k, int x_offset, int y_offset, int z_offset, Block bl, int *error_pos, real_t *UI, real_t *LUI, bool *error) //, sycl::stream stream_ct1
{
	int Xmax = bl.Xmax;
	int Ymax = bl.Ymax;
	if (i >= Xmax - bl.Bwidth_X)
		return;
	if (j >= Ymax - bl.Bwidth_Y)
		return;
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;
	int id = (Xmax * Ymax * k + Xmax * j + i) * Emax;

	bool tempnegv = UI[0 + id] < 0 ? true : false, tempnans[Emax];
	for (size_t ii = 0; ii < Emax; ii++)
	{
		tempnans[ii] = (sycl::isnan(UI[ii + id]) || sycl::isinf(UI[ii + id])) || (sycl::isnan(LUI[ii + id]) || sycl::isinf(LUI[ii + id]));
		tempnegv = tempnegv || tempnans[ii];
		if (tempnans[ii])
			error_pos[ii] = 1;
	}
	if (tempnegv)
		*error = true, error_pos[Emax + 1] = i, error_pos[Emax + 2] = j, error_pos[Emax + 3] = k;
}
