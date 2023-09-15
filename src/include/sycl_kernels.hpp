#pragma once

#include "sycl_devices.hpp"

// extern void XDirThetaItegralKernel(int k, Block bl, real_t *ThetaXeIn, real_t *ThetaN2In, real_t *ThetaXNIn,
//                                                  real_t *ThetaXeOut, real_t *ThetaN2Out, real_t *ThetaXNOut)
// {
//     MARCO_DOMAIN_GHOST();
//     if (k >= Z_inner + Bwidth_Z)
//         return;

//     ThetaXeOut[k] = _DF(0.0), ThetaN2Out[k] = _DF(0.0), ThetaXNOut[k] = _DF(0.0);
//     for (size_t i = bl.Bwidth_Y; i < bl.Ymax; i++)
//     {
//         int id = Xmax * Ymax * k + Xmax * j + i + 1;
//         ThetaXe[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 2];
//         ThetaN2[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 1];
//         ThetaXN[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 2] * yi[NUM_SPECIES * id - 1];
//     }
// }

extern void YDirThetaItegralKernel(int i, int k, Block bl, real_t *y, real_t *ThetaXe, real_t *ThetaN2, real_t *ThetaXN)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	int ii = i - Bwidth_X, kk = k - Bwidth_Z;
	ThetaXe[X_inner * kk + ii] = _DF(0.0), ThetaN2[X_inner * kk + ii] = _DF(0.0), ThetaXN[X_inner * kk + ii] = _DF(0.0);
	for (size_t j = bl.Bwidth_Y; j < bl.Ymax - bl.Bwidth_Y; j++)
	{
		int id = Xmax * Ymax * k + Xmax * j + i;
		real_t *yi = &(y[NUM_SPECIES * id]);
		ThetaXe[X_inner * kk + ii] += yi[bl.Xe_id];
		ThetaN2[X_inner * kk + ii] += yi[bl.N2_id];
		ThetaXN[X_inner * kk + ii] += yi[bl.Xe_id] * yi[bl.N2_id];
	}
}
