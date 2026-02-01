#pragma once

#include "Recon_device.hpp"

extern void PositivityPreservingKernel(int i, int j, int k, int id_l, int id_r, Block bl, Thermal thermal,
									   real_t *UI, real_t *Fl, real_t *Fwall, real_t *T,
									   const real_t lambda_0, const real_t lambda, const real_t *epsilon) // , sycl::stream stream epsilon[NUM_SPECIES+2]={rho, e, y(0), ..., y(n)}
{
	if (i >= bl.Xmax - bl.Bwidth_X)
		return;
	if (j >= bl.Ymax - bl.Bwidth_Y)
		return;
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;

	real_t T_l = T[id_l], T_r = T[id_r];
	id_l *= Emax, id_r *= Emax;
	// stream << "eps: " << epsilon[0] << " " << epsilon[3] << " " << epsilon[5] << " " << epsilon[NUM_SPECIES + 1] << "\n";
	// int id_l = (Xmax * Ymax * k + Xmax * j + i) * Emax;
	// int id_r = (Xmax * Ymax * k + Xmax * j + i + 1) * Emax;
	// Positivity preserving flux limiter form Dr.Hu.https://www.sciencedirect.com/science/article/pii/S0021999113000557, expand to multicomponent, need positivity Initial value
	real_t rho_min, theta, theta_u, theta_p, F_LF[Emax], FF_LF[Emax], FF[Emax], *UU = &(UI[id_l]), *UP = &(UI[id_r]); // UU[Emax], UP[Emax],
	for (int n = 0; n < Emax; n++)
	{
		F_LF[n] = _DF(0.5) * (Fl[n + id_l] + Fl[n + id_r] + lambda_0 * (UI[n + id_l] - UI[n + id_r])); // FF_LF == F_(i+1/2) of Lax-Friedrichs
		FF_LF[n] = _DF(2.0) * lambda * F_LF[n];														   // U_i^+ == (U_i^n-FF_LF)
		FF[n] = _DF(2.0) * lambda * Fwall[n + id_l];												   // FF from original high accuracy scheme
	}

	// // correct for positive density
	theta_u = _DF(1.0), theta_p = _DF(1.0);
	rho_min = sycl::min(UU[0], epsilon[0]);
	if (UU[0] - FF[0] < rho_min)
		theta_u = (UU[0] - FF_LF[0] - rho_min) / (FF[0] - FF_LF[0]);
	rho_min = sycl::min(UP[0], epsilon[0]);
	if (UP[0] + FF[0] < rho_min)
		theta_p = (UP[0] + FF_LF[0] - rho_min) / (FF_LF[0] - FF[0]);
	theta = sycl::min(theta_u, theta_p);
	for (int n = 0; n < Emax; n++)
	{
		FF[n] = (_DF(1.0) - theta) * FF_LF[n] + theta * FF[n];
		Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
	}

	// // correct for yi // need Ini of yi > 0(1.0e-10 set)
	real_t yi_q[NUM_SPECIES], yi_u[NUM_SPECIES], yi_qp[NUM_SPECIES], yi_up[NUM_SPECIES], _rhoq, _rhou, _rhoqp, _rhoup;
	_rhoq = _DF(1.0) / (UU[0] - FF[0]), _rhou = _DF(1.0) / (UU[0] - FF_LF[0]);	 //, yi_q[NUM_COP] = _DF(1.0), yi_u[NUM_COP] = _DF(1.0);
	_rhoqp = _DF(1.0) / (UP[0] + FF[0]), _rhoup = _DF(1.0) / (UP[0] + FF_LF[0]); //, yi_qp[NUM_COP] = _DF(1.0), yi_up[NUM_COP] = _DF(1.0);
	for (size_t n = 0; n < NUM_COP; n++)
	{
		int tid = n + 5;
		yi_q[n] = (UU[tid] - FF[tid]) * _rhoq;		//, yi_q[NUM_COP] -= yi_q[n];
		yi_u[n] = (UU[tid] - FF_LF[tid]) * _rhou;	//, yi_u[NUM_COP] -= yi_u[n];
		yi_qp[n] = (UP[tid] + FF[tid]) * _rhoqp;	//, yi_qp[NUM_COP] -= yi_qp[n];
		yi_up[n] = (UP[tid] + FF_LF[tid]) * _rhoup; //, yi_up[NUM_COP] -= yi_up[n];
		theta_u = _DF(1.0), theta_p = _DF(1.0);
		real_t temp = epsilon[n + 2];
		if (yi_q[n] < temp)
		{
			real_t yi_min = sycl::min(yi_u[n], temp);
			theta_u = (yi_u[n] - yi_min + _DF(1.0e-40)) / (yi_u[n] - yi_q[n] + _DF(1.0e-40));
		}
		if (yi_qp[n] < temp)
		{
			real_t yi_min = sycl::min(yi_up[n], temp);
			theta_p = (yi_up[n] - yi_min + _DF(1.0e-40)) / (yi_up[n] - yi_qp[n] + _DF(1.0e-40));
		}
		theta = sycl::min(theta_u, theta_p);
		for (int nn = 0; nn < Emax; nn++)
			Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
	}
}
