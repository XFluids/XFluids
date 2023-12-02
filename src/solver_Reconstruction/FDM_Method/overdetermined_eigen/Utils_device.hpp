#pragma once

#include "Eigen_callback.h"
#include "../../Recon_device.hpp"

// =======================================================
//    get c2 #ifdef COP inside Reconstructflux
inline real_t ReconstructSoundSpeed(Thermal thermal, size_t const id_l, size_t const id_r,
									real_t const D, real_t const D1, real_t const _rho, real_t const _P,
									real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *p, real_t *T, real_t *H,
									real_t *_yi, real_t *z, real_t &b1, real_t &b3, real_t &_k, real_t &_ht, real_t &Gamma)
{
	real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]);
	real_t hi_l[MAX_SPECIES], hi_r[MAX_SPECIES], _hi[MAX_SPECIES], _h = _DF(0.0);
	real_t _dpdrhoi[MAX_SPECIES], drhoi[MAX_SPECIES], z_l[MAX_SPECIES], z_r[MAX_SPECIES];
	real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);
	real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);
	Gamma = get_RoeAverage(gamma_l, gamma_r, D, D1);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		_yi[n] = get_RoeAverage(hi_l[n], hi_r[n], D, D1);
		hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);
		hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);
		_hi[n] = get_RoeAverage(hi_l[n], hi_r[n], D, D1), _h += _hi[n] * _yi[n];
		z_l[n] = (hi_l[n] - (hi_l[n] - T[id_l] * thermal._Wi[n]) * gamma_l) / (gamma_l - _DF(1.0));
		z_r[n] = (hi_r[n] - (hi_r[n] - T[id_r] * thermal._Wi[n]) * gamma_r) / (gamma_r - _DF(1.0));
		z[n] = get_RoeAverage(z_l[n], z_r[n], D, D1);
	}
	real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];
	real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];
	_k = _DF(0.5) * get_RoeAverage(q2_l, q2_r, D, D1), _ht = _h + _k;
	// real_t R_l = get_CopR(thermal._Wi, yi_l), R_r = get_CopR(thermal._Wi, yi_r);
	// real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]), real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);
	// real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l], real_t e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];
	// real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_SPECIES - 1], thermal.Ri[NUM_SPECIES - 1], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),
	// 								get_DpDrho(hi_r[NUM_SPECIES - 1], thermal.Ri[NUM_SPECIES - 1], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);
	// for (size_t nn = 0; nn < NUM_SPECIES - 1; nn++)
	// {
	// 	_dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal.Ri[nn], hi_l[NUM_SPECIES - 1], thermal.Ri[NUM_SPECIES - 1], T[id_l], Cp_l, R_l, gamma_l),
	// 								  get_DpDrhoi(hi_r[nn], thermal.Ri[nn], hi_r[NUM_SPECIES - 1], thermal.Ri[NUM_SPECIES - 1], T[id_r], Cp_r, R_r, gamma_r), D, D1);
	// 	drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];
	// }
	// real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * ((u[id_r] - u[id_l]) * (u[id_r] - u[id_l]) + (v[id_r] - v[id_l]) * (v[id_r] - v[id_l]) + (w[id_r] - w[id_l]) * (w[id_r] - w[id_l]));
	// real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);
	// real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);
	// real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);
	// /*add support while c2<0 use c2 Refed in https://doi.org/10.1006/jcph.1996.5622 */
	// real_t c2w = sycl::step(c2, _DF(0.0)); /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/
	// c2 = Gamma * _P * _rho * c2w + (_DF(1.0) - c2w) * c2;
	real_t c2 = Gamma * _P * _rho;
	b1 = Gamma - _DF(1.0) / c2, b3 = b1 * _k;

	// // return value
	return c2;
}
