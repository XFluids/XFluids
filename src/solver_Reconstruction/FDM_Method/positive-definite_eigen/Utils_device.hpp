#pragma once

#include "Eigen_callback.h"
#include "../../Recon_device.hpp"

/**
 * @brief \frac{\partial p}{\partial \rho}
 * @param hiN: hi[NUM_SPECIES-1]
 * @param RiN: Ru/thermal.specie_chara[NUM_COP*SPCH_Sz+6]
 * @param q2: u*u+v*v+w*w
 * @param
 * @return real_t
 */
real_t get_DpDrho(const real_t hN, const real_t RN, const real_t q2, const real_t Cp, const real_t R, const real_t T, const real_t e, const real_t gamma)
{
	real_t RNT = RN * T; // unit: J/kg
	return (gamma - _DF(1.0)) * (_DF(0.5) * q2 - hN + Cp * RNT / R);
	// NOTE:return gamma * RNT + (gamma - _DF(1.0)) * (e - hN); // is not right
}

/**
 * @brief \frac{\partial p}{\partial \rho_i}
 * @param hin: hi of the n-th species
 * @param hiN: hi of the N-th species
 * @param Cp: get_CopCp for mixture
 * @param R: get_CopR for mixture
 * @return real_t
 */
real_t get_DpDrhoi(const real_t hin, const real_t Rin, const real_t hiN, const real_t RiN, const real_t T, const real_t Cp, const real_t R, const real_t gamma)
{
	real_t hN_minus_hi = -hin + hiN;  // unit: J/kg
	real_t Ri_minus_RN = (Rin - RiN); // unit: J/kg/K
	real_t temp = (gamma - _DF(1.0)) * (hN_minus_hi + Cp * Ri_minus_RN * T / R);
	return temp;
}

/**
 * @brief compute Roe-averaged sound speed of multicomponent flows
 * @param zi: for eigen matrix
 * @return real_t c*c
 */
real_t SoundSpeedMultiSpecies(real_t *zi, real_t &b1, real_t &b3, real_t *_Yi, real_t *_dpdrhoi, real_t *drhoi, const real_t _dpdrho, const real_t _dpde,
							  const real_t _dpdE, const real_t _prho, const real_t dp, const real_t drho, const real_t de, const real_t _rho)
{
	// sum
	real_t _dpdrhoi_new[NUM_COP], Sum_dpdrhoi = _DF(0.0), Sum_drhoi = _DF(0.0), Sum_dpdrhoi2 = _DF(0.0), Sum_Yidpdrhoi = _DF(0.0);
	for (int n = 0; n < NUM_COP; n++)
	{
		Sum_dpdrhoi += _dpdrhoi[n] * drhoi[n];
		Sum_dpdrhoi2 += _dpdrhoi[n] * drhoi[n] * _dpdrhoi[n] * drhoi[n];
	}
	// method 1
	real_t temp1 = dp - (_dpdrho * drho + _dpde * de + Sum_dpdrhoi);
	real_t temp = temp1 / (_dpdrho * _dpdrho * drho * drho + _dpde * de * _dpde * de + Sum_dpdrhoi2 + _DF(1e-19));

	for (int n = 0; n < NUM_COP; n++)
	{
		Sum_drhoi += drhoi[n] * drhoi[n];
		Sum_Yidpdrhoi += _Yi[n] * _dpdrhoi[n];
	}

	real_t _dpdE_new = _dpdE + _dpdE * _dpdE * de * _rho * temp;
	real_t _dpdrho_new = _dpdrho + _dpdrho * _dpdrho * drho * temp;

	// sound speed
	for (int n = 0; n < NUM_COP; n++)
		_dpdrhoi_new[n] = _dpdrhoi[n] + _dpdrhoi[n] * _dpdrhoi[n] * drhoi[n] * temp;

	real_t csqr = _dpdrho_new + _dpdE_new * _prho + Sum_Yidpdrhoi;
	b1 = _dpdE_new / csqr;
	for (int n = 0; n < NUM_COP; n++)
	{
		zi[n] = -_dpdrhoi_new[n] / _dpdE_new;
		b3 += _Yi[n] * zi[n];
	}
	b3 *= b1;

	return csqr;
}

/**
 * @brief calculate c^2, b1, b3 of the mixture at given point
 * ref.High Accuracy Numerical Methods for Thermally Perfect Gas Flows with Chemistry.https://doi.org/10.1006/jcph.1996.5622
 * // NOTE: realted with yn=yi[0] or yi[N] : hi[] Ri[]
 */
real_t get_CopC2(real_t *z, real_t &b1, real_t &b3, real_t const *Ri, real_t const *yi, real_t const *hi,
				 const real_t gamma, const real_t R, const real_t Cp, const real_t T)
{
	size_t NUM_LOOP = NUM_SPECIES - 1;
	real_t _R = _DF(1.0) / R, _dpdrhoi[NUM_SPECIES], _CopC2 = gamma * R * T, temp = _DF(0.0);
	b1 = (gamma - _DF(1.0)) / _CopC2;
	for (size_t n = 0; n < NUM_LOOP; n++)
		z[n] = (hi[n] - hi[NUM_LOOP] + Cp * T * (Ri[NUM_LOOP] - Ri[n]) * _R); // related with yi
	for (size_t nn = 0; nn < NUM_LOOP; nn++)
		temp += yi[nn] * z[nn];
	b3 = b1 * temp;
	return _CopC2;
}

// =======================================================
//    get c2 #ifdef COP inside Reconstructflux
inline real_t ReconstructSoundSpeed(Thermal thermal, size_t const id_l, size_t const id_r,
									real_t const D, real_t const D1, real_t const _rho, real_t const _P,
									real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *p, real_t *T, real_t *H,
									real_t *_yi, real_t *z, real_t &b1, real_t &b3, real_t &k, real_t &_ht, real_t &Gamma)
{
	real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]);
	real_t hi_l[MAX_SPECIES], hi_r[MAX_SPECIES], _dpdrhoi[MAX_SPECIES], drhoi[MAX_SPECIES];
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);
		hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);
		_yi[n] = (yi_l[n] + D * yi_r[n]) * D1; /*_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;*/
	}
	real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);
	real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);
	Gamma = get_RoeAverage(gamma_l, gamma_r, D, D1);
	real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];
	real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];
	real_t R_l = get_CopR(thermal._Wi, yi_l), R_r = get_CopR(thermal._Wi, yi_r);
	real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]), Cp_r = get_CopCp(thermal, yi_r, T[id_r]);
	real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l], e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];
	real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], thermal.Ri[NUM_COP], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),
									get_DpDrho(hi_r[NUM_COP], thermal.Ri[NUM_COP], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);
	for (size_t nn = 0; nn < NUM_COP; nn++)
	{
		_dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal.Ri[nn], hi_l[NUM_COP], thermal.Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),
									  get_DpDrhoi(hi_r[nn], thermal.Ri[nn], hi_r[NUM_COP], thermal.Ri[NUM_COP], T[id_r], Cp_r, R_r, gamma_r), D, D1);
		drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];
	}
	real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * ((u[id_r] - u[id_l]) * (u[id_r] - u[id_l]) + (v[id_r] - v[id_l]) * (v[id_r] - v[id_l]) + (w[id_r] - w[id_l]) * (w[id_r] - w[id_l]));
	real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);
	real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);
	real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);
	/*add support while c2<0 use c2 Refed in https://doi.org/10.1006/jcph.1996.5622 */
	real_t c2w = sycl::step(c2, _DF(0.0)); /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/
	c2 = Gamma * _P * _rho * c2w + (_DF(1.0) - c2w) * c2;
	// // return value
	return c2;
}
