#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"

/**
 * @brief Roe average of value u_l and u_r
 * @brief  _u = (u[id_l] + D * u[id_r]) * D1
 * @param ul left value
 * @param ur right value
 * @param D
 * @return real_t
 */
real_t get_RoeAverage(const real_t left, const real_t right, const real_t D, const real_t D1)
{
	return (left + D * right) * D1;
}

/**
 * @brief \frac{\partial p}{\partial \rho}
 * @param hiN: hi[NUM_COP]
 * @param RiN: Ru/thermal.specie_chara[NUM_COP*SPCH_Sz+6]
 * @param q2: u*u+v*v+w*w
 * @param
 * @return real_t
 */
real_t get_DpDrho(const real_t hN, const real_t RN, const real_t q2, const real_t Cp, const real_t R, const real_t T, const real_t e, const real_t gamma)
{
#if CJ
	return (Gamma0 - _DF(1.0)) * e; // p/rho;//
#else
	real_t RNT = RN * T; // unit: J/kg
	return (gamma - _DF(1.0)) * (_DF(0.5) * q2 - hN + Cp * RNT / R);
	// NOTE:return gamma * RNT + (gamma - _DF(1.0)) * (e - hN); // is not right
#endif
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
#if CJ
	return 0; //(Gamma0-1.0)*(-heat_release[n]);
#else
	real_t hN_minus_hi = -hin + hiN;  // unit: J/kg
	real_t Ri_minus_RN = (Rin - RiN); // unit: J/kg/K
	real_t temp = (gamma - _DF(1.0)) * (hN_minus_hi + Cp * Ri_minus_RN * T / R);
	return temp;
#endif
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
real_t get_CopC2(real_t z[NUM_COP], real_t &b1, real_t &b3, real_t const Ri[NUM_SPECIES], real_t const yi[NUM_SPECIES], real_t const hi[NUM_SPECIES], const real_t gamma, const real_t R, const real_t Cp, const real_t T)
{
	real_t _R = _DF(1.0) / R, _dpdrhoi[NUM_SPECIES], _CopC2 = gamma * R * T, temp = _DF(0.0);
	b1 = (gamma - _DF(1.0)) / _CopC2;
	for (size_t n = 0; n < NUM_COP; n++)
		z[n] = (hi[n] - hi[NUM_COP] + Cp * T * (Ri[NUM_COP] - Ri[n]) * _R); // related with yi
	for (size_t nn = 0; nn < NUM_COP; nn++)
		temp += yi[nn] * z[nn];
	b3 = b1 * temp;
	return _CopC2;
}

// /**
//  * form LYX, not right for zi
//  * @brief calculate c^2 of the mixture at given point
//  * // NOTE: realted with yn=yi[0] or yi[N] : hi[] Ri[]
//  */
// real_t get_CopC2(real_t z[NUM_COP], real_t &b1, real_t &b3, real_t const Ri[NUM_SPECIES], real_t const yi[NUM_SPECIES], real_t const hi[NUM_SPECIES], const real_t gamma, const real_t h, const real_t T)
// {
// 	real_t Sum_dpdrhoi = _DF(0.0);				   // Sum_dpdrhoi:first of c2,存在累加项
// 	real_t _dpdrhoi[NUM_SPECIES];
// 	for (size_t n = 0; n < NUM_COP; n++)
// 	{
// 		_dpdrhoi[n] = (gamma - _DF(1.0)) * (hi[NUM_COP] - hi[n]) + gamma * (Ri[n] - Ri[NUM_COP]) * T; // related with yi
// 		z[n] = -_DF(1.0) * _dpdrhoi[n] / (gamma - _DF(1.0));
// 		Sum_dpdrhoi += yi[n] * _dpdrhoi[n];
// 	}
// 	real_t _CopC2 = Sum_dpdrhoi + (gamma - _DF(1.0)) * (h - hi[NUM_COP]) + gamma * Ri[NUM_COP] * T; // related with yi
// 	return _CopC2;
// }