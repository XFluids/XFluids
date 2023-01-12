#pragma once

#include "schemes_device.hpp"

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
	return (Gamma0 - 1.0) * e; // p/rho;//
#else
	double RNT = RN * T; // unit: J/kg
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
	real_t temp = temp1 / (_dpdrho * _dpdrho * drho * drho + _dpde * de * _dpde * de + Sum_dpdrhoi2 + 1e-19);

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

inline void RoeAverageLeft_x(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _u_c = _u * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _u_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u + _c1);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_u - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case 2:
		eigen_l[0] = _v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = -_DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_l[0] = -_w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _u_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u + _c1);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_u + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_x(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u - _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case 2:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = -_DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(1.0);
		eigen_r[4] = _w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u + _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageLeft_y(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _v_c = _v * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _v_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v + _c1);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_v - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = -_u;
		eigen_l[1] = _DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case 3:
		eigen_l[0] = _w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = -_DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[n + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _v_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v + _c1);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_v + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_y(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v - _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case 3:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = -_DF(1.0);
		eigen_r[4] = -_w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v + _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageLeft_z(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _w_c = _w * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _w_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_w - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = _u;
		eigen_l[1] = -_DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_l[0] = -_v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1; //-q2 + _H;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _w_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_w + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_z(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w - _c;
		eigen_r[4] = _H - _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = -_DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_r[0] = b1;
		eigen_r[1] = b1 * _u;
		eigen_r[2] = b1 * _v;
		eigen_r[3] = b1 * _w;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w + _c;
		eigen_r[4] = _H + _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

#if 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC

#if 1 == EIGEN_ALLOC
inline void RoeAverage_x(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_x(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{

	MARCO_PREEIGEN();

	real_t _u_c = _u * _c1;

	eigen_l[0][0] = _DF(0.5) * (b2 + _u_c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u + _c1);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = (_DF(1.0) - b2 - b3) / b1;
	eigen_l[1][1] = _u;
	eigen_l[1][2] = _v;
	eigen_l[1][3] = _w;
	eigen_l[1][4] = -_DF(1.0);

	eigen_l[2][0] = _v;
	eigen_l[2][1] = _DF(0.0);
	eigen_l[2][2] = -_DF(1.0);
	eigen_l[2][3] = _DF(0.0);
	eigen_l[2][4] = _DF(0.0);

	eigen_l[3][0] = -_w;
	eigen_l[3][1] = _DF(0.0);
	eigen_l[3][2] = _DF(0.0);
	eigen_l[3][3] = _DF(1.0);
	eigen_l[3][4] = _DF(0.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _u_c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u + _c1);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = b1;
	eigen_r[0][2] = _DF(0.0);
	eigen_r[0][3] = _DF(0.0);
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u - _c;
	eigen_r[1][1] = _u * b1;
	eigen_r[1][2] = _DF(0.0);
	eigen_r[1][3] = _DF(0.0);
	eigen_r[1][Emax - 1] = _u + _c;

	eigen_r[2][0] = _v;
	eigen_r[2][1] = _v * b1;
	eigen_r[2][2] = -_DF(1.0);
	eigen_r[2][3] = _DF(0.0);
	eigen_r[2][Emax - 1] = _v;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _w * b1;
	eigen_r[3][2] = _DF(0.0);
	eigen_r[3][3] = _DF(1.0);
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _u * _c;
	eigen_r[4][1] = _H * b1 - _DF(1.0);
	eigen_r[4][2] = -_v;
	eigen_r[4][3] = _w;
	eigen_r[4][Emax - 1] = _H + _u * _c;

	eigen_value[0] = sycl::fabs<real_t>(_u - _c);
	eigen_value[1] = sycl::fabs<real_t>(_u);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_u + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n]; // NOTE: related with yi eigen values
		eigen_l[1][n + Emax - NUM_COP] = z[n];
		eigen_l[2][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[3][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][2] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][3] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}

#if 1 == EIGEN_ALLOC
inline void RoeAverage_y(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_y(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{
	MARCO_PREEIGEN();

	real_t _v_c = _v * _c1;

	// left eigen vectors
	eigen_l[0][0] = _DF(0.5) * (b2 + _v_c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v + _c1);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = -_u;
	eigen_l[1][1] = _DF(1.0);
	eigen_l[1][2] = _DF(0.0);
	eigen_l[1][3] = _DF(0.0);
	eigen_l[1][4] = _DF(0.0);

	eigen_l[2][0] = (_DF(1.0) - b2 - b3) / b1;
	eigen_l[2][1] = _u;
	eigen_l[2][2] = _v;
	eigen_l[2][3] = _w;
	eigen_l[2][4] = -_DF(1.0);

	eigen_l[3][0] = _w;
	eigen_l[3][1] = _DF(0.0);
	eigen_l[3][2] = _DF(0.0);
	eigen_l[3][3] = -_DF(1.0);
	eigen_l[3][4] = _DF(0.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _v_c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v + _c1);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = _DF(0.0);
	eigen_r[0][2] = b1;
	eigen_r[0][3] = _DF(0.0);
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u;
	eigen_r[1][1] = _DF(1.0);
	eigen_r[1][2] = _u * b1;
	eigen_r[1][3] = _DF(0.0);
	eigen_r[1][Emax - 1] = _u;

	eigen_r[2][0] = _v - _c;
	eigen_r[2][1] = _DF(0.0);
	eigen_r[2][2] = _v * b1;
	eigen_r[2][3] = _DF(0.0);
	eigen_r[2][Emax - 1] = _v + _c;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _DF(0.0);
	eigen_r[3][2] = _w * b1;
	eigen_r[3][3] = -_DF(1.0);
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _v * _c;
	eigen_r[4][1] = _u;
	eigen_r[4][2] = _H * b1 - _DF(1.0);
	eigen_r[4][3] = -_w;
	eigen_r[4][Emax - 1] = _H + _v * _c;

	eigen_value[0] = sycl::fabs<real_t>(_v - _c);
	eigen_value[1] = sycl::fabs<real_t>(_v);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_v + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
		eigen_l[1][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[2][n + Emax - NUM_COP] = z[n];
		eigen_l[3][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][2] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][3] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}

#if 1 == EIGEN_ALLOC
inline void RoeAverage_z(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_z(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{
	MARCO_PREEIGEN();

	real_t _w_c = _w * _c1;

	// left eigen vectors
	eigen_l[0][0] = _DF(0.5) * (b2 + _w_c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w + _c1);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = _u;
	eigen_l[1][1] = -_DF(1.0);
	eigen_l[1][2] = _DF(0.0);
	eigen_l[1][3] = _DF(0.0);
	eigen_l[1][4] = _DF(0.0);

	eigen_l[2][0] = -_v;
	eigen_l[2][1] = _DF(0.0);
	eigen_l[2][2] = _DF(1.0);
	eigen_l[2][3] = _DF(0.0);
	eigen_l[2][4] = _DF(0.0);

	eigen_l[3][0] = (_DF(1.0) - b2 - b3) / b1; //-q2 + _H;
	eigen_l[3][1] = _u;
	eigen_l[3][2] = _v;
	eigen_l[3][3] = _w;
	eigen_l[3][4] = -_DF(1.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _w_c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w + _c1);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = _DF(0.0);
	eigen_r[0][2] = _DF(0.0);
	eigen_r[0][3] = b1;
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u;
	eigen_r[1][1] = -_DF(1.0);
	eigen_r[1][2] = _DF(0.0);
	eigen_r[1][3] = b1 * _u;
	eigen_r[1][Emax - 1] = _u;

	eigen_r[2][0] = _v;
	eigen_r[2][1] = _DF(0.0);
	eigen_r[2][2] = _DF(1.0);
	eigen_r[2][3] = b1 * _v;
	eigen_r[2][Emax - 1] = _v;

	eigen_r[3][0] = _w - _c;
	eigen_r[3][1] = _DF(0.0);
	eigen_r[3][2] = _DF(0.0);
	eigen_r[3][3] = b1 * _w;
	eigen_r[3][Emax - 1] = _w + _c;

	eigen_r[4][0] = _H - _w * _c;
	eigen_r[4][1] = -_u;
	eigen_r[4][2] = _v;
	eigen_r[4][3] = _H * b1 - _DF(1.0);
	eigen_r[4][Emax - 1] = _H + _w * _c;

	eigen_value[0] = sycl::fabs<real_t>(_w - _c);
	eigen_value[1] = sycl::fabs<real_t>(_w);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_w + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
		eigen_l[1][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[2][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[3][n + Emax - NUM_COP] = z[n];
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][2] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][3] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}
#endif // end EIGEN_ALLOC