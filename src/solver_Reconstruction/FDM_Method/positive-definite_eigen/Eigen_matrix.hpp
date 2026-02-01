#pragma once

#include "Eigen_callback.h"

#if 0 == EIGEN_ALLOC

SYCL_DEVICE inline void RoeAverageLeft_x(int const n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _u_c = _u * _c1;

	if (0 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 + _u_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u + _c1);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_u - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else if (1 == n)
	{
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
	}
	else if (2 == n)
	{
		eigen_l[0] = _v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = -_DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (3 == n)
	{
		eigen_l[0] = -_w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (Emax - 1 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 - _u_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u + _c1);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_u + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else
	{
#ifdef COP
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

SYCL_DEVICE inline void RoeAverageRight_x(int const n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	if (0 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u - _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else if (1 == n)
	{
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
	}
	else if (2 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = -_DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (3 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(1.0);
		eigen_r[4] = _w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (Emax - 1 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u + _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else
	{
#ifdef COP
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

SYCL_DEVICE inline void RoeAverageLeft_y(int const n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _v_c = _v * _c1;

	if (0 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 + _v_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v + _c1);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_v - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else if (1 == n)
	{
		eigen_l[0] = -_u;
		eigen_l[1] = _DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (2 == n)
	{
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
	}
	else if (3 == n)
	{
		eigen_l[0] = _w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = -_DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (Emax - 1 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 - _v_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v + _c1);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_v + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else
	{
#ifdef COP
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

SYCL_DEVICE inline void RoeAverageRight_y(int const n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	if (0 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v - _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else if (1 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (2 == n)
	{
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
	}
	else if (3 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = -_DF(1.0);
		eigen_r[4] = -_w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (Emax - 1 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v + _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else
	{
#ifdef COP
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

SYCL_DEVICE inline void RoeAverageLeft_z(int const n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _w_c = _w * _c1;

	if (0 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 + _w_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_w - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else if (1 == n)
	{
		eigen_l[0] = _u;
		eigen_l[1] = -_DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (2 == n)
	{
		eigen_l[0] = -_v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (3 == n)
	{
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1; //-q2 + _H;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
	}
	else if (Emax - 1 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 - _w_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs(_w + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
	}
	else
	{
#ifdef COP
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

SYCL_DEVICE inline void RoeAverageRight_z(int const n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	if (0 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w - _c;
		eigen_r[4] = _H - _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else if (1 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = -_DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (2 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
	}
	else if (3 == n)
	{
		eigen_r[0] = b1;
		eigen_r[1] = b1 * _u;
		eigen_r[2] = b1 * _v;
		eigen_r[3] = b1 * _w;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
	}
	else if (Emax - 1 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w + _c;
		eigen_r[4] = _H + _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
	}
	else
	{
#ifdef COP
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
#endif // COP
	}
}

#elif 1 == EIGEN_ALLOC

SYCL_DEVICE inline void RoeAverage_x(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
									 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
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

	eigen_value[0] = sycl::fabs(_u - _c);
	eigen_value[1] = sycl::fabs(_u);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs(_u + _c);

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

SYCL_DEVICE inline void RoeAverage_y(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
									 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
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

	eigen_value[0] = sycl::fabs(_v - _c);
	eigen_value[1] = sycl::fabs(_v);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs(_v + _c);

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

SYCL_DEVICE inline void RoeAverage_z(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
									 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
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

	eigen_value[0] = sycl::fabs(_w - _c);
	eigen_value[1] = sycl::fabs(_w);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs(_w + _c);

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
