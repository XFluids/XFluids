#pragma once

#include "Eigen_callback.h"

/**
 * @param c:  artificial sound speed at the point of mesh(i+1/2)
 * @param k:  k = _DF(0.5) * (u * u + v * v + w * w)
 * @param X:  X=b2-b1*ht=b2-b1*(h+k), h=sum(yi[i]*hi[i])
 */
SYCL_DEVICE inline void RoeAverageLeft_x(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi,
										 real_t const c, real_t const u, real_t const v, real_t const w, real_t const k,
										 real_t const b1, real_t const b2, real_t const X)
{ // the n th row of Left Eigen matrix
	real_t _c = _DF(1.0) / c;
	if (0 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 + u * _c);
		eigen_l[1] = -_DF(0.5) * (b1 * u + _c);
		eigen_l[2] = -_DF(0.5) * b1 * v;
		eigen_l[3] = -_DF(0.5) * b1 * w;
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = u - c;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = _DF(0.5) * b1 * z[i];
#endif // end COP
	}
	else if (1 == n)
	{
		eigen_l[0] = _DF(1.0) - b2;
		eigen_l[1] = b1 * u;
		eigen_l[2] = b1 * v;
		eigen_l[3] = b1 * w;
		eigen_l[4] = -b1;
		eigen_value = u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = -b1 * z[i];
#endif // end COP
	}
	else if (2 == n)
	{
		eigen_l[0] = _DF(0.5) * (b2 - u * _c);
		eigen_l[1] = -_DF(0.5) * (b1 * u - _c);
		eigen_l[2] = -_DF(0.5) * b1 * v;
		eigen_l[3] = -_DF(0.5) * b1 * w;
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = u + c;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = _DF(0.5) * b1 * z[i];
#endif // end COP
	}
	else if (3 == n)
	{
		eigen_l[0] = -v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = _DF(0.0);
#endif // end COP
	}
	else if (4 == n)
	{
		eigen_l[0] = -w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = _DF(0.0);
#endif // end COP
	}
	else if (5 == n)
	{
		eigen_l[0] = k * (_DF(1.0) + X);
		eigen_l[1] = -u * (_DF(1.0) + X);
		eigen_l[2] = -v * (_DF(1.0) + X);
		eigen_l[3] = -w * (_DF(1.0) + X);
		eigen_l[4] = _DF(1.0) + X;
		eigen_value = u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = X * z[i];
#endif // end COP
	}
	else
	{
#ifdef COP
		eigen_l[0] = -b2 * yi[n - 5];
		eigen_l[1] = b1 * u * yi[n - 5];
		eigen_l[2] = b1 * v * yi[n - 5];
		eigen_l[3] = b1 * w * yi[n - 5];
		eigen_l[4] = -b1 * yi[n - 5];
		eigen_value = u;
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_l[5 + i] = n - 6 == i ? _DF(1.0) - b1 * yi[n - 6] * z[i] : -b1 * yi[n - 6] * z[i];
#endif // end COP
	}
}

/**
 * @param c: artificial sound speed at the point of mesh(i+1/2)
 */
SYCL_DEVICE inline void RoeAverageRight_x(int n, real_t *eigen_r, real_t *z, const real_t *yi,
										  real_t const c, real_t const u, real_t const v, real_t const w, real_t const k, real_t const ht)
{
	real_t zn = _DF(1.0) / z[NUM_SPECIES - 1];
	if (0 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = u - c;
		eigen_r[2] = v;
		eigen_r[3] = w;
		eigen_r[4] = ht - c * u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = yi[i];
#endif // end COP
	}
	else if (1 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = u;
		eigen_r[2] = v;
		eigen_r[3] = w;
		eigen_r[4] = k;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = _DF(0.0);
#endif // end COP
	}
	else if (2 == n)
	{
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = u + c;
		eigen_r[2] = v;
		eigen_r[3] = w;
		eigen_r[4] = ht + c * u;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = yi[i];
#endif // end COP
	}
	else if (3 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = v;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = _DF(0.0);
#endif // end COP
	}
	else if (4 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = w;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = _DF(0.0);
#endif // end COP
	}
	else if (5 == n)
	{
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _DF(1.0);
		eigen_r[Emax - 1] = -zn;
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = _DF(0.0);
#endif // end COP
	}
	else
	{
#ifdef COP
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _DF(0.0);
		eigen_r[Emax - 1] = -zn * z[n - 6];
		for (size_t i = 0; i < NUM_SPECIES; i++)
			eigen_r[5 + i] = n - 6 == i ? _DF(1.0) : _DF(0.0);
#endif // end COP
	}
}

SYCL_DEVICE inline void RoeAverageLeft_y(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi,
										 real_t const c, real_t const u, real_t const v, real_t const w, real_t const k,
										 real_t const b1, real_t const b2, real_t const X)
{
	// 	if (0 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (1 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (2 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (3 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (4 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else
	// 	{
	// #ifdef COP
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
}

SYCL_DEVICE inline void RoeAverageRight_y(int n, real_t *eigen_r, real_t *z, const real_t *yi,
										  real_t const c, real_t const u, real_t const v, real_t const w, real_t const k, real_t const ht)
{
	// 	if (0 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (1 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (2 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (3 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (4 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else
	// 	{
	// #ifdef COP
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
}

SYCL_DEVICE inline void RoeAverageLeft_z(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi,
										 real_t const c, real_t const u, real_t const v, real_t const w, real_t const k,
										 real_t const b1, real_t const b2, real_t const X)
{
	// 	if (0 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (1 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (2 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (3 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (4 == n)
	// 	{
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else
	// 	{
	// #ifdef COP
	// 		eigen_l[0] = ;
	// 		eigen_l[1] = ;
	// 		eigen_l[2] = ;
	// 		eigen_l[3] = ;
	// 		eigen_l[4] = ;
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_l[5 + i] = ;
	// #endif // end COP
	// 	}
}

SYCL_DEVICE inline void RoeAverageRight_z(int n, real_t *eigen_r, real_t *z, const real_t *yi,
										  real_t const c, real_t const u, real_t const v, real_t const w, real_t const k, real_t const ht)
{
	// 	if (0 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (1 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (2 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (3 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else if (4 == n)
	// 	{
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// #ifdef COP
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
	// 	else
	// 	{
	// #ifdef COP
	// 		eigen_r[0] = ;
	// 		eigen_r[1] = ;
	// 		eigen_r[2] = ;
	// 		eigen_r[3] = ;
	// 		eigen_r[4] = ;
	// 		for (size_t i = 0; i < NUM_SPECIES; i++)
	// 			eigen_r[5 + i] = ;
	// #endif // end COP
	// 	}
}
