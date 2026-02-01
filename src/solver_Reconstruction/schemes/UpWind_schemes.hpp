#pragma once

#include "Utils_schemes.hpp"

/**
 * @brief upwind scheme
 *
 * @param f
 * @param delta
 * @return real_t
 */
SYCL_DEVICE inline real_t upwind_P(real_t *f, real_t delta)
{
	return *f;
}
SYCL_DEVICE inline real_t upwind_M(real_t *f, real_t delta)
{
	return *(f + 1);
}

SYCL_DEVICE inline real_t linear_3rd_P(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	real_t vv = -(-2.0 * v4 - 5.0 * v3 + v2) / 6.0;

	return vv;
}
SYCL_DEVICE inline real_t linear_3rd_M(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	real_t vv = (-v5 + 5.0 * v4 + 2 * v3) / 6.0;

	return vv;
}

SYCL_DEVICE inline real_t linear_5th_P(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	real_t vv = (2.0 * v1 - 13.0 * v2 + 47.0 * v3 + 27.0 * v4 - 3.0 * v5) / 60.0;

	return vv;
}

SYCL_DEVICE inline real_t linear_5th_M(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	real_t vv = (2.0 * v6 - 13.0 * v5 + 47.0 * v4 + 27.0 * v3 - 3.0 * v2) / 60.0;

	return vv;
}

SYCL_DEVICE inline real_t linear_2th(real_t *f, real_t delta)
{
	real_t v1 = *f;
	real_t v2 = *(f + 1);
	real_t vv = (v1 + v2) / 2.0;

	return vv;
}

SYCL_DEVICE inline real_t linear_4th(real_t *f, real_t delta)
{
	real_t v1 = *(f - 1);
	real_t v2 = *f;
	real_t v3 = *(f + 1);
	real_t v4 = *(f + 2);
	real_t vv = (-v1 + 7.0 * v2 + 7.0 * v3 - v4) / 12.0;

	return vv;
}

SYCL_DEVICE inline real_t linear_6th(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	real_t vv = (v1 - 8.0 * v2 + 37.0 * v3 + 37.0 * v4 - 8.0 * v5 + v6) / 60.0;

	return vv;
}

//-------------------------------------------------------------------------------------------------
//  linear scheme for hybrid method
//-------------------------------------------------------------------------------------------------
SYCL_DEVICE inline real_t du_upwind5(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	return (v1 - 5.0 * v2 + 10.0 * v3 - 10.0 * v4 + 5.0 * v5 - v6) / 60.0;
}
//-------------------------------------------------------------------------------------------------
//  linear scheme for hybrid method
//-------------------------------------------------------------------------------------------------
SYCL_DEVICE inline real_t f2_upwind5(real_t *f, real_t delta)
{
	real_t v1 = *(f - 2);
	real_t v2 = *(f - 1);
	real_t v3 = *f;
	real_t v4 = *(f + 1);
	real_t v5 = *(f + 2);
	real_t v6 = *(f + 3);
	return (v1 - 8.0 * v2 + 37.0 * v3 + 37.0 * v4 - 8.0 * v5 + v6) / 60.0;
}
