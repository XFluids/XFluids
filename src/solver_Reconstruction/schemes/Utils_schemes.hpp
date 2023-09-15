#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"

const real_t _six = _DF(1.0) / _DF(6.0);
// TODO: NO std::cmath functions used if schemes function referenced, use sycl::math_function<real_t>

// NOTE: for WENO-CU6
const real_t _ohtz = _DF(1.0) / _DF(120.0);
const real_t _ohff = _DF(1.0) / _DF(144.0);
const real_t _ftss = _DF(1.0) / _DF(5760.0);
// const real_t _twfr = _DF(1.0) / _DF(24.0); // defined in marco.h
// const real_t _sxtn = _DF(1.0) / _DF(16.0); // defined in marco.h
const real_t wu6a2a2 = _DF(13.0) / _DF(3.0), wu6a3a3 = _DF(3129.0) / _DF(80.0);
const real_t wu6a4a4 = _DF(87617.0) / _DF(140.0), wu6a3a5 = _DF(14127.0) / _DF(224.0), wu6a5a5 = _DF(252337135.0) / _DF(16128.0);

/**
 * @brief Fast Inverse Square Root by Jhon Carmack
 *
 * @param number
 * @return 1.0f/sqrt(number)
 */
float CarmackRsqrt(float number)
{
	float y = number;
	float x2 = number * 0.5F;
	uint32_t i = *(uint32_t *)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float *)&i;
	y = y * (1.5F - (x2 * y * y));

	return y;
}

/**
 * @brief Fast Inverse based Jhon Carmack
 *
 * @param number
 * @return 1.0f/number
 */
float CarmackR(float number)
{
	float y = CarmackRsqrt(number);
	return y * y;
}
