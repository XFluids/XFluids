#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

/**
 * @brief Roe average of value u_l and u_r
 * @brief  _u = (u[id_l] + D * u[id_r]) * D1
 * @param ul left value
 * @param ur right value
 * @param D
 * @return real_t
 */
SYCL_DEVICE real_t get_RoeAverage(const real_t left, const real_t right, const real_t D, const real_t D1)
{
	return (left + D * right) * D1;
}
