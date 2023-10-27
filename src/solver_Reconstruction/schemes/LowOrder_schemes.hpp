#pragma once

#include "Utils_schemes.hpp"

/**
 * @brief the minmod limiter
 *
 * @param r
 * @return real_t
 */
inline real_t minmod(real_t r)
{
	real_t minmod = 0;
	real_t aa = 1.0;
	if (r > 0)
		minmod = std::min(r, aa);
	return minmod;
}

/**
 * @brief van Leer limiter
 *
 * @param r
 * @return real_t
 */
inline real_t van_Leer(real_t r)
{
	return (r + sycl::fabs(r)) / (1.0 + sycl::fabs(r));
}

/**
 * @brief van Albada limiter
 *
 * @param r
 * @return real_t
 */
inline real_t van_Albada(real_t r)
{
	return (r * r + r) / (1.0 + r * r);
}

/**
 * @brief the MUSCL reconstruction
 *
 * @param p
 * @param LL
 * @param RR
 * @param flag
 */
inline void MUSCL(real_t p[4], real_t &LL, real_t &RR, int flag)
{
	real_t tol = 1e-20, k = 1.0 / 3.0;
	real_t a0 = p[1] - p[0], a1 = p[2] - p[1], a2 = p[3] - p[2];
	if (a0 == -tol || a2 == -tol || a1 == -tol)
		tol *= 0.1;

	real_t r1 = a1 / (a0 + tol), r2 = a1 / (a2 + tol), r11 = a0 / (a1 + tol), r22 = a2 / (a1 + tol);
	real_t LL1 = 0, LL2 = 0, LR1 = 0, LR2 = 0;
	switch (flag)
	{
	case 1:
		LL1 = minmod(r1);
		LL2 = minmod(r11);
		LR1 = minmod(r2);
		LR2 = minmod(r22);
		break;
	case 2:
		LL1 = van_Leer(r1);
		LL2 = van_Leer(r11);
		LR1 = van_Leer(r2);
		LR2 = van_Leer(r22);
		break;
	case 3:
		LL1 = van_Albada(r1);
		LL2 = van_Albada(r11);
		LR1 = van_Albada(r2);
		LR2 = van_Albada(r22);
		break;
	}
	LL = p[1] + 0.25 * ((1.0 - k) * LL1 + (1.0 + k) * LL2 * r1) * a0;
	RR = p[2] - 0.25 * ((1.0 - k) * LR1 + (1.0 + k) * LR2 * r2) * a2;
}

/**
 * @brief KroneckerDelta
 * @return real_t
 */
inline real_t KroneckerDelta(const int i, const int j)
{
	real_t f = i == j ? 1 : 0;
	return f;
}
