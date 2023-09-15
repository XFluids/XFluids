#pragma once

#include "Utils_schemes.hpp"

/**
 * @brief  WENO-AO(5,3) scheme from
 *                  Balsara et al., An efficient class of WENO schemes with adaptive order. (2016)
 *                  Kumar et al., Simple smoothness indicator and multi-level adaptive order WENO scheme for hyperbolic conservation laws. (2018)
 */
inline real_t WENOAO53_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t a1, a2, a3, a5, w1, w2, w3, w5;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v1 - 2.0 * v2 + 3.0 / 2.0 * v3, s12 = 1.0 / 2.0 * v1 - 1.0 * v2 + 1.0 / 2.0 * v3;
	real_t s21 = -1.0 / 2.0 * v2 + 1.0 / 2.0 * v4, s22 = 1.0 / 2.0 * v2 - v3 + 1.0 / 2.0 * v4;
	real_t s31 = -3.0 / 2.0 * v3 + 2.0 * v4 - 1.0 / 2.0 * v5, s32 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s51 = 11.0 / 120.0 * v1 - 82.0 / 120.0 * v2 + 82.0 / 120.0 * v4 - 11.0 / 120.0 * v5;
	real_t s52 = -3.0 / 56.0 * v1 + 40.0 / 56.0 * v2 + -74.0 / 56.0 * v3 + 40.0 / 56.0 * v4 - 3.0 / 56.0 * v5;
	real_t s53 = -1.0 / 12.0 * v1 + 2.0 / 12.0 * v2 - 2.0 / 12.0 * v4 + 1.0 / 12.0 * v5;
	real_t s54 = 1.0 / 24.0 * v1 - 4.0 / 24.0 * v2 + 6.0 / 24.0 * v3 - 4.0 / 24.0 * v4 + 1.0 / 24.0 * v5;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s5 = 1.0 * (s51 + 1.0 / 10.0 * s53) * (s51 + 1.0 / 10.0 * s53) + 13.0 / 3.0 * (s52 + 123.0 / 455.0 * s54) * (s52 + 123.0 / 455.0 * s54) + 781.0 / 20.0 * s53 * s53 + 1421461.0 / 2275.0 * s54 * s54;

	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau = (sycl::fabs(s5 - s1) + sycl::fabs(s5 - s2) + sycl::fabs(s5 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_5 = 0.85;
	a1 = coef_weights_1 * (1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2 = coef_weights_2 * (1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3 = coef_weights_3 * (1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a5 = coef_weights_5 * (1.0 + (tau * tau) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a5);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;
	w5 = a5 * tw1;

	// Compute coefficients of the Legendre basis polynomial
	real_t u0 = v3;
	real_t u1 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1 * s11 + w2 * s21 + w3 * s31;
	real_t u2 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1 * s12 + w2 * s22 + w3 * s32;
	real_t u3 = (w5 / coef_weights_5) * s53;
	real_t u4 = (w5 / coef_weights_5) * s54;
	// Return value of reconstructed polynomial
	return u0 + u1 * 1.0 / 2.0 + u2 * 1.0 / 6.0 + u3 * 1.0 / 20.0 + u4 * 1.0 / 70.0;
}

inline real_t WENOAO53_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t a1, a2, a3, a5, w1, w2, w3, w5;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v1 - 2.0 * v2 + 3.0 / 2.0 * v3, s12 = 1.0 / 2.0 * v1 - 1.0 * v2 + 1.0 / 2.0 * v3;
	real_t s21 = -1.0 / 2.0 * v2 + 1.0 / 2.0 * v4, s22 = 1.0 / 2.0 * v2 - v3 + 1.0 / 2.0 * v4;
	real_t s31 = -3.0 / 2.0 * v3 + 2.0 * v4 - 1.0 / 2.0 * v5, s32 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s51 = 11.0 / 120.0 * v1 - 82.0 / 120.0 * v2 + 82.0 / 120.0 * v4 - 11.0 / 120.0 * v5;
	real_t s52 = -3.0 / 56.0 * v1 + 40.0 / 56.0 * v2 + -74.0 / 56.0 * v3 + 40.0 / 56.0 * v4 - 3.0 / 56.0 * v5;
	real_t s53 = -1.0 / 12.0 * v1 + 2.0 / 12.0 * v2 - 2.0 / 12.0 * v4 + 1.0 / 12.0 * v5;
	real_t s54 = 1.0 / 24.0 * v1 - 4.0 / 24.0 * v2 + 6.0 / 24.0 * v3 - 4.0 / 24.0 * v4 + 1.0 / 24.0 * v5;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s5 = 1.0 * (s51 + 1.0 / 10.0 * s53) * (s51 + 1.0 / 10.0 * s53) + 13.0 / 3.0 * (s52 + 123.0 / 455.0 * s54) * (s52 + 123.0 / 455.0 * s54) + 781.0 / 20.0 * s53 * s53 + 1421461.0 / 2275.0 * s54 * s54;

	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau = (sycl::fabs(s5 - s1) + sycl::fabs(s5 - s2) + sycl::fabs(s5 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_5 = 0.85;
	a1 = coef_weights_1 * (1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2 = coef_weights_2 * (1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3 = coef_weights_3 * (1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a5 = coef_weights_5 * (1.0 + (tau * tau) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a5);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;
	w5 = a5 * tw1;

	// Compute coefficients of the Legendre basis polynomial
	real_t u0 = v3;
	real_t u1 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1 * s11 + w2 * s21 + w3 * s31;
	real_t u2 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1 * s12 + w2 * s22 + w3 * s32;
	real_t u3 = (w5 / coef_weights_5) * s53;
	real_t u4 = (w5 / coef_weights_5) * s54;
	// Return value of reconstructed polynomial
	return u0 + u1 * 1.0 / 2.0 + u2 * 1.0 / 6.0 + u3 * 1.0 / 20.0 + u4 * 1.0 / 70.0;
}

/**
 * @brief  WENO-AO(7,3) scheme from Balsara (2016)
 */
inline real_t WENOAO73_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6, v7;
	real_t u0, u1, u2, u3, u4, u5, u6;
	real_t a1, a2, a3, a7;
	real_t w1, w2, w3, w7;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 3);
	v2 = *(f + k - 2);
	v3 = *(f + k - 1);
	v4 = *(f + k);
	v5 = *(f + k + 1);
	v6 = *(f + k + 2);
	v7 = *(f + k + 3);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v2 - 2.0 * v3 + 3.0 / 2.0 * v4, s12 = 1.0 / 2.0 * v2 - 1.0 * v3 + 1.0 / 2.0 * v4;
	real_t s21 = -1.0 / 2.0 * v3 + 1.0 / 2.0 * v5, s22 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s31 = -3.0 / 2.0 * v4 + 2.0 * v5 - 1.0 / 2.0 * v6, s32 = 1.0 / 2.0 * v4 - v5 + 1.0 / 2.0 * v6;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s71 = -191.0 / 10080.0 * v1 + 1688.0 / 10080.0 * v2 - 7843.0 / 10080.0 * v3 + 7843.0 / 10080.0 * v5 - 1688.0 / 10080.0 * v6 + 191.0 / 10080.0 * v7;
	real_t s72 = 79.0 / 10080.0 * v1 - 1014.0 / 10080.0 * v2 + 8385.0 / 10080.0 * v3 - 14900.0 / 10080.0 * v4 + 8385.0 / 10080.0 * v5 - 1014.0 / 10080.0 * v6 + 79.0 / 10080.0 * v7;
	real_t s73 = 5.0 / 216.0 * v1 - 38.0 / 216.0 * v2 + 61.0 / 216.0 * v3 - 61.0 / 216.0 * v5 + 38.0 / 216.0 * v6 - 5.0 / 216.0 * v7;
	real_t s74 = -13.0 / 1584.0 * v1 + 144.0 / 1584.0 * v2 + 459.0 / 1584.0 * v3 + 656.0 / 1584.0 * v4 - 459.0 / 1584.0 * v5 + 144.0 / 1584.0 * v6 - 13.0 / 1584.0 * v7;
	real_t s75 = -1.0 / 240.0 * v1 + 4.0 / 240.0 * v2 - 5.0 / 240.0 * v3 + 5.0 / 240.0 * v5 - 4.0 / 240.0 * v6 + 1 / 240.0 * v7;
	real_t s76 = 1.0 / 720.0 * v1 - 6.0 / 720.0 * v2 + 15.0 / 720.0 * v3 - 20.0 / 720.0 * v4 + 15.0 / 720.0 * v5 - 6.0 / 720.0 * v6 + 1 / 720.0 * v7;
	real_t s7 = 1.0 * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) + 13.0 / 3.0 * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) + 781.0 / 20.0 * (s73 + 26045.0 / 49203.0 * s75) * (s73 + 26045.0 / 49203.0 * s75) + 1421461.0 / 2275.0 * (s74 + 81596225.0 / 93816426.0 * s76) * (s74 + 81596225.0 / 93816426.0 * s76) + 21520059541.0 / 1377684.0 * s75 * s75 + 15510384942580921.0 / 27582029244.0 * s76 * s76;
	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau = (sycl::fabs(s7 - s1) + sycl::fabs(s7 - s2) + sycl::fabs(s7 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_7 = 0.85;
	a1 = coef_weights_1 * (1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2 = coef_weights_2 * (1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3 = coef_weights_3 * (1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a7 = coef_weights_7 * (1.0 + (tau * tau) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a7);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;
	w7 = a7 * tw1;

	// Compute coefficients of the Legendre basis polynomial
	u0 = v4;
	u1 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1 * s11 + w2 * s21 + w3 * s31;
	u2 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1 * s12 + w2 * s22 + w3 * s32;
	u3 = (w7 / coef_weights_7) * s73;
	u4 = (w7 / coef_weights_7) * s74;
	u5 = (w7 / coef_weights_7) * s75;
	u6 = (w7 / coef_weights_7) * s76;

	// Return value of reconstructed polynomial
	return u0 + u1 * 1.0 / 2.0 + u2 * 1.0 / 6.0 + u3 * 1.0 / 20.0 + u4 * 1.0 / 70.0 + u5 * 1.0 / 252.0 + u6 * 1.0 / 924.0;
}

inline real_t WENOAO73_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6, v7;
	real_t u0, u1, u2, u3, u4, u5, u6;
	real_t a1, a2, a3, a7;
	real_t w1, w2, w3, w7;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 3);
	v2 = *(f + k + 2);
	v3 = *(f + k + 1);
	v4 = *(f + k);
	v5 = *(f + k - 1);
	v6 = *(f + k - 2);
	v7 = *(f + k - 3);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v2 - 2.0 * v3 + 3.0 / 2.0 * v4, s12 = 1.0 / 2.0 * v2 - 1.0 * v3 + 1.0 / 2.0 * v4;
	real_t s21 = -1.0 / 2.0 * v3 + 1.0 / 2.0 * v5, s22 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s31 = -3.0 / 2.0 * v4 + 2.0 * v5 - 1.0 / 2.0 * v6, s32 = 1.0 / 2.0 * v4 - v5 + 1.0 / 2.0 * v6;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s71 = -191.0 / 10080.0 * v1 + 1688.0 / 10080.0 * v2 - 7843.0 / 10080.0 * v3 + 7843.0 / 10080.0 * v5 - 1688.0 / 10080.0 * v6 + 191.0 / 10080.0 * v7;
	real_t s72 = 79.0 / 10080.0 * v1 - 1014.0 / 10080.0 * v2 + 8385.0 / 10080.0 * v3 - 14900.0 / 10080.0 * v4 + 8385.0 / 10080.0 * v5 - 1014.0 / 10080.0 * v6 + 79.0 / 10080.0 * v7;
	real_t s73 = 5.0 / 216.0 * v1 - 38.0 / 216.0 * v2 + 61.0 / 216.0 * v3 - 61.0 / 216.0 * v5 + 38.0 / 216.0 * v6 - 5.0 / 216.0 * v7;
	real_t s74 = -13.0 / 1584.0 * v1 + 144.0 / 1584.0 * v2 + 459.0 / 1584.0 * v3 + 656.0 / 1584.0 * v4 - 459.0 / 1584.0 * v5 + 144.0 / 1584.0 * v6 - 13.0 / 1584.0 * v7;
	real_t s75 = -1.0 / 240.0 * v1 + 4.0 / 240.0 * v2 - 5.0 / 240.0 * v3 + 5.0 / 240.0 * v5 - 4.0 / 240.0 * v6 + 1 / 240.0 * v7;
	real_t s76 = 1.0 / 720.0 * v1 - 6.0 / 720.0 * v2 + 15.0 / 720.0 * v3 - 20.0 / 720.0 * v4 + 15.0 / 720.0 * v5 - 6.0 / 720.0 * v6 + 1 / 720.0 * v7;
	real_t s7 = 1.0 * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) + 13.0 / 3.0 * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) + 781.0 / 20.0 * (s73 + 26045.0 / 49203.0 * s75) * (s73 + 26045.0 / 49203.0 * s75) + 1421461.0 / 2275.0 * (s74 + 81596225.0 / 93816426.0 * s76) * (s74 + 81596225.0 / 93816426.0 * s76) + 21520059541.0 / 1377684.0 * s75 * s75 + 15510384942580921.0 / 27582029244.0 * s76 * s76;
	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau = (sycl::fabs(s7 - s1) + sycl::fabs(s7 - s2) + sycl::fabs(s7 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_7 = 0.85;
	a1 = coef_weights_1 * (1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2 = coef_weights_2 * (1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3 = coef_weights_3 * (1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a7 = coef_weights_7 * (1.0 + (tau * tau) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a7);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;
	w7 = a7 * tw1;

	// Compute coefficients of the Legendre basis polynomial
	u0 = v4;
	u1 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1 * s11 + w2 * s21 + w3 * s31;
	u2 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1 * s12 + w2 * s22 + w3 * s32;
	u3 = (w7 / coef_weights_7) * s73;
	u4 = (w7 / coef_weights_7) * s74;
	u5 = (w7 / coef_weights_7) * s75;
	u6 = (w7 / coef_weights_7) * s76;

	// Return value of reconstructed polynomial
	return u0 + u1 * 1.0 / 2.0 + u2 * 1.0 / 6.0 + u3 * 1.0 / 20.0 + u4 * 1.0 / 70.0 + u5 * 1.0 / 252.0 + u6 * 1.0 / 924.0;
}

/**
 * @brief  WENO-AO(7,5,3) scheme from Balsara (2016)
 */
inline real_t WENOAO753_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6, v7;
	real_t u0_5, u1_5, u2_5, u3_5, u4_5;
	real_t u0_7, u1_7, u2_7, u3_7, u4_7, u5_7, u6_7;
	real_t a1_5, a2_5, a3_5, a1_7, a2_7, a3_7, a5, a7;
	real_t w1_5, w2_5, w3_5, w1_7, w2_7, w3_7, w5, w7;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 3);
	v2 = *(f + k - 2);
	v3 = *(f + k - 1);
	v4 = *(f + k);
	v5 = *(f + k + 1);
	v6 = *(f + k + 2);
	v7 = *(f + k + 3);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v2 - 2.0 * v3 + 3.0 / 2.0 * v4, s12 = 1.0 / 2.0 * v2 - 1.0 * v3 + 1.0 / 2.0 * v4;
	real_t s21 = -1.0 / 2.0 * v3 + 1.0 / 2.0 * v5, s22 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s31 = -3.0 / 2.0 * v4 + 2.0 * v5 - 1.0 / 2.0 * v6, s32 = 1.0 / 2.0 * v4 - v5 + 1.0 / 2.0 * v6;
	real_t s51 = 11.0 / 120.0 * v2 - 82.0 / 120.0 * v3 + 82.0 / 120.0 * v5 - 11.0 / 120.0 * v6;
	real_t s52 = -3.0 / 56.0 * v2 + 40.0 / 56.0 * v3 + -74.0 / 56.0 * v4 + 40.0 / 56.0 * v5 - 3.0 / 56.0 * v6;
	real_t s53 = -1.0 / 12.0 * v2 + 2.0 / 12.0 * v3 - 2.0 / 12.0 * v5 + 1.0 / 12.0 * v6;
	real_t s54 = 1.0 / 24.0 * v2 - 4.0 / 24.0 * v3 + 6.0 / 24.0 * v4 - 4.0 / 24.0 * v5 + 1.0 / 24.0 * v6;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s5 = 1.0 * (s51 + 1.0 / 10.0 * s53) * (s51 + 1.0 / 10.0 * s53) + 13.0 / 3.0 * (s52 + 123.0 / 455.0 * s54) * (s52 + 123.0 / 455.0 * s54) + 781.0 / 20.0 * s53 * s53 + 1421461.0 / 2275.0 * s54 * s54;

	real_t s71 = -191.0 / 10080.0 * v1 + 1688.0 / 10080.0 * v2 - 7843.0 / 10080.0 * v3 + 7843.0 / 10080.0 * v5 - 1688.0 / 10080.0 * v6 + 191.0 / 10080.0 * v7;
	real_t s72 = 79.0 / 10080.0 * v1 - 1014.0 / 10080.0 * v2 + 8385.0 / 10080.0 * v3 - 14900.0 / 10080.0 * v4 + 8385.0 / 10080.0 * v5 - 1014.0 / 10080.0 * v6 + 79.0 / 10080.0 * v7;
	real_t s73 = 5.0 / 216.0 * v1 - 38.0 / 216.0 * v2 + 61.0 / 216.0 * v3 - 61.0 / 216.0 * v5 + 38.0 / 216.0 * v6 - 5.0 / 216.0 * v7;
	real_t s74 = -13.0 / 1584.0 * v1 + 144.0 / 1584.0 * v2 + 459.0 / 1584.0 * v3 + 656.0 / 1584.0 * v4 - 459.0 / 1584.0 * v5 + 144.0 / 1584.0 * v6 - 13.0 / 1584.0 * v7;
	real_t s75 = -1.0 / 240.0 * v1 + 4.0 / 240.0 * v2 - 5.0 / 240.0 * v3 + 5.0 / 240.0 * v5 - 4.0 / 240.0 * v6 + 1 / 240.0 * v7;
	real_t s76 = 1.0 / 720.0 * v1 - 6.0 / 720.0 * v2 + 15.0 / 720.0 * v3 - 20.0 / 720.0 * v4 + 15.0 / 720.0 * v5 - 6.0 / 720.0 * v6 + 1 / 720.0 * v7;
	real_t s7 = 1.0 * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) + 13.0 / 3.0 * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) + 781.0 / 20.0 * (s73 + 26045.0 / 49203.0 * s75) * (s73 + 26045.0 / 49203.0 * s75) + 1421461.0 / 2275.0 * (s74 + 81596225.0 / 93816426.0 * s76) * (s74 + 81596225.0 / 93816426.0 * s76) + 21520059541.0 / 1377684.0 * s75 * s75 + 15510384942580921.0 / 27582029244.0 * s76 * s76;

	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau_7 = (sycl::fabs(s7 - s1) + sycl::fabs(s7 - s2) + sycl::fabs(s7 - s3)) / 3.0;
	real_t tau_5 = (sycl::fabs(s5 - s1) + sycl::fabs(s5 - s2) + sycl::fabs(s5 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_5 = 0.85;
	real_t coef_weights_7 = 0.85;

	a1_7 = coef_weights_1 * (1.0 + (tau_7 * tau_7) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2_7 = coef_weights_2 * (1.0 + (tau_7 * tau_7) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3_7 = coef_weights_3 * (1.0 + (tau_7 * tau_7) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a7 = coef_weights_7 * (1.0 + (tau_7 * tau_7) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
	a1_5 = coef_weights_1 * (1.0 + (tau_5 * tau_5) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2_5 = coef_weights_2 * (1.0 + (tau_5 * tau_5) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3_5 = coef_weights_3 * (1.0 + (tau_5 * tau_5) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a5 = coef_weights_5 * (1.0 + (tau_5 * tau_5) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));

	real_t one_a_sum_7 = 1.0 / (a1_7 + a2_7 + a3_7 + a7);
	real_t one_a_sum_5 = 1.0 / (a1_5 + a2_5 + a3_5 + a5);

	w1_7 = a1_7 * one_a_sum_7;
	w2_7 = a2_7 * one_a_sum_7;
	w3_7 = a3_7 * one_a_sum_7;
	w7 = a7 * one_a_sum_7;

	w1_5 = a1_5 * one_a_sum_5;
	w2_5 = a2_5 * one_a_sum_5;
	w3_5 = a3_5 * one_a_sum_5;
	w5 = a5 * one_a_sum_5;

	// Compute coefficients of the Legendre basis polynomial of order 7
	u0_7 = v4;
	u1_7 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1_7 * s11 + w2_7 * s21 + w3_7 * s31;
	u2_7 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1_7 * s12 + w2_7 * s22 + w3_7 * s32;
	u3_7 = (w7 / coef_weights_7) * s73;
	u4_7 = (w7 / coef_weights_7) * s74;
	u5_7 = (w7 / coef_weights_7) * s75;
	u6_7 = (w7 / coef_weights_7) * s76;

	// Compute coefficients of the Legendre basis polynomial of order 5
	u0_5 = v4;
	u1_5 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1_5 * s11 + w2_5 * s21 + w3_5 * s31;
	u2_5 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1_5 * s12 + w2_5 * s22 + w3_5 * s32;
	u3_5 = (w5 / coef_weights_5) * s53;
	u4_5 = (w5 / coef_weights_5) * s54;

	// Compute values of reconstructed Legendre basis polynomials
	real_t polynomial_7 = u0_7 + u1_7 * 1.0 / 2.0 + u2_7 * 1.0 / 6.0 + u3_7 * 1.0 / 20.0 + u4_7 * 1.0 / 70.0 + u5_7 * 1.0 / 252.0 + u6_7 * 1.0 / 924.0;
	real_t polynomial_5 = u0_5 + u1_5 * 1.0 / 2.0 + u2_5 * 1.0 / 6.0 + u3_5 * 1.0 / 20.0 + u4_5 * 1.0 / 70.0;

	// Compute normalized weights for hybridization Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t sigma = sycl::fabs(s7 - s5);
	real_t b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
	real_t b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

	real_t one_b_sum = b7 + b5;

	real_t w_ao_7 = b7 / one_b_sum;
	real_t w_ao_5 = b5 / one_b_sum;

	// Return value of hybridized reconstructed polynomial
	return (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}

inline real_t WENOAO753_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6, v7;
	real_t u0_5, u1_5, u2_5, u3_5, u4_5;
	real_t u0_7, u1_7, u2_7, u3_7, u4_7, u5_7, u6_7;
	real_t a1_5, a2_5, a3_5, a1_7, a2_7, a3_7, a5, a7;
	real_t w1_5, w2_5, w3_5, w1_7, w2_7, w3_7, w5, w7;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 3);
	v2 = *(f + k + 2);
	v3 = *(f + k + 1);
	v4 = *(f + k);
	v5 = *(f + k - 1);
	v6 = *(f + k - 2);
	v7 = *(f + k - 3);

	// smoothness indicator
	real_t s11 = 1.0 / 2.0 * v2 - 2.0 * v3 + 3.0 / 2.0 * v4, s12 = 1.0 / 2.0 * v2 - 1.0 * v3 + 1.0 / 2.0 * v4;
	real_t s21 = -1.0 / 2.0 * v3 + 1.0 / 2.0 * v5, s22 = 1.0 / 2.0 * v3 - v4 + 1.0 / 2.0 * v5;
	real_t s31 = -3.0 / 2.0 * v4 + 2.0 * v5 - 1.0 / 2.0 * v6, s32 = 1.0 / 2.0 * v4 - v5 + 1.0 / 2.0 * v6;
	real_t s51 = 11.0 / 120.0 * v2 - 82.0 / 120.0 * v3 + 82.0 / 120.0 * v5 - 11.0 / 120.0 * v6;
	real_t s52 = -3.0 / 56.0 * v2 + 40.0 / 56.0 * v3 + -74.0 / 56.0 * v4 + 40.0 / 56.0 * v5 - 3.0 / 56.0 * v6;
	real_t s53 = -1.0 / 12.0 * v2 + 2.0 / 12.0 * v3 - 2.0 / 12.0 * v5 + 1.0 / 12.0 * v6;
	real_t s54 = 1.0 / 24.0 * v2 - 4.0 / 24.0 * v3 + 6.0 / 24.0 * v4 - 4.0 / 24.0 * v5 + 1.0 / 24.0 * v6;
	real_t s1 = 1.0 * s11 * s11 + 13.0 / 3.0 * s12 * s12;
	real_t s2 = 1.0 * s21 * s21 + 13.0 / 3.0 * s22 * s22;
	real_t s3 = 1.0 * s31 * s31 + 13.0 / 3.0 * s32 * s32;
	real_t s5 = 1.0 * (s51 + 1.0 / 10.0 * s53) * (s51 + 1.0 / 10.0 * s53) + 13.0 / 3.0 * (s52 + 123.0 / 455.0 * s54) * (s52 + 123.0 / 455.0 * s54) + 781.0 / 20.0 * s53 * s53 + 1421461.0 / 2275.0 * s54 * s54;

	real_t s71 = -191.0 / 10080.0 * v1 + 1688.0 / 10080.0 * v2 - 7843.0 / 10080.0 * v3 + 7843.0 / 10080.0 * v5 - 1688.0 / 10080.0 * v6 + 191.0 / 10080.0 * v7;
	real_t s72 = 79.0 / 10080.0 * v1 - 1014.0 / 10080.0 * v2 + 8385.0 / 10080.0 * v3 - 14900.0 / 10080.0 * v4 + 8385.0 / 10080.0 * v5 - 1014.0 / 10080.0 * v6 + 79.0 / 10080.0 * v7;
	real_t s73 = 5.0 / 216.0 * v1 - 38.0 / 216.0 * v2 + 61.0 / 216.0 * v3 - 61.0 / 216.0 * v5 + 38.0 / 216.0 * v6 - 5.0 / 216.0 * v7;
	real_t s74 = -13.0 / 1584.0 * v1 + 144.0 / 1584.0 * v2 + 459.0 / 1584.0 * v3 + 656.0 / 1584.0 * v4 - 459.0 / 1584.0 * v5 + 144.0 / 1584.0 * v6 - 13.0 / 1584.0 * v7;
	real_t s75 = -1.0 / 240.0 * v1 + 4.0 / 240.0 * v2 - 5.0 / 240.0 * v3 + 5.0 / 240.0 * v5 - 4.0 / 240.0 * v6 + 1 / 240.0 * v7;
	real_t s76 = 1.0 / 720.0 * v1 - 6.0 / 720.0 * v2 + 15.0 / 720.0 * v3 - 20.0 / 720.0 * v4 + 15.0 / 720.0 * v5 - 6.0 / 720.0 * v6 + 1 / 720.0 * v7;
	real_t s7 = 1.0 * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) * (s71 + 1.0 / 10.0 * s73 + 1.0 / 126.0 * s75) + 13.0 / 3.0 * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) * (s72 + 123.0 / 455.0 * s74 + 85.0 / 2002.0 * s76) + 781.0 / 20.0 * (s73 + 26045.0 / 49203.0 * s75) * (s73 + 26045.0 / 49203.0 * s75) + 1421461.0 / 2275.0 * (s74 + 81596225.0 / 93816426.0 * s76) * (s74 + 81596225.0 / 93816426.0 * s76) + 21520059541.0 / 1377684.0 * s75 * s75 + 15510384942580921.0 / 27582029244.0 * s76 * s76;

	// Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t tau_7 = (sycl::fabs(s7 - s1) + sycl::fabs(s7 - s2) + sycl::fabs(s7 - s3)) / 3.0;
	real_t tau_5 = (sycl::fabs(s5 - s1) + sycl::fabs(s5 - s2) + sycl::fabs(s5 - s3)) / 3.0;
	real_t coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_2 = (1 - 0.85) * 0.85;
	real_t coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
	real_t coef_weights_5 = 0.85;
	real_t coef_weights_7 = 0.85;

	a1_7 = coef_weights_1 * (1.0 + (tau_7 * tau_7) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2_7 = coef_weights_2 * (1.0 + (tau_7 * tau_7) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3_7 = coef_weights_3 * (1.0 + (tau_7 * tau_7) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a7 = coef_weights_7 * (1.0 + (tau_7 * tau_7) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
	a1_5 = coef_weights_1 * (1.0 + (tau_5 * tau_5) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
	a2_5 = coef_weights_2 * (1.0 + (tau_5 * tau_5) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
	a3_5 = coef_weights_3 * (1.0 + (tau_5 * tau_5) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
	a5 = coef_weights_5 * (1.0 + (tau_5 * tau_5) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));

	real_t one_a_sum_7 = 1.0 / (a1_7 + a2_7 + a3_7 + a7);
	real_t one_a_sum_5 = 1.0 / (a1_5 + a2_5 + a3_5 + a5);

	w1_7 = a1_7 * one_a_sum_7;
	w2_7 = a2_7 * one_a_sum_7;
	w3_7 = a3_7 * one_a_sum_7;
	w7 = a7 * one_a_sum_7;

	w1_5 = a1_5 * one_a_sum_5;
	w2_5 = a2_5 * one_a_sum_5;
	w3_5 = a3_5 * one_a_sum_5;
	w5 = a5 * one_a_sum_5;

	// Compute coefficients of the Legendre basis polynomial of order 7
	u0_7 = v4;
	u1_7 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1_7 * s11 + w2_7 * s21 + w3_7 * s31;
	u2_7 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1_7 * s12 + w2_7 * s22 + w3_7 * s32;
	u3_7 = (w7 / coef_weights_7) * s73;
	u4_7 = (w7 / coef_weights_7) * s74;
	u5_7 = (w7 / coef_weights_7) * s75;
	u6_7 = (w7 / coef_weights_7) * s76;

	// Compute coefficients of the Legendre basis polynomial of order 5
	u0_5 = v4;
	u1_5 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31) + w1_5 * s11 + w2_5 * s21 + w3_5 * s31;
	u2_5 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32) + w1_5 * s12 + w2_5 * s22 + w3_5 * s32;
	u3_5 = (w5 / coef_weights_5) * s53;
	u4_5 = (w5 / coef_weights_5) * s54;

	// Compute values of reconstructed Legendre basis polynomials
	real_t polynomial_7 = u0_7 + u1_7 * 1.0 / 2.0 + u2_7 * 1.0 / 6.0 + u3_7 * 1.0 / 20.0 + u4_7 * 1.0 / 70.0 + u5_7 * 1.0 / 252.0 + u6_7 * 1.0 / 924.0;
	real_t polynomial_5 = u0_5 + u1_5 * 1.0 / 2.0 + u2_5 * 1.0 / 6.0 + u3_5 * 1.0 / 20.0 + u4_5 * 1.0 / 70.0;

	// Compute normalized weights for hybridization Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
	real_t sigma = sycl::fabs(s7 - s5);
	real_t b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
	real_t b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

	real_t one_b_sum = b7 + b5;

	real_t w_ao_7 = b7 / one_b_sum;
	real_t w_ao_5 = b5 / one_b_sum;

	// Return value of hybridized reconstructed polynomial
	return (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}
