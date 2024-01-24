#pragma once

#include "Utils_schemes.hpp"

/**
 * @brief the 5th WENO Scheme
 *
 * @param f
 * @param delta
 * @return real_t
 */
SYCL_DEVICE inline real_t weno5old_BODY(const real_t v1, const real_t v2, const real_t v3, const real_t v4, const real_t v5)
{
	real_t a1, a2, a3;
	real_t dtwo = _DF(2.0), dtre = _DF(3.0);
	a1 = v1 - dtwo * v2 + v3;
	real_t s1 = _DF(13.0) * a1 * a1;
	a1 = v1 - _DF(4.0) * v2 + dtre * v3;
	s1 += dtre * a1 * a1;
	a1 = v2 - dtwo * v3 + v4;
	real_t s2 = _DF(13.0) * a1 * a1;
	a1 = v2 - v4;
	s2 += dtre * a1 * a1;
	a1 = v3 - dtwo * v4 + v5;
	real_t s3 = _DF(13.0) * a1 * a1;
	a1 = dtre * v3 - _DF(4.0) * v4 + v5;
	s3 += dtre * a1 * a1;

	real_t tol = _DF(1.0E-6);
	s1 += tol, s2 += tol, s3 += tol;

	// // TODO: float precison
	// real_t _s1 = CarmackR(s1), _s2 = CarmackR(s2), _s3 = CarmackR(s3);
	// int cs1 = sycl::log10(s1), cs2 = sycl::log10(s2), cs3 = sycl::log10(s3);
	// int c12 = cs1 - cs2, c13 = cs1 - cs3, c1 = -bit_maxmag(c12, c13);
	// int c21 = cs2 - cs1, c23 = cs2 - cs3, c2 = -bit_maxmag(c21, c23);
	// int c31 = cs3 - cs1, c32 = cs3 - cs2, c3 = -bit_maxmag(c31, c32);
	// real_t fc1 = pown(_DF(10.0), c1), fc2 = pown(_DF(10.0), c2), fc3 = pown(_DF(10.0), c3);

	// a1 = CarmackR(_DF(1.0) * fc1 * fc1 + _DF(2.0) * (s1 * fc1 * _s2) * (s1 * fc1 * _s2) + _DF(3.0) * (s1 * fc1 * _s3) * (s1 * fc1 * _s3)) * _DF(1.0) * fc1 * fc1;
	// a2 = CarmackR(_DF(1.0) * (s2 * fc2 * _s1) * (s2 * fc2 * _s1) + _DF(2.0) * fc2 * fc2 + _DF(3.0) * (s2 * fc2 * _s3) * (s2 * fc2 * _s3)) * _DF(2.0) * fc2 * fc2;
	// a3 = CarmackR(_DF(1.0) * (s3 * fc3 * _s1) * (s3 * fc3 * _s1) + _DF(2.0) * (s3 * fc3 * _s2) * (s3 * fc3 * _s2) + _DF(3.0) * fc3 * fc3) * _DF(3.0) * fc3 * fc3;

#ifdef USE_DOUBLE
	// // for double precision
	a1 = _DF(0.1) * s2 * s2 * s3 * s3;
	a2 = _DF(0.2) * s1 * s1 * s3 * s3;
	a3 = _DF(0.3) * s1 * s1 * s2 * s2;

	real_t tw1 = _DF(1.0) / (a1 + a2 + a3);
	a1 = a1 * tw1, a2 = a2 * tw1, a3 = a3 * tw1;

#else

	// // // for float precision
	double Da1 = 0.1 * s2 * s2 * s3 * s3;
	double Da2 = 0.2 * s1 * s1 * s3 * s3;
	double Da3 = 0.3 * s1 * s1 * s2 * s2;

	double tw1 = 1.0 / (Da1 + Da2 + Da3);
	a1 = Da1 * tw1, a2 = Da2 * tw1, a3 = Da3 * tw1;
#endif

	s1 = a1 * (dtwo * v1 - _DF(7.0) * v2 + _DF(11.0) * v3);
	s2 = a2 * (-v2 + _DF(5.0) * v3 + dtwo * v4);
	s3 = a3 * (dtwo * v3 + _DF(5.0) * v4 - v5);

	return (s1 + s2 + s3);
}

SYCL_DEVICE inline real_t weno5old_GPU(real_t *f, real_t *m)
{
	int k;
	real_t v1, v2, v3, v4, v5, temf, temm;
	// ff
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	temf = weno5old_BODY(v1, v2, v3, v4, v5); //* sum;
	// mm
	k = 1;
	v1 = *(m + k + 2);
	v2 = *(m + k + 1);
	v3 = *(m + k);
	v4 = *(m + k - 1);
	v5 = *(m + k - 2);

	temm = weno5old_BODY(v1, v2, v3, v4, v5); //* sum;

	return (temf + temm) * _six;
}

SYCL_DEVICE inline real_t weno5_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t s1, s2, s3;
	real_t a1, a2, a3, w1, w2, w3;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	// smoothness indicator
	s1 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	s2 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 0.25 * (v2 - v4) * (v2 - v4);
	s3 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);

	// weights
	a1 = 0.1 / (1.0e-6 + s1) / (1.0e-15 + s1);
	a2 = 0.6 / (1.0e-6 + s2) / (1.0e-15 + s2);
	a3 = 0.3 / (1.0e-6 + s3) / (1.0e-15 + s3);

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	// return weighted average
	return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}

SYCL_DEVICE inline real_t weno5_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t s1, s2, s3;
	real_t a1, a2, a3, w1, w2, w3;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);

	// smoothness indicator
	s1 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	s2 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 0.25 * (v2 - v4) * (v2 - v4);
	s3 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);

	// weights
	a1 = 0.1 / (1.0e-6 + s1) / (1.0e-15 + s1);
	a2 = 0.6 / (1.0e-6 + s2) / (1.0e-15 + s2);
	a3 = 0.3 / (1.0e-6 + s3) / (1.0e-15 + s3);

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	// return weighted average
	return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}

SYCL_DEVICE inline real_t Weno5L2_P(real_t *f, real_t delta, real_t lambda)
{
	// assign value to v1, v2,...
	int k = 0;
	real_t v1 = *(f + k - 2);
	real_t v2 = *(f + k - 1);
	real_t v3 = *(f + k);
	real_t v4 = *(f + k + 1);
	real_t v5 = *(f + k + 2);
	real_t v6 = *(f + k + 3);

	// smoothness indicator
	real_t epsilon = 1.0e-20;
	real_t s0 = std::pow((v4 - v3), 2.0);
	real_t s1 = std::pow((v3 - v2), 2.0);
	real_t s2 = (13.0 * std::pow(v3 - 2.0 * v4 + v5, 2.0) + 3.0 * std::pow(3.0 * v3 - 4.0 * v4 + v5, 2.0)) / 12.0;
	real_t s3 = (13.0 * std::pow(v1 - 2.0 * v2 + v3, 2.0) + 3.0 * std::pow(v1 - 4.0 * v2 + 3.0 * v3, 2.0)) / 12.0;
	real_t t5 = (13.0 * std::pow(v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1, 2.0) + 3.0 * std::pow(v5 - 2.0 * v4 + 2.0 * v2 - v1, 2.0)) / 12.0;

	real_t e0 = (v4 * v4 - 4.0 * v3 * v4 + 2.0 * v2 * v4 + 4.0 * v3 * v3 - 4.0 * v2 * v3 + v2 * v2) / 45.0;
	real_t e1 = e0;
	real_t e2 = 0.0;
	real_t e3 = 0.0;

	real_t a0 = 0.4 * (1.0 + lambda * t5 / (lambda * s0 + e0 + epsilon));
	real_t a1 = 0.2 * (1.0 + lambda * t5 / (lambda * s1 + e1 + epsilon));
	real_t a2 = 0.3 * (1.0 + lambda * t5 / (lambda * s2 + e2 + epsilon));
	real_t a3 = 0.1 * (1.0 + lambda * t5 / (lambda * s3 + e3 + epsilon));

	// real_t a0 = 0.4*(1.0 + t5/(s0 + epsilon));
	// real_t a1 = 0.2*(1.0 + t5/(s1 + epsilon));
	// real_t a2 = 0.3*(1.0 + t5/(s2 + epsilon));
	// real_t a3 = 0.1*(1.0 + t5/(s3 + epsilon));

	real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);
	real_t w0 = a0 * tw1;
	real_t w1 = a1 * tw1;
	real_t w2 = a2 * tw1;
	real_t w3 = a3 * tw1;

	// return weighted average
	return (w3 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w1 * (-3.0 * v2 + 9.0 * v3) + w0 * (3.0 * v3 + 3.0 * v4)) / 6.0;
}

SYCL_DEVICE inline real_t Weno5L2_M(real_t *f, real_t delta, real_t lambda)
{
	// assign value to v1, v2,...
	int k = 1;
	real_t v1 = *(f + k + 2);
	real_t v2 = *(f + k + 1);
	real_t v3 = *(f + k);
	real_t v4 = *(f + k - 1);
	real_t v5 = *(f + k - 2);
	real_t v6 = *(f + k - 3);

	// smoothness indicator
	real_t epsilon = 1.0e-20;
	real_t s0 = std::pow((v4 - v3), 2.0);
	real_t s1 = std::pow((v3 - v2), 2.0);
	real_t s2 = (13.0 * std::pow(v3 - 2.0 * v4 + v5, 2.0) + 3.0 * std::pow(3.0 * v3 - 4.0 * v4 + v5, 2.0)) / 12.0;
	real_t s3 = (13.0 * std::pow(v1 - 2.0 * v2 + v3, 2.0) + 3.0 * std::pow(v1 - 4.0 * v2 + 3.0 * v3, 2.0)) / 12.0;
	real_t t5 = (13.0 * std::pow(v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1, 2.0) + 3.0 * std::pow(v5 - 2.0 * v4 + 2.0 * v2 - v1, 2.0)) / 12.0;

	real_t e0 = (v4 * v4 - 4.0 * v3 * v4 + 2.0 * v2 * v4 + 4.0 * v3 * v3 - 4.0 * v2 * v3 + v2 * v2) / 45.0;
	real_t e1 = e0;
	real_t e2 = 0.0;
	real_t e3 = 0.0;

	real_t a0 = 0.4 * (1.0 + lambda * t5 / (lambda * s0 + e0 + epsilon));
	real_t a1 = 0.2 * (1.0 + lambda * t5 / (lambda * s1 + e1 + epsilon));
	real_t a2 = 0.3 * (1.0 + lambda * t5 / (lambda * s2 + e2 + epsilon));
	real_t a3 = 0.1 * (1.0 + lambda * t5 / (lambda * s3 + e3 + epsilon));

	real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);
	real_t w0 = a0 * tw1;
	real_t w1 = a1 * tw1;
	real_t w2 = a2 * tw1;
	real_t w3 = a3 * tw1;

	// return weighted average
	return (w3 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w1 * (-3.0 * v2 + 9.0 * v3) + w0 * (3.0 * v3 + 3.0 * v4)) / 6.0;
}

SYCL_DEVICE inline real_t weno5Z_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6;
	real_t s1, s2, s3;
	real_t a1, a2, a3, w1, w2, w3;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);
	v6 = *(f + k + 3);

	// smoothness indicator
	real_t epsilon = 1.0e-6;
	s1 = 13.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	s2 = 13.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 * (v2 - v4) * (v2 - v4);
	s3 = 13.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);

	// weights
	real_t s5 = sycl::fabs(s1 - s3);
	a1 = 0.1 * (1.0 + s5 / (s1 + epsilon));
	a2 = 0.6 * (1.0 + s5 / (s2 + epsilon));
	a3 = 0.3 * (1.0 + s5 / (s3 + epsilon));

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	// return weighted average
	return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}

SYCL_DEVICE inline real_t weno5Z_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6;
	real_t s1, s2, s3;
	real_t a1, a2, a3, w1, w2, w3;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);
	v6 = *(f + k - 3);

	// smoothness indicator
	real_t epsilon = 1.0e-6;
	s1 = 13.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	s2 = 13.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 * (v2 - v4) * (v2 - v4);
	s3 = 13.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);

	// weights
	real_t s5 = sycl::fabs(s1 - s3);
	a1 = 0.1 * (1.0 + s5 / (s1 + epsilon));
	a2 = 0.6 * (1.0 + s5 / (s2 + epsilon));
	a3 = 0.3 * (1.0 + s5 / (s3 + epsilon));

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	// return weighted average
	return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}
