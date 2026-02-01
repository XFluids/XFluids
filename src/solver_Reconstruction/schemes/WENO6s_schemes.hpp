#pragma once

#include "Utils_schemes.hpp"

SYCL_DEVICE inline real_t WENOCU6_BODYGPU(const real_t v1, const real_t v2, const real_t v3, const real_t v4, const real_t v5, const real_t v6, const real_t epsilon)
{ // smoothness indicator // real_t epsilon = 1.e-8 * delta * delta;
	real_t s11 = v1 - _DF(2.0) * v2 + v3;
	real_t s12 = v1 - _DF(4.0) * v2 + _DF(3.0) * v3;
	real_t s1 = _DF(13.0) * s11 * s11 + _DF(3.0) * s12 * s12;
	real_t s21 = v2 - _DF(2.0) * v3 + v4;
	real_t s22 = v2 - v4;
	real_t s2 = _DF(13.0) * s21 * s21 + _DF(3.0) * s22 * s22;
	real_t s31 = v3 - _DF(2.0) * v4 + v5;
	real_t s32 = _DF(3.0) * v3 - _DF(4.0) * v4 + v5;
	real_t s3 = _DF(13.0) * s31 * s31 + _DF(3.0) * s32 * s32;
	real_t tau61 = (_DF(259.0) * v6 - _DF(1895.0) * v5 + _DF(6670.0) * v4 - _DF(2590.0) * v3 - _DF(2785.0) * v2 + _DF(341.0) * v1) * _ftss;
	real_t tau62 = -(v5 - _DF(12.0) * v4 + _DF(22.0) * v3 - _DF(12.0) * v2 + v1) * _sxtn;
	real_t tau63 = -(_DF(7.0) * v6 - _DF(47.0) * v5 + _DF(94.0) * v4 - _DF(70.0) * v3 + _DF(11.0) * v2 + _DF(5.0) * v1) * _ohff;
	real_t tau64 = (v5 - _DF(4.0) * v4 + _DF(6.0) * v3 - _DF(4.0) * v2 + v1) * _twfr;
	real_t tau65 = -(-v6 + _DF(5.0) * v5 - _DF(10.0) * v4 + _DF(10.0) * v3 - _DF(5.0) * v2 + v1) * _ohtz;
	real_t a1a1 = _DF(1.0), a2a2 = wu6a2a2, a1a3 = _DF(0.5), a3a3 = wu6a3a3, a2a4 = _DF(4.2); // real_t a1a1 = 1.0, a2a2 = 13.0 / 3.0, a1a3 = 0.5, a3a3 = 3129.0 / 80.0, a2a4 = 21.0 / 5.0;
	real_t a1a5 = _DF(0.125), a4a4 = wu6a4a4, a3a5 = wu6a3a5, a5a5 = wu6a5a5;				  // a1a5 = 1.0 / 8.0, a4a4 = 87617.0 / 140.0, a3a5 = 14127.0 / 224.0, a5a5 = 252337135.0 / 16128.0;
	real_t s6 = (tau61 * tau61 * a1a1 + tau62 * tau62 * a2a2 + tau61 * tau63 * a1a3 + tau63 * tau63 * a3a3 + tau62 * tau64 * a2a4 + tau61 * tau65 * a1a5 + tau64 * tau64 * a4a4 + tau63 * tau65 * a3a5 + tau65 * tau65 * a5a5) * _DF(12.0);

	// weights
	real_t s55 = (s1 + s3 + _DF(4.0) * s2) * _six;
	real_t s5 = sycl::fabs(s6 - s55);
	real_t r0 = _DF(20.0);
	real_t r1 = r0 + s5 / (s1 + epsilon);
	real_t r2 = r0 + s5 / (s2 + epsilon);
	real_t r3 = r0 + s5 / (s3 + epsilon);
	real_t r4 = r0 + s5 / (s6 + epsilon);
	real_t a1 = _DF(0.05) * r1;
	real_t a2 = _DF(0.45) * r2;
	real_t a3 = _DF(0.45) * r3;
	real_t a4 = _DF(0.05) * r4;
	real_t tw1 = _DF(1.0) / (a1 + a2 + a3 + a4);
	real_t w1 = a1 * tw1;
	real_t w2 = a2 * tw1;
	real_t w3 = a3 * tw1;
	real_t w4 = a4 * tw1;

	// return weighted average
	real_t temp = _DF(0.0);
	temp += w1 * (_DF(2.0) * v1 - _DF(7.0) * v2 + _DF(11.0) * v3);
	temp += w2 * (-v2 + _DF(5.0) * v3 + _DF(2.0) * v4);
	temp += w3 * (_DF(2.0) * v3 + _DF(5.0) * v4 - v5);
	temp += w4 * (_DF(11.0) * v4 - _DF(7.0) * v5 + _DF(2.0) * v6);
	return temp;
}

// this is WENOCU6
SYCL_DEVICE inline real_t WENOCU6_GPU(real_t *f, real_t *m, real_t delta)
{
	// assign value to v1, v2,...
	int k = 0;
	real_t v1, v2, v3, v4, v5, v6, temf, temm, epsilon = _DF(1.e-8) * delta * delta;
	// f
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);
	v6 = *(f + k + 3);
	temf = WENOCU6_BODYGPU(v1, v2, v3, v4, v5, v6, epsilon);

	// m
	k = 1;
	v1 = *(m + k + 2);
	v2 = *(m + k + 1);
	v3 = *(m + k);
	v4 = *(m + k - 1);
	v5 = *(m + k - 2);
	v6 = *(m + k - 3);
	temm = WENOCU6_BODYGPU(v1, v2, v3, v4, v5, v6, epsilon);

	return (temf + temm) * _six;
}

// SYCL_DEVICE inline real_t WENOCU6_P(real_t *f, real_t delta)
// {
//     // assign value to v1, v2,...
//     int k = 0;
//     real_t v1 = *(f + k - 2);
//     real_t v2 = *(f + k - 1);
//     real_t v3 = *(f + k);
//     real_t v4 = *(f + k + 1);
//     real_t v5 = *(f + k + 2);
//     real_t v6 = *(f + k + 3);

//     // smoothness indicator
//     real_t epsilon = 1.e-8 * delta * delta;
//     real_t s11 = v1 - 2.0 * v2 + v3;
//     real_t s12 = v1 - 4.0 * v2 + 3.0 * v3;
//     real_t s1 = 13.0 * s11 * s11 + 3.0 * s12 * s12;
//     real_t s21 = v2 - 2.0 * v3 + v4;
//     real_t s22 = v2 - v4;
//     real_t s2 = 13.0 * s21 * s21 + 3.0 * s22 * s22;
//     real_t s31 = v3 - 2.0 * v4 + v5;
//     real_t s32 = 3.0 * v3 - 4.0 * v4 + v5;
//     real_t s3 = 13.0 * s31 * s31 + 3.0 * s32 * s32;
//     real_t tau61 = (259.0 * v6 - 1895.0 * v5 + 6670.0 * v4 - 2590.0 * v3 - 2785.0 * v2 + 341.0 * v1) / 5760.0;
//     real_t tau62 = -(v5 - 12.0 * v4 + 22.0 * v3 - 12.0 * v2 + v1) / 16.0;
//     real_t tau63 = -(7.0 * v6 - 47.0 * v5 + 94.0 * v4 - 70.0 * v3 + 11.0 * v2 + 5.0 * v1) / 144.0;
//     real_t tau64 = (v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1) / 24.0;
//     real_t tau65 = -(-v6 + 5.0 * v5 - 10.0 * v4 + 10.0 * v3 - 5.0 * v2 + v1) / 120.0;
//     real_t a1a1 = 1.0, a2a2 = 13.0 / 3.0, a1a3 = 0.5, a3a3 = 3129.0 / 80.0, a2a4 = 21.0 / 5.0;
//     real_t a1a5 = 1.0 / 8.0, a4a4 = 87617.0 / 140.0, a3a5 = 14127.0 / 224.0, a5a5 = 252337135.0 / 16128.0;
//     real_t s6 = (tau61 * tau61 * a1a1 + tau62 * tau62 * a2a2 + tau61 * tau63 * a1a3 + tau63 * tau63 * a3a3 + tau62 * tau64 * a2a4 + tau61 * tau65 * a1a5 + tau64 * tau64 * a4a4 + tau63 * tau65 * a3a5 + tau65 * tau65 * a5a5) * 12.0;

//     // weights
//     real_t s55 = (s1 + s3 + 4.0 * s2) / 6.0;
//     real_t s5 = sycl::fabs(s6 - s55);
//     real_t r0 = 20.0;
//     real_t r1 = r0 + s5 / (s1 + epsilon);
//     real_t r2 = r0 + s5 / (s2 + epsilon);
//     real_t r3 = r0 + s5 / (s3 + epsilon);
//     real_t r4 = r0 + s5 / (s6 + epsilon);
//     real_t a1 = 0.05 * r1;
//     real_t a2 = 0.45 * r2;
//     real_t a3 = 0.45 * r3;
//     real_t a4 = 0.05 * r4;
//     real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
//     real_t w1 = a1 * tw1;
//     real_t w2 = a2 * tw1;
//     real_t w3 = a3 * tw1;
//     real_t w4 = a4 * tw1;

//     // return weighted average
//     return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
// }

// // this is WENOCU6
// SYCL_DEVICE inline real_t WENOCU6_M(real_t *f, real_t delta)
// {
//     // assign value to v1, v2,...
//     int k = 1;
//     real_t v1 = *(f + k + 2);
//     real_t v2 = *(f + k + 1);
//     real_t v3 = *(f + k);
//     real_t v4 = *(f + k - 1);
//     real_t v5 = *(f + k - 2);
//     real_t v6 = *(f + k - 3);

//     real_t epsilon = 1.e-8 * delta * delta;
//     real_t s11 = v1 - 2.0 * v2 + v3;
//     real_t s12 = v1 - 4.0 * v2 + 3.0 * v3;
//     real_t s1 = 13.0 * s11 * s11 + 3.0 * s12 * s12;
//     real_t s21 = v2 - 2.0 * v3 + v4;
//     real_t s22 = v2 - v4;
//     real_t s2 = 13.0 * s21 * s21 + 3.0 * s22 * s22;
//     real_t s31 = v3 - 2.0 * v4 + v5;
//     real_t s32 = 3.0 * v3 - 4.0 * v4 + v5;
//     real_t s3 = 13.0 * s31 * s31 + 3.0 * s32 * s32;
//     real_t tau61 = (259.0 * v6 - 1895.0 * v5 + 6670.0 * v4 - 2590.0 * v3 - 2785.0 * v2 + 341.0 * v1) / 5760.0;
//     real_t tau62 = -(v5 - 12.0 * v4 + 22.0 * v3 - 12.0 * v2 + v1) / 16.0;
//     real_t tau63 = -(7.0 * v6 - 47.0 * v5 + 94.0 * v4 - 70.0 * v3 + 11.0 * v2 + 5.0 * v1) / 144.0;
//     real_t tau64 = (v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1) / 24.0;
//     real_t tau65 = -(-v6 + 5.0 * v5 - 10.0 * v4 + 10.0 * v3 - 5.0 * v2 + v1) / 120.0;
//     real_t a1a1 = 1.0, a2a2 = 13.0 / 3.0, a1a3 = 0.5, a3a3 = 3129.0 / 80.0, a2a4 = 21.0 / 5.0;
//     real_t a1a5 = 1.0 / 8.0, a4a4 = 87617.0 / 140.0, a3a5 = 14127.0 / 224.0, a5a5 = 252337135.0 / 16128.0;
//     real_t s6 = (tau61 * tau61 * a1a1 + tau62 * tau62 * a2a2 + tau61 * tau63 * a1a3 + tau63 * tau63 * a3a3 + tau62 * tau64 * a2a4 + tau61 * tau65 * a1a5 + tau64 * tau64 * a4a4 + tau63 * tau65 * a3a5 + tau65 * tau65 * a5a5) * 12.0;

//     // weights
//     real_t s55 = (s1 + s3 + 4.0 * s2) / 6.0;
//     real_t s5 = sycl::fabs(s6 - s55);
//     real_t r0 = 20.0;
//     real_t r1 = r0 + s5 / (s1 + epsilon);
//     real_t r2 = r0 + s5 / (s2 + epsilon);
//     real_t r3 = r0 + s5 / (s3 + epsilon);
//     real_t r4 = r0 + s5 / (s6 + epsilon);
//     real_t a1 = 0.05 * r1;
//     real_t a2 = 0.45 * r2;
//     real_t a3 = 0.45 * r3;
//     real_t a4 = 0.05 * r4;
//     real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
//     real_t w1 = a1 * tw1;
//     real_t w2 = a2 * tw1;
//     real_t w3 = a3 * tw1;
//     real_t w4 = a4 * tw1;

//     // return weighted average
//     return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
// }

SYCL_DEVICE inline real_t WENOCU6M1_P(real_t *f, real_t delta)
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
	real_t epsilon = 1.e-8 * delta * delta;
	real_t s11 = v1 - 2.0 * v2 + v3;
	real_t s12 = v1 - 4.0 * v2 + 3.0 * v3;
	real_t s1 = 13.0 * s11 * s11 + 3.0 * s12 * s12;
	real_t s21 = v2 - 2.0 * v3 + v4;
	real_t s22 = v2 - v4;
	real_t s2 = 13.0 * s21 * s21 + 3.0 * s22 * s22;
	real_t s31 = v3 - 2.0 * v4 + v5;
	real_t s32 = 3.0 * v3 - 4.0 * v4 + v5;
	real_t s3 = 13.0 * s31 * s31 + 3.0 * s32 * s32;
	real_t tau61 = (259.0 * v6 - 1895.0 * v5 + 6670.0 * v4 - 2590.0 * v3 - 2785.0 * v2 + 341.0 * v1) / 5760.0;
	real_t tau62 = -(v5 - 12.0 * v4 + 22.0 * v3 - 12.0 * v2 + v1) / 16.0;
	real_t tau63 = -(7.0 * v6 - 47.0 * v5 + 94.0 * v4 - 70.0 * v3 + 11.0 * v2 + 5.0 * v1) / 144.0;
	real_t tau64 = (v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1) / 24.0;
	real_t tau65 = -(-v6 + 5.0 * v5 - 10.0 * v4 + 10.0 * v3 - 5.0 * v2 + v1) / 120.0;
	real_t a1a1 = 1.0, a2a2 = 13.0 / 3.0, a1a3 = 0.5, a3a3 = 3129.0 / 80.0, a2a4 = 21.0 / 5.0;
	real_t a1a5 = 1.0 / 8.0, a4a4 = 87617.0 / 140.0, a3a5 = 14127.0 / 224.0, a5a5 = 252337135.0 / 16128.0;
	real_t s6 = (tau61 * tau61 * a1a1 + tau62 * tau62 * a2a2 + tau61 * tau63 * a1a3 + tau63 * tau63 * a3a3 + tau62 * tau64 * a2a4 + tau61 * tau65 * a1a5 + tau64 * tau64 * a4a4 + tau63 * tau65 * a3a5 + tau65 * tau65 * a5a5) * 12.0;

	// weights
	real_t s55 = (s1 + s3 + 4.0 * s2) / 6.0;
	real_t s5 = sycl::fabs(s6 - s55);
	real_t r0 = 1.0e3;
	real_t r1 = r0 + s5 / (s1 + epsilon);
	real_t r2 = r0 + s5 / (s2 + epsilon);
	real_t r3 = r0 + s5 / (s3 + epsilon);
	real_t r4 = r0 + s5 / (s6 + epsilon);
	real_t a1 = 0.05 * r1 * r1 * r1 * r1;
	real_t a2 = 0.45 * r2 * r2 * r2 * r2;
	real_t a3 = 0.45 * r3 * r3 * r3 * r3;
	real_t a4 = 0.05 * r4 * r4 * r4 * r4;
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
	real_t w1 = a1 * tw1;
	real_t w2 = a2 * tw1;
	real_t w3 = a3 * tw1;
	real_t w4 = a4 * tw1;

	// return weighted average
	return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}

// this is WENOCU6_M
SYCL_DEVICE inline real_t WENOCU6M1_M(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 1;
	real_t v1 = *(f + k + 2);
	real_t v2 = *(f + k + 1);
	real_t v3 = *(f + k);
	real_t v4 = *(f + k - 1);
	real_t v5 = *(f + k - 2);
	real_t v6 = *(f + k - 3);

	real_t epsilon = 1.e-8 * delta * delta;
	real_t s11 = v1 - 2.0 * v2 + v3;
	real_t s12 = v1 - 4.0 * v2 + 3.0 * v3;
	real_t s1 = 13.0 * s11 * s11 + 3.0 * s12 * s12;
	real_t s21 = v2 - 2.0 * v3 + v4;
	real_t s22 = v2 - v4;
	real_t s2 = 13.0 * s21 * s21 + 3.0 * s22 * s22;
	real_t s31 = v3 - 2.0 * v4 + v5;
	real_t s32 = 3.0 * v3 - 4.0 * v4 + v5;
	real_t s3 = 13.0 * s31 * s31 + 3.0 * s32 * s32;
	real_t tau61 = (259.0 * v6 - 1895.0 * v5 + 6670.0 * v4 - 2590.0 * v3 - 2785.0 * v2 + 341.0 * v1) / 5760.0;
	real_t tau62 = -(v5 - 12.0 * v4 + 22.0 * v3 - 12.0 * v2 + v1) / 16.0;
	real_t tau63 = -(7.0 * v6 - 47.0 * v5 + 94.0 * v4 - 70.0 * v3 + 11.0 * v2 + 5.0 * v1) / 144.0;
	real_t tau64 = (v5 - 4.0 * v4 + 6.0 * v3 - 4.0 * v2 + v1) / 24.0;
	real_t tau65 = -(-v6 + 5.0 * v5 - 10.0 * v4 + 10.0 * v3 - 5.0 * v2 + v1) / 120.0;
	real_t a1a1 = 1.0, a2a2 = 13.0 / 3.0, a1a3 = 0.5, a3a3 = 3129.0 / 80.0, a2a4 = 21.0 / 5.0;
	real_t a1a5 = 1.0 / 8.0, a4a4 = 87617.0 / 140.0, a3a5 = 14127.0 / 224.0, a5a5 = 252337135.0 / 16128.0;
	real_t s6 = (tau61 * tau61 * a1a1 + tau62 * tau62 * a2a2 + tau61 * tau63 * a1a3 + tau63 * tau63 * a3a3 + tau62 * tau64 * a2a4 + tau61 * tau65 * a1a5 + tau64 * tau64 * a4a4 + tau63 * tau65 * a3a5 + tau65 * tau65 * a5a5) * 12.0;

	// weights
	real_t s55 = (s1 + s3 + 4.0 * s2) / 6.0;
	real_t s5 = sycl::fabs(s6 - s55);
	real_t r0 = 1.0e3;
	real_t r1 = r0 + s5 / (s1 + epsilon);
	real_t r2 = r0 + s5 / (s2 + epsilon);
	real_t r3 = r0 + s5 / (s3 + epsilon);
	real_t r4 = r0 + s5 / (s6 + epsilon);
	real_t a1 = 0.05 * r1 * r1 * r1 * r1;
	real_t a2 = 0.45 * r2 * r2 * r2 * r2;
	real_t a3 = 0.45 * r3 * r3 * r3 * r3;
	real_t a4 = 0.05 * r4 * r4 * r4 * r4;
	real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
	real_t w1 = a1 * tw1;
	real_t w2 = a2 * tw1;
	real_t w3 = a3 * tw1;
	real_t w4 = a4 * tw1;

	// return weighted average
	return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}

SYCL_DEVICE inline real_t WENOCU6M2_P(real_t *f, real_t delta)
{
	real_t epsilon = 1.0e-8;
	int k = 0;
	real_t v1 = *(f + k - 2); // i-2
	real_t v2 = *(f + k - 1); // i-1
	real_t v3 = *(f + k);	  // i
	real_t v4 = *(f + k + 1); // i+1
	real_t v5 = *(f + k + 2); // i+2
	real_t v6 = *(f + k + 3); // i+3

	real_t epsdelta2 = epsilon * delta * delta; // epsilon*delta^2

	// smoothness indicator
	real_t s1 = 13.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3) + epsdelta2;
	// beta_1 + epsilon*delta^2
	real_t s2 = 13.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 * (v2 - v4) * (v2 - v4) + epsdelta2;
	// beta_2 + epsilon*delta^2
	real_t s3 = 13.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5) + epsdelta2;
	// beta_3 + epsilon*delta^2
	real_t s64 = sycl::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 2863984.0 * v4 * v6 - 2792660.0 * v3 * v6 + 1325006.0 * v2 * v6 - 245620.0 * v1 * v6 + 3824847.0 * v5 * v5 - 15880404.0 * v4 * v5 + 15929912.0 * v3 * v5 - 7727988.0 * v2 * v5 + 1458762.0 * v1 * v5 + 17195652.0 * v4 * v4 - 35817664.0 * v3 * v4 + 17905032.0 * v2 * v4 - 3462252.0 * v1 * v4 + 19510972.0 * v3 * v3 - 20427884.0 * v2 * v3 + 4086352.0 * v1 * v3 + 5653317.0 * v2 * v2 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1) / 10080.0 + epsdelta2;

	// weights
	real_t beta_ave = (s1 + s3 + 4.0 * s2 - 6.0 * epsdelta2) / 6.0;
	real_t tau_6 = s64 - beta_ave - epsdelta2;
	real_t chidelta2 = 1.0 / epsilon * delta * delta;

	// 	real_t s5 = sycl::fabs(s64 - s56) + epsilon; // tau_6 + epsilon
	real_t c_q = 1000.0;																					   // C on page 7242
																											   // 	real_t q = 4.0;
	real_t a0 = 0.05 * sycl::pow((c_q + tau_6 / s1 * (beta_ave + chidelta2) / (s1 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_0
	real_t a1 = 0.45 * sycl::pow((c_q + tau_6 / s2 * (beta_ave + chidelta2) / (s2 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_1
	real_t a2 = 0.45 * sycl::pow((c_q + tau_6 / s3 * (beta_ave + chidelta2) / (s3 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_2
	real_t a3 = 0.05 * sycl::pow((c_q + tau_6 / s64 * (beta_ave + chidelta2) / (s64 - epsdelta2 + chidelta2)), _DF(4.0)); // alpha_3
	real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);																	   // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
	real_t w0 = a0 * tw1;																					   // omega_0
	real_t w1 = a1 * tw1;																					   // omega_1
	real_t w2 = a2 * tw1;																					   // omega_2
	real_t w3 = a3 * tw1;																					   // omega_3
	// return weighted average
	return (w0 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w1 * (-v2 + 5.0 * v3 + 2.0 * v4) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w3 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}

SYCL_DEVICE inline real_t WENOCU6M2_M(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 1;
	real_t v1 = *(f + k + 2);
	real_t v2 = *(f + k + 1);
	real_t v3 = *(f + k);
	real_t v4 = *(f + k - 1);
	real_t v5 = *(f + k - 2);
	real_t v6 = *(f + k - 3);

	real_t epsilon = 1.0e-8;

	real_t epsdelta2 = epsilon * delta * delta; // epsilon*delta^2

	// smoothness indicator

	// BIG QUESTION: there is always a " + 3.0"
	// beta_0 + epsilon*delta^2
	real_t s1 = 13.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3) + epsdelta2;
	// beta_1 + epsilon*delta^2
	real_t s2 = 13.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 * (v2 - v4) * (v2 - v4) + epsdelta2;
	// beta_2 + epsilon*delta^2
	real_t s3 = 13.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5) + epsdelta2;
	// beta_3 + epsilon*delta^2
	real_t s64 = sycl::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 2863984.0 * v4 * v6 - 2792660.0 * v3 * v6 + 1325006.0 * v2 * v6 - 245620.0 * v1 * v6 + 3824847.0 * v5 * v5 - 15880404.0 * v4 * v5 + 15929912.0 * v3 * v5 - 7727988.0 * v2 * v5 + 1458762.0 * v1 * v5 + 17195652.0 * v4 * v4 - 35817664.0 * v3 * v4 + 17905032.0 * v2 * v4 - 3462252.0 * v1 * v4 + 19510972.0 * v3 * v3 - 20427884.0 * v2 * v3 + 4086352.0 * v1 * v3 + 5653317.0 * v2 * v2 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1) / 10080.0 + epsdelta2;

	// weights

	real_t beta_ave = (s1 + s3 + 4.0 * s2 - 6.0 * epsdelta2) / 6.0;
	real_t tau_6 = s64 - beta_ave - epsdelta2;
	real_t chidelta2 = 1.0 / epsilon * delta * delta;

	// 	real_t s5 = sycl::fabs(s64 - s56) + epsilon; // tau_6 + epsilon
	real_t c_q = 1000.0;																					   // C on page 7242
																											   // 	real_t q = 4.0;
	real_t a0 = 0.05 * sycl::pow((c_q + tau_6 / s1 * (beta_ave + chidelta2) / (s1 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_0
	real_t a1 = 0.45 * sycl::pow((c_q + tau_6 / s2 * (beta_ave + chidelta2) / (s2 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_1
	real_t a2 = 0.45 * sycl::pow((c_q + tau_6 / s3 * (beta_ave + chidelta2) / (s3 - epsdelta2 + chidelta2)), _DF(4.0));	  // alpha_2
	real_t a3 = 0.05 * sycl::pow((c_q + tau_6 / s64 * (beta_ave + chidelta2) / (s64 - epsdelta2 + chidelta2)), _DF(4.0)); // alpha_3
	real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);																	   // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
	real_t w0 = a0 * tw1;																					   // omega_0
	real_t w1 = a1 * tw1;																					   // omega_1
	real_t w2 = a2 * tw1;																					   // omega_2
	real_t w3 = a3 * tw1;																					   // omega_3
	// return weighted average
	return (w0 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w1 * (-v2 + 5.0 * v3 + 2.0 * v4) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w3 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}
