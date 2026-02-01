#pragma once

#include "Utils_schemes.hpp"

//-----------------------------------------------------------------------------------------
//		the 7th WENO Scheme
//-----------------------------------------------------------------------------------------
SYCL_DEVICE inline real_t weno7_P(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 0;
	real_t v1 = *(f + k - 3);
	real_t v2 = *(f + k - 2);
	real_t v3 = *(f + k - 1);
	real_t v4 = *(f + k);
	real_t v5 = *(f + k + 1);
	real_t v6 = *(f + k + 2);
	real_t v7 = *(f + k + 3);

	real_t ep = 1.0e-7;
	real_t C0 = 1.0 / 35.0;
	real_t C1 = 12.0 / 35.0;
	real_t C2 = 18.0 / 35.0;
	real_t C3 = 4.0 / 35.0;
	// 1  阶导数
	real_t S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
	real_t S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
	real_t S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
	real_t S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
	// 2 阶导数
	real_t S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
	real_t S21 = v3 - 2.0 * v4 + v5;
	real_t S22 = v4 - 2.0 * v5 + v6;
	real_t S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
	// 3 阶导数
	real_t S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
	real_t S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
	real_t S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
	real_t S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
	// 光滑度量因子
	real_t S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
	real_t S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
	real_t S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
	real_t S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
	// // 新的光滑度量因子, Xinliang Li
	// real_t S0 = S10 * S10 + S20 * S20;
	// real_t S1 = S11 * S11 + S21 * S21;
	// real_t S2 = S12 * S12 + S22 * S22;
	// real_t S3 = S13 * S13 + S23 * S23;
	// Alpha weights
	real_t a0 = C0 / ((ep + S0) * (ep + S0));
	real_t a1 = C1 / ((ep + S1) * (ep + S1));
	real_t a2 = C2 / ((ep + S2) * (ep + S2));
	real_t a3 = C3 / ((ep + S3) * (ep + S3));
	// Non-linear weigths
	real_t W0 = a0 / (a0 + a1 + a2 + a3);
	real_t W1 = a1 / (a0 + a1 + a2 + a3);
	real_t W2 = a2 / (a0 + a1 + a2 + a3);
	real_t W3 = a3 / (a0 + a1 + a2 + a3);
	// 4阶差分格式的通量
	real_t q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
	real_t q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
	real_t q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
	real_t q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
	// 由4个4阶差分格式组合成1个7阶差分格式
	return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}

SYCL_DEVICE inline real_t weno7_M(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 1;
	real_t v1 = *(f + k + 3);
	real_t v2 = *(f + k + 2);
	real_t v3 = *(f + k + 1);
	real_t v4 = *(f + k);
	real_t v5 = *(f + k - 1);
	real_t v6 = *(f + k - 2);
	real_t v7 = *(f + k - 3);

	real_t ep = 1.0e-7;
	real_t C0 = 1.0 / 35.0;
	real_t C1 = 12.0 / 35.0;
	real_t C2 = 18.0 / 35.0;
	real_t C3 = 4.0 / 35.0;
	// 1  阶导数
	real_t S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
	real_t S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
	real_t S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
	real_t S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
	// 2 阶导数
	real_t S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
	real_t S21 = v3 - 2.0 * v4 + v5;
	real_t S22 = v4 - 2.0 * v5 + v6;
	real_t S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
	// 3 阶导数
	real_t S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
	real_t S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
	real_t S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
	real_t S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
	// 光滑度量因子
	real_t S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
	real_t S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
	real_t S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
	real_t S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
	// // 新的光滑度量因子, Xinliang Li
	// real_t S0 = S10 * S10 + S20 * S20;
	// real_t S1 = S11 * S11 + S21 * S21;
	// real_t S2 = S12 * S12 + S22 * S22;
	// real_t S3 = S13 * S13 + S23 * S23;
	// Alpha weights
	real_t a0 = C0 / ((ep + S0) * (ep + S0));
	real_t a1 = C1 / ((ep + S1) * (ep + S1));
	real_t a2 = C2 / ((ep + S2) * (ep + S2));
	real_t a3 = C3 / ((ep + S3) * (ep + S3));
	// Non-linear weigths
	real_t W0 = a0 / (a0 + a1 + a2 + a3);
	real_t W1 = a1 / (a0 + a1 + a2 + a3);
	real_t W2 = a2 / (a0 + a1 + a2 + a3);
	real_t W3 = a3 / (a0 + a1 + a2 + a3);
	// 4阶差分格式的通量
	real_t q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
	real_t q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
	real_t q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
	real_t q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
	// 由4个4阶差分格式组合成1个7阶差分格式
	return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}

SYCL_DEVICE inline real_t weno7Z_P(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 0;
	real_t v1 = *(f + k - 3);
	real_t v2 = *(f + k - 2);
	real_t v3 = *(f + k - 1);
	real_t v4 = *(f + k);
	real_t v5 = *(f + k + 1);
	real_t v6 = *(f + k + 2);
	real_t v7 = *(f + k + 3);

	real_t ep = 1.0e-7;
	real_t C0 = 1.0 / 35.0;
	real_t C1 = 12.0 / 35.0;
	real_t C2 = 18.0 / 35.0;
	real_t C3 = 4.0 / 35.0;
	// 1  阶导数
	real_t S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
	real_t S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
	real_t S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
	real_t S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
	// 2 阶导数
	real_t S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
	real_t S21 = v3 - 2.0 * v4 + v5;
	real_t S22 = v4 - 2.0 * v5 + v6;
	real_t S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
	// 3 阶导数
	real_t S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
	real_t S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
	real_t S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
	real_t S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
	// 光滑度量因子
	real_t S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
	real_t S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
	real_t S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
	real_t S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
	// Alpha weights
	real_t tau7 = sycl::fabs(S0 - S3);
	real_t a0 = C0 * (1.0 + tau7 / (S0 + 1.0e-40));
	real_t a1 = C1 * (1.0 + tau7 / (S1 + 1.0e-40));
	real_t a2 = C2 * (1.0 + tau7 / (S2 + 1.0e-40));
	real_t a3 = C3 * (1.0 + tau7 / (S3 + 1.0e-40));
	// Non-linear weigths
	real_t W0 = a0 / (a0 + a1 + a2 + a3);
	real_t W1 = a1 / (a0 + a1 + a2 + a3);
	real_t W2 = a2 / (a0 + a1 + a2 + a3);
	real_t W3 = a3 / (a0 + a1 + a2 + a3);
	// 4阶差分格式的通量
	real_t q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
	real_t q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
	real_t q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
	real_t q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
	// 由4个4阶差分格式组合成1个7阶差分格式
	return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}

SYCL_DEVICE inline real_t weno7Z_M(real_t *f, real_t delta)
{
	// assign value to v1, v2,...
	int k = 1;
	real_t v1 = *(f + k + 3);
	real_t v2 = *(f + k + 2);
	real_t v3 = *(f + k + 1);
	real_t v4 = *(f + k);
	real_t v5 = *(f + k - 1);
	real_t v6 = *(f + k - 2);
	real_t v7 = *(f + k - 3);

	real_t ep = 1.0e-7;
	real_t C0 = 1.0 / 35.0;
	real_t C1 = 12.0 / 35.0;
	real_t C2 = 18.0 / 35.0;
	real_t C3 = 4.0 / 35.0;
	// 1  阶导数
	real_t S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
	real_t S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
	real_t S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
	real_t S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
	// 2 阶导数
	real_t S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
	real_t S21 = v3 - 2.0 * v4 + v5;
	real_t S22 = v4 - 2.0 * v5 + v6;
	real_t S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
	// 3 阶导数
	real_t S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
	real_t S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
	real_t S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
	real_t S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
	// 光滑度量因子
	real_t S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
	real_t S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
	real_t S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
	real_t S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
	// Alpha weights
	real_t tau7 = sycl::fabs(S0 - S3);
	real_t a0 = C0 * (1.0 + tau7 / (S0 + 1.0e-40));
	real_t a1 = C1 * (1.0 + tau7 / (S1 + 1.0e-40));
	real_t a2 = C2 * (1.0 + tau7 / (S2 + 1.0e-40));
	real_t a3 = C3 * (1.0 + tau7 / (S3 + 1.0e-40));
	// Non-linear weigths
	real_t W0 = a0 / (a0 + a1 + a2 + a3);
	real_t W1 = a1 / (a0 + a1 + a2 + a3);
	real_t W2 = a2 / (a0 + a1 + a2 + a3);
	real_t W3 = a3 / (a0 + a1 + a2 + a3);
	// 4阶差分格式的通量
	real_t q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
	real_t q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
	real_t q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
	real_t q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
	// 由4个4阶差分格式组合成1个7阶差分格式
	return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}
