#pragma once

#include "global_class.h"
// TODO: NO std::cmath functions used if schemes function referenced, use sycl::math_function<real_t>

real_t minmod(real_t r);
real_t van_Leer(real_t r);
real_t van_Albada(real_t r);
void MUSCL(real_t p[4], real_t &LL, real_t &RR, int flag);
real_t KroneckerDelta(const int i, const int j);
// schemes
real_t upwind_P(real_t *f, real_t delta);
real_t upwind_M(real_t *f, real_t delta);
// 5th-upwind
real_t linear_5th_P(real_t *f, real_t delta);
real_t linear_5th_M(real_t *f, real_t delta);
real_t linear_2th(real_t *f, real_t delta);
real_t linear_4th(real_t *f, real_t delta);
real_t linear_6th(real_t *f, real_t delta);
real_t linear_3rd_P(real_t *f, real_t delta);
real_t linear_3rd_M(real_t *f, real_t delta);
real_t du_upwind5(real_t *f, real_t delta);
real_t f2_upwind5(real_t *f, real_t delta);
// original weno
real_t weno5_P(real_t *f, real_t delta);
real_t weno5_M(real_t *f, real_t delta);
real_t weno5old_P(real_t *f, real_t delta);
real_t weno5old_M(real_t *f, real_t delta);
real_t weno7_P(real_t *f, real_t delta);
real_t weno7_M(real_t *f, real_t delta);
// 6th-order weno
real_t WENOCU6_P(real_t *f, real_t delta);
real_t WENOCU6_M(real_t *f, real_t delta);
real_t WENOCU6M1_P(real_t *f, real_t delta);
real_t WENOCU6M1_M(real_t *f, real_t delta);
real_t WENOCU6M2_P(real_t *f, real_t delta);
real_t WENOCU6M2_M(real_t *f, real_t delta);
real_t TENO5_P(real_t *f, real_t delta);
real_t TENO5_M(real_t *f, real_t delta);
real_t TENO6_OPT_P(real_t *f, real_t delta);
real_t TENO6_OPT_M(real_t *f, real_t delta);
// wenoZ
real_t weno5Z_P(real_t *f, real_t delta);
real_t weno5Z_M(real_t *f, real_t delta);
real_t weno7Z_P(real_t *f, real_t delta);
real_t weno7Z_M(real_t *f, real_t delta);
// WENO-AO
real_t WENOAO53_P(real_t *f, real_t delta);
real_t WENOAO53_M(real_t *f, real_t delta);
real_t WENOAO73_P(real_t *f, real_t delta);
real_t WENOAO73_M(real_t *f, real_t delta);
real_t WENOAO753_P(real_t *f, real_t delta);
real_t WENOAO753_M(real_t *f, real_t delta);

real_t Weno5L2_P(real_t *f, real_t delta, real_t lambda);
real_t Weno5L2_M(real_t *f, real_t delta, real_t lambda);

/**
 * @brief KroneckerDelta
 * @return real_t
 */
real_t KroneckerDelta(const int i, const int j)
{
    real_t f = i == j ? 1 : 0;
    return f;
}
/**
 * @brief van Leer limiter
 *
 * @param r
 * @return real_t
 */
real_t van_Leer(real_t r)
{
    return (r + std::abs(r)) / (1.0 + std::abs(r));
}
/**
 * @brief van Albada limiter
 *
 * @param r
 * @return real_t
 */
real_t van_Albada(real_t r)
{
    return (r * r + r) / (1.0 + r * r);
}
/**
 * @brief the minmod limiter
 *
 * @param r
 * @return real_t
 */
real_t minmod(real_t r)
{
    real_t minmod = 0;
    real_t aa = 1.0;
    if (r > 0)
        minmod = std::min(r, aa);
    return minmod;
}
/**
 * @brief the MUSCL reconstruction
 *
 * @param p
 * @param LL
 * @param RR
 * @param flag
 */
void MUSCL(real_t p[4], real_t &LL, real_t &RR, int flag)
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
 * @brief upwind scheme
 *
 * @param f
 * @param delta
 * @return real_t
 */
real_t upwind_P(real_t *f, real_t delta)
{
    return *f;
}
real_t upwind_M(real_t *f, real_t delta)
{
    return *(f + 1);
}
real_t linear_3rd_P(real_t *f, real_t delta)
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
real_t linear_3rd_M(real_t *f, real_t delta)
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
real_t linear_5th_P(real_t *f, real_t delta)
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
real_t linear_5th_M(real_t *f, real_t delta)
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
real_t linear_2th(real_t *f, real_t delta)
{
    real_t v1 = *f;
    real_t v2 = *(f + 1);
    real_t vv = (v1 + v2) / 2.0;

    return vv;
}
real_t linear_4th(real_t *f, real_t delta)
{
    real_t v1 = *(f - 1);
    real_t v2 = *f;
    real_t v3 = *(f + 1);
    real_t v4 = *(f + 2);
    real_t vv = (-v1 + 7.0 * v2 + 7.0 * v3 - v4) / 12.0;

    return vv;
}
real_t linear_6th(real_t *f, real_t delta)
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
real_t du_upwind5(real_t *f, real_t delta)
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
real_t f2_upwind5(real_t *f, real_t delta)
{
    real_t v1 = *(f - 2);
    real_t v2 = *(f - 1);
    real_t v3 = *f;
    real_t v4 = *(f + 1);
    real_t v5 = *(f + 2);
    real_t v6 = *(f + 3);
    return (v1 - 8.0 * v2 + 37.0 * v3 + 37.0 * v4 - 8.0 * v5 + v6) / 60.0;
}

real_t weno5_P(real_t *f, real_t delta)
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
real_t weno5_M(real_t *f, real_t delta)
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

/**
 * @brief the 5th WENO Scheme
 *
 * @param f
 * @param delta
 * @return real_t
 */
const double epsilon_weno = 1.0e-6;
real_t weno5old_P(real_t *f, real_t delta)
{ // version from Lyx
    int k;
    real_t v1, v2, v3, v4, v5;
    real_t a1, a2, a3;
    real_t w1, w2, w3;

    // assign value to v1, v2,...
    k = 0;
    v1 = *(f + k - 2);
    v2 = *(f + k - 1);
    v3 = *(f + k);
    v4 = *(f + k + 1);
    v5 = *(f + k + 2);

    // smoothness indicator
    real_t s1 = _DF(13.0) * (v1 - _DF(2.0) * v2 + v3) * (v1 - _DF(2.0) * v2 + v3) + _DF(3.0) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3);
    s1 /= _DF(12.0);
    real_t s2 = _DF(13.0) * (v2 - _DF(2.0) * v3 + v4) * (v2 - _DF(2.0) * v3 + v4) + _DF(3.0) * (v2 - v4) * (v2 - v4);
    s2 /= _DF(12.0);
    real_t s3 = _DF(13.0) * (v3 - _DF(2.0) * v4 + v5) * (v3 - _DF(2.0) * v4 + v5) + _DF(3.0) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5);
    s3 /= _DF(12.0);

    // weights
    a1 = _DF(0.1) / ((epsilon_weno + s1) * (epsilon_weno + s1));
    a2 = _DF(0.6) / ((epsilon_weno + s2) * (epsilon_weno + s2));
    a3 = _DF(0.3) / ((epsilon_weno + s3) * (epsilon_weno + s3));
    real_t tw1 = _DF(1.0) / (a1 + a2 + a3);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;
    real_t temp = w1 * (_DF(2.0) * v1 - _DF(7.0) * v2 + _DF(11.0) * v3) / _DF(6.0) + w2 * (-v2 + _DF(5.0) * v3 + _DF(2.0) * v4) / _DF(6.0) + w3 * (_DF(2.0) * v3 + _DF(5.0) * v4 - v5) / _DF(6.0);
    // return weighted average
    return temp;
}
/**
 * @brief the 5th WENO Scheme
 *
 * @param f
 * @param delta
 * @return real_t
 */
real_t weno5old_M(real_t *f, real_t delta)
{ // version from Lyx
    int k;
    real_t v1, v2, v3, v4, v5;
    real_t a1, a2, a3;
    real_t w1, w2, w3;

    // assign value to v1, v2,...
    k = 1;
    v1 = *(f + k + 2);
    v2 = *(f + k + 1);
    v3 = *(f + k);
    v4 = *(f + k - 1);
    v5 = *(f + k - 2);

    // smoothness indicator
    double s1 = _DF(13.0) * (v1 - _DF(2.0) * v2 + v3) * (v1 - _DF(2.0) * v2 + v3) + _DF(3.0) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3);
    s1 /= _DF(12.0);
    double s2 = _DF(13.0) * (v2 - _DF(2.0) * v3 + v4) * (v2 - _DF(2.0) * v3 + v4) + _DF(3.0) * (v2 - v4) * (v2 - v4);
    s2 /= _DF(12.0);
    double s3 = _DF(13.0) * (v3 - _DF(2.0) * v4 + v5) * (v3 - _DF(2.0) * v4 + v5) + _DF(3.0) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5);
    s3 /= _DF(12.0);

    // weights
    a1 = _DF(0.1) / ((epsilon_weno + s1) * (epsilon_weno + s1));
    a2 = _DF(0.6) / ((epsilon_weno + s2) * (epsilon_weno + s2));
    a3 = _DF(0.3) / ((epsilon_weno + s3) * (epsilon_weno + s3));
    double tw1 = _DF(1.0) / (a1 + a2 + a3);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;

    real_t temp = w1 * (_DF(2.0) * v1 - _DF(7.0) * v2 + _DF(11.0) * v3) / _DF(6.0) + w2 * (-v2 + _DF(5.0) * v3 + _DF(2.0) * v4) / _DF(6.0) + w3 * (_DF(2.0) * v3 + _DF(5.0) * v4 - v5) / _DF(6.0);
    // return weighted average
    return temp;
}
//-----------------------------------------------------------------------------------------
//		the 7th WENO Scheme
//-----------------------------------------------------------------------------------------
real_t weno7_P(real_t *f, real_t delta)
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
real_t weno7_M(real_t *f, real_t delta)
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

/**
 * @brief  WENO-AO(5,3) scheme from
 *                  Balsara et al., An efficient class of WENO schemes with adaptive order. (2016)
 *                  Kumar et al., Simple smoothness indicator and multi-level adaptive order WENO scheme for hyperbolic conservation laws. (2018)
 */
real_t WENOAO53_P(real_t *f, real_t delta)
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
    real_t tau = (std::abs(s5 - s1) + std::abs(s5 - s2) + std::abs(s5 - s3)) / 3.0;
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

real_t WENOAO53_M(real_t *f, real_t delta)
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
    real_t tau = (std::abs(s5 - s1) + std::abs(s5 - s2) + std::abs(s5 - s3)) / 3.0;
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
real_t WENOAO73_P(real_t *f, real_t delta)
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
    real_t tau = (std::abs(s7 - s1) + std::abs(s7 - s2) + std::abs(s7 - s3)) / 3.0;
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
real_t WENOAO73_M(real_t *f, real_t delta)
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
    real_t tau = (std::abs(s7 - s1) + std::abs(s7 - s2) + std::abs(s7 - s3)) / 3.0;
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
real_t WENOAO753_P(real_t *f, real_t delta)
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
    real_t tau_7 = (std::abs(s7 - s1) + std::abs(s7 - s2) + std::abs(s7 - s3)) / 3.0;
    real_t tau_5 = (std::abs(s5 - s1) + std::abs(s5 - s2) + std::abs(s5 - s3)) / 3.0;
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
    real_t sigma = std::abs(s7 - s5);
    real_t b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
    real_t b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

    real_t one_b_sum = b7 + b5;

    real_t w_ao_7 = b7 / one_b_sum;
    real_t w_ao_5 = b5 / one_b_sum;

    // Return value of hybridized reconstructed polynomial
    return (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}

real_t WENOAO753_M(real_t *f, real_t delta)
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
    real_t tau_7 = (std::abs(s7 - s1) + std::abs(s7 - s2) + std::abs(s7 - s3)) / 3.0;
    real_t tau_5 = (std::abs(s5 - s1) + std::abs(s5 - s2) + std::abs(s5 - s3)) / 3.0;
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
    real_t sigma = std::abs(s7 - s5);
    real_t b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
    real_t b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

    real_t one_b_sum = b7 + b5;

    real_t w_ao_7 = b7 / one_b_sum;
    real_t w_ao_5 = b5 / one_b_sum;

    // Return value of hybridized reconstructed polynomial
    return (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}

real_t WENOCU6_P(real_t *f, real_t delta)
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
    real_t s5 = fabs(s6 - s55);
    real_t r0 = 20.0;
    real_t r1 = r0 + s5 / (s1 + epsilon);
    real_t r2 = r0 + s5 / (s2 + epsilon);
    real_t r3 = r0 + s5 / (s3 + epsilon);
    real_t r4 = r0 + s5 / (s6 + epsilon);
    real_t a1 = 0.05 * r1;
    real_t a2 = 0.45 * r2;
    real_t a3 = 0.45 * r3;
    real_t a4 = 0.05 * r4;
    real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
    real_t w1 = a1 * tw1;
    real_t w2 = a2 * tw1;
    real_t w3 = a3 * tw1;
    real_t w4 = a4 * tw1;

    // return weighted average
    return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}
// this is WENOCU6
real_t WENOCU6_M(real_t *f, real_t delta)
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
    real_t s5 = fabs(s6 - s55);
    real_t r0 = 20.0;
    real_t r1 = r0 + s5 / (s1 + epsilon);
    real_t r2 = r0 + s5 / (s2 + epsilon);
    real_t r3 = r0 + s5 / (s3 + epsilon);
    real_t r4 = r0 + s5 / (s6 + epsilon);
    real_t a1 = 0.05 * r1;
    real_t a2 = 0.45 * r2;
    real_t a3 = 0.45 * r3;
    real_t a4 = 0.05 * r4;
    real_t tw1 = 1.0 / (a1 + a2 + a3 + a4);
    real_t w1 = a1 * tw1;
    real_t w2 = a2 * tw1;
    real_t w3 = a3 * tw1;
    real_t w4 = a4 * tw1;

    // return weighted average
    return (w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) + w3 * (2.0 * v3 + 5.0 * v4 - v5) + w4 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}

real_t WENOCU6M1_P(real_t *f, real_t delta)
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
    real_t s5 = fabs(s6 - s55);
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
real_t WENOCU6M1_M(real_t *f, real_t delta)
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
    real_t s5 = fabs(s6 - s55);
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

real_t TENO5_P(real_t *f, real_t delta)
{
    int k;
    real_t v1, v2, v3, v4, v5;
    real_t b1, b2, b3;
    real_t a1, a2, a3, w1, w2, w3;
    real_t Variation1, Variation2, Variation3;

    // assign value to v1, v2,...
    k = 0;
    v1 = *(f + k - 2);
    v2 = *(f + k - 1);
    v3 = *(f + k);
    v4 = *(f + k + 1);
    v5 = *(f + k + 2);

    real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
    real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
    real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);

    real_t tau5 = std::abs(s3 - s2);

    a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40), 6.0);
    a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40), 6.0);
    a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40), 6.0);

    b1 = a1 / (a1 + a2 + a3);
    b2 = a2 / (a1 + a2 + a3);
    b3 = a3 / (a1 + a2 + a3);

    b1 = b1 < 1.0e-7 ? 0. : 1.;
    b2 = b2 < 1.0e-7 ? 0. : 1.;
    b3 = b3 < 1.0e-7 ? 0. : 1.;

    Variation1 = -1.0 * v2 + 5.0 * v3 + 2.0 * v4;
    Variation2 = 2.0 * v3 + 5.0 * v4 - 1.0 * v5;
    Variation3 = 2.0 * v1 - 7.0 * v2 + 11.0 * v3;

    a1 = 0.600 * b1;
    a2 = 0.300 * b2;
    a3 = 0.100 * b3;

    w1 = a1 / (a1 + a2 + a3);
    w2 = a2 / (a1 + a2 + a3);
    w3 = a3 / (a1 + a2 + a3);

    return 1.0 / 6.0 * (w1 * Variation1 + w2 * Variation2 + w3 * Variation3);
}

real_t TENO5_M(real_t *f, real_t delta)
{
    int k;
    real_t v1, v2, v3, v4, v5;
    real_t b1, b2, b3;
    real_t a1, a2, a3, w1, w2, w3;
    real_t Variation1, Variation2, Variation3;

    // assign value to v1, v2,...
    k = 1;
    v1 = *(f + k + 2);
    v2 = *(f + k + 1);
    v3 = *(f + k);
    v4 = *(f + k - 1);
    v5 = *(f + k - 2);

    real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
    real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
    real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);

    real_t tau5 = std::abs(s3 - s2);

    a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40), 6.0);
    a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40), 6.0);
    a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40), 6.0);

    b1 = a1 / (a1 + a2 + a3);
    b2 = a2 / (a1 + a2 + a3);
    b3 = a3 / (a1 + a2 + a3);

    b1 = b1 < 1.0e-7 ? 0. : 1.;
    b2 = b2 < 1.0e-7 ? 0. : 1.;
    b3 = b3 < 1.0e-7 ? 0. : 1.;

    Variation1 = -1.0 * v2 + 5.0 * v3 + 2.0 * v4;
    Variation2 = 2.0 * v3 + 5.0 * v4 - 1.0 * v5;
    Variation3 = 2.0 * v1 - 7.0 * v2 + 11.0 * v3;

    a1 = 0.600 * b1;
    a2 = 0.300 * b2;
    a3 = 0.100 * b3;

    w1 = a1 / (a1 + a2 + a3);
    w2 = a2 / (a1 + a2 + a3);
    w3 = a3 / (a1 + a2 + a3);

    return 1.0 / 6.0 * (w1 * Variation1 + w2 * Variation2 + w3 * Variation3);
}

real_t TENO6_OPT_P(real_t *f, real_t delta)
{
    int k;
    real_t v1, v2, v3, v4, v5, v6;
    real_t b1, b2, b3, b4, b5;
    real_t a1, a2, a3, a4, a5, w1, w2, w3, w4, w5;
    real_t Variation1, Variation2, Variation3, Variation4, Variation5;

    // assign value to v1, v2,...
    k = 0;
    v1 = *(f + k - 2);
    v2 = *(f + k - 1);
    v3 = *(f + k);
    v4 = *(f + k + 1);
    v5 = *(f + k + 2);
    v6 = *(f + k + 3);

    real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
    real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
    real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
    real_t s4 = 1. / 240. * std::fabs((2107.0 * v3 * v3 - 9402.0 * v3 * v4 + 11003.0 * v4 * v4 + 7042.0 * v3 * v5 - 17246.0 * v4 * v5 + 7043.0 * v5 * v5 - 1854.0 * v3 * v6 + 4642.0 * v4 * v6 - 3882.0 * v5 * v6 + 547.0 * v6 * v6));

    real_t s64 = 1.0 / 12.0 * std::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 3824847.0 * v5 * v5 + 2863984.0 * v4 * v6 - 15880404.0 * v4 * v5 + 17195652.0 * v4 * v4 - 2792660.0 * v3 * v6 - 35817664.0 * v3 * v4 + 19510972.0 * v3 * v3 + 1325006.0 * v2 * v6 - 7727988.0 * v2 * v5 + 17905032.0 * v2 * v4 - 20427884.0 * v2 * v3 + 5653317.0 * v2 * v2 - 245620.0 * v1 * v6 + 1458762.0 * v1 * v5 - 3462252.0 * v1 * v4 + 4086352.0 * v1 * v3 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1 + 15929912.0 * v3 * v5) / 10080.0;

    real_t tau6 = std::abs(s64 - (s3 + s2 + 4.0 * s1) / 6.0);

    a1 = 1. / 4. * std::pow(1.0 + tau6 / (s1 + 1.0e-40), 6.0);
    a2 = 1. / 4. * std::pow(1.0 + tau6 / (s2 + 1.0e-40), 6.0);
    a3 = 1. / 4. * std::pow(1.0 + tau6 / (s3 + 1.0e-40), 6.0);
    a4 = 1. / 4. * std::pow(1.0 + tau6 / (s4 + 1.0e-40), 6.0);

    b1 = a1 / (a1 + a2 + a3 + a4);
    b2 = a2 / (a1 + a2 + a3 + a4);
    b3 = a3 / (a1 + a2 + a3 + a4);
    b4 = a4 / (a1 + a2 + a3 + a4);

    b1 = b1 < 1.0e-7 ? 0. : 1.;
    b2 = b2 < 1.0e-7 ? 0. : 1.;
    b3 = b3 < 1.0e-7 ? 0. : 1.;
    b4 = b4 < 1.0e-7 ? 0. : 1.;

    Variation1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 2.0 / 6.0 * v4 - v3;
    Variation2 = 2. / 6. * v3 + 5. / 6. * v4 - 1. / 6. * v5 - v3;
    Variation3 = 2. / 6. * v1 - 7. / 6. * v2 + 11. / 6. * v3 - v3;
    Variation4 = 3. / 12. * v3 + 13. / 12. * v4 - 5. / 12. * v5 + 1. / 12. * v6 - v3;

    a1 = 0.462 * b1;
    a2 = 0.300 * b2;
    a3 = 0.054 * b3;
    a4 = 0.184 * b4;

    w1 = a1 / (a1 + a2 + a3 + a4);
    w2 = a2 / (a1 + a2 + a3 + a4);
    w3 = a3 / (a1 + a2 + a3 + a4);
    w4 = a4 / (a1 + a2 + a3 + a4);

    return v3 + w1 * Variation1 + w2 * Variation2 + w3 * Variation3 + w4 * Variation4;
}
real_t TENO6_OPT_M(real_t *f, real_t delta)
{
    int k;
    real_t v1, v2, v3, v4, v5, v6;
    real_t b1, b2, b3, b4;
    real_t a1, a2, a3, a4, w1, w2, w3, w4;
    real_t Variation1, Variation2, Variation3, Variation4;

    k = 1;
    v1 = *(f + k + 2);
    v2 = *(f + k + 1);
    v3 = *(f + k);
    v4 = *(f + k - 1);
    v5 = *(f + k - 2);
    v6 = *(f + k - 3);

    real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
    real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
    real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
    real_t s4 = 1. / 240. * std::fabs((2107.0 * v3 * v3 - 9402.0 * v3 * v4 + 11003.0 * v4 * v4 + 7042.0 * v3 * v5 - 17246.0 * v4 * v5 + 7043.0 * v5 * v5 - 1854.0 * v3 * v6 + 4642.0 * v4 * v6 - 3882.0 * v5 * v6 + 547.0 * v6 * v6));

    real_t s64 = 1.0 / 12.0 * std::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 3824847.0 * v5 * v5 + 2863984.0 * v4 * v6 - 15880404.0 * v4 * v5 + 17195652.0 * v4 * v4 - 2792660.0 * v3 * v6 - 35817664.0 * v3 * v4 + 19510972.0 * v3 * v3 + 1325006.0 * v2 * v6 - 7727988.0 * v2 * v5 + 17905032.0 * v2 * v4 - 20427884.0 * v2 * v3 + 5653317.0 * v2 * v2 - 245620.0 * v1 * v6 + 1458762.0 * v1 * v5 - 3462252.0 * v1 * v4 + 4086352.0 * v1 * v3 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1 + 15929912.0 * v3 * v5) / 10080.0;

    real_t tau6 = std::abs(s64 - (s3 + s2 + 4.0 * s1) / 6.0);

    a1 = 1. / 4. * std::pow(1.0 + tau6 / (s1 + 1.0e-40), 6.0);
    a2 = 1. / 4. * std::pow(1.0 + tau6 / (s2 + 1.0e-40), 6.0);
    a3 = 1. / 4. * std::pow(1.0 + tau6 / (s3 + 1.0e-40), 6.0);
    a4 = 1. / 4. * std::pow(1.0 + tau6 / (s4 + 1.0e-40), 6.0);

    b1 = a1 / (a1 + a2 + a3 + a4);
    b2 = a2 / (a1 + a2 + a3 + a4);
    b3 = a3 / (a1 + a2 + a3 + a4);
    b4 = a4 / (a1 + a2 + a3 + a4);

    b1 = b1 < 1.0e-7 ? 0. : 1.;
    b2 = b2 < 1.0e-7 ? 0. : 1.;
    b3 = b3 < 1.0e-7 ? 0. : 1.;
    b4 = b4 < 1.0e-7 ? 0. : 1.;

    Variation1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 2.0 / 6.0 * v4 - v3;
    Variation2 = 2. / 6. * v3 + 5. / 6. * v4 - 1. / 6. * v5 - v3;
    Variation3 = 2. / 6. * v1 - 7. / 6. * v2 + 11. / 6. * v3 - v3;
    Variation4 = 3. / 12. * v3 + 13. / 12. * v4 - 5. / 12. * v5 + 1. / 12. * v6 - v3;

    a1 = 0.462 * b1;
    a2 = 0.300 * b2;
    a3 = 0.054 * b3;
    a4 = 0.184 * b4;

    w1 = a1 / (a1 + a2 + a3 + a4);
    w2 = a2 / (a1 + a2 + a3 + a4);
    w3 = a3 / (a1 + a2 + a3 + a4);
    w4 = a4 / (a1 + a2 + a3 + a4);

    return v3 + w1 * Variation1 + w2 * Variation2 + w3 * Variation3 + w4 * Variation4;
}

real_t weno5Z_P(real_t *f, real_t delta)
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
    real_t s5 = fabs(s1 - s3);
    a1 = 0.1 * (1.0 + s5 / (s1 + epsilon));
    a2 = 0.6 * (1.0 + s5 / (s2 + epsilon));
    a3 = 0.3 * (1.0 + s5 / (s3 + epsilon));

    w1 = a1 / (a1 + a2 + a3);
    w2 = a2 / (a1 + a2 + a3);
    w3 = a3 / (a1 + a2 + a3);

    // return weighted average
    return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}
real_t weno5Z_M(real_t *f, real_t delta)
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
    real_t s5 = fabs(s1 - s3);
    a1 = 0.1 * (1.0 + s5 / (s1 + epsilon));
    a2 = 0.6 * (1.0 + s5 / (s2 + epsilon));
    a3 = 0.3 * (1.0 + s5 / (s3 + epsilon));

    w1 = a1 / (a1 + a2 + a3);
    w2 = a2 / (a1 + a2 + a3);
    w3 = a3 / (a1 + a2 + a3);

    // return weighted average
    return w1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) / 6.0 + w2 * (-v2 + 5.0 * v3 + 2.0 * v4) / 6.0 + w3 * (2.0 * v3 + 5.0 * v4 - v5) / 6.0;
}

real_t weno7Z_P(real_t *f, real_t delta)
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
    real_t tau7 = std::abs(S0 - S3);
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
real_t weno7Z_M(real_t *f, real_t delta)
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
    real_t tau7 = std::abs(S0 - S3);
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

real_t WENOCU6M2_P(real_t *f, real_t delta)
{
    real_t epsilon = 1.0e-8;
    int k = 0;
    real_t v1 = *(f + k - 2); // i-2
    real_t v2 = *(f + k - 1); // i-1
    real_t v3 = *(f + k);     // i
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
    real_t s64 = fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 2863984.0 * v4 * v6 - 2792660.0 * v3 * v6 + 1325006.0 * v2 * v6 - 245620.0 * v1 * v6 + 3824847.0 * v5 * v5 - 15880404.0 * v4 * v5 + 15929912.0 * v3 * v5 - 7727988.0 * v2 * v5 + 1458762.0 * v1 * v5 + 17195652.0 * v4 * v4 - 35817664.0 * v3 * v4 + 17905032.0 * v2 * v4 - 3462252.0 * v1 * v4 + 19510972.0 * v3 * v3 - 20427884.0 * v2 * v3 + 4086352.0 * v1 * v3 + 5653317.0 * v2 * v2 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1) / 10080.0 + epsdelta2;

    // weights
    real_t beta_ave = (s1 + s3 + 4.0 * s2 - 6.0 * epsdelta2) / 6.0;
    real_t tau_6 = s64 - beta_ave - epsdelta2;
    real_t chidelta2 = 1.0 / epsilon * delta * delta;

    // 	real_t s5 = fabs(s64 - s56) + epsilon; // tau_6 + epsilon
    real_t c_q = 1000.0;                                                                                       // C on page 7242
                                                                                                               // 	real_t q = 4.0;
    real_t a0 = 0.05 * pow((c_q + tau_6 / s1 * (beta_ave + chidelta2) / (s1 - epsdelta2 + chidelta2)), 4.0);   // alpha_0
    real_t a1 = 0.45 * pow((c_q + tau_6 / s2 * (beta_ave + chidelta2) / (s2 - epsdelta2 + chidelta2)), 4.0);   // alpha_1
    real_t a2 = 0.45 * pow((c_q + tau_6 / s3 * (beta_ave + chidelta2) / (s3 - epsdelta2 + chidelta2)), 4.0);   // alpha_2
    real_t a3 = 0.05 * pow((c_q + tau_6 / s64 * (beta_ave + chidelta2) / (s64 - epsdelta2 + chidelta2)), 4.0); // alpha_3
    real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);                                                                    // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
    real_t w0 = a0 * tw1;                                                                                      // omega_0
    real_t w1 = a1 * tw1;                                                                                      // omega_1
    real_t w2 = a2 * tw1;                                                                                      // omega_2
    real_t w3 = a3 * tw1;                                                                                      // omega_3
    // return weighted average
    return (w0 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w1 * (-v2 + 5.0 * v3 + 2.0 * v4) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w3 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}
real_t WENOCU6M2_M(real_t *f, real_t delta)
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
    real_t s64 = fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 2863984.0 * v4 * v6 - 2792660.0 * v3 * v6 + 1325006.0 * v2 * v6 - 245620.0 * v1 * v6 + 3824847.0 * v5 * v5 - 15880404.0 * v4 * v5 + 15929912.0 * v3 * v5 - 7727988.0 * v2 * v5 + 1458762.0 * v1 * v5 + 17195652.0 * v4 * v4 - 35817664.0 * v3 * v4 + 17905032.0 * v2 * v4 - 3462252.0 * v1 * v4 + 19510972.0 * v3 * v3 - 20427884.0 * v2 * v3 + 4086352.0 * v1 * v3 + 5653317.0 * v2 * v2 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1) / 10080.0 + epsdelta2;

    // weights

    real_t beta_ave = (s1 + s3 + 4.0 * s2 - 6.0 * epsdelta2) / 6.0;
    real_t tau_6 = s64 - beta_ave - epsdelta2;
    real_t chidelta2 = 1.0 / epsilon * delta * delta;

    // 	real_t s5 = fabs(s64 - s56) + epsilon; // tau_6 + epsilon
    real_t c_q = 1000.0;                                                                                       // C on page 7242
                                                                                                               // 	real_t q = 4.0;
    real_t a0 = 0.05 * pow((c_q + tau_6 / s1 * (beta_ave + chidelta2) / (s1 - epsdelta2 + chidelta2)), 4.0);   // alpha_0
    real_t a1 = 0.45 * pow((c_q + tau_6 / s2 * (beta_ave + chidelta2) / (s2 - epsdelta2 + chidelta2)), 4.0);   // alpha_1
    real_t a2 = 0.45 * pow((c_q + tau_6 / s3 * (beta_ave + chidelta2) / (s3 - epsdelta2 + chidelta2)), 4.0);   // alpha_2
    real_t a3 = 0.05 * pow((c_q + tau_6 / s64 * (beta_ave + chidelta2) / (s64 - epsdelta2 + chidelta2)), 4.0); // alpha_3
    real_t tw1 = 1.0 / (a0 + a1 + a2 + a3);                                                                    // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
    real_t w0 = a0 * tw1;                                                                                      // omega_0
    real_t w1 = a1 * tw1;                                                                                      // omega_1
    real_t w2 = a2 * tw1;                                                                                      // omega_2
    real_t w3 = a3 * tw1;                                                                                      // omega_3
    // return weighted average
    return (w0 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3) + w1 * (-v2 + 5.0 * v3 + 2.0 * v4) + w2 * (2.0 * v3 + 5.0 * v4 - v5) + w3 * (11.0 * v4 - 7.0 * v5 + 2.0 * v6)) / 6.0;
}

real_t Weno5L2_P(real_t *f, real_t delta, real_t lambda)
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
real_t Weno5L2_M(real_t *f, real_t delta, real_t lambda)
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