#pragma once

#include "global_setup.h"
// =======================================================
// =======================================================

// =======================================================
//    MARCO argus define for Cpi
#define MARCO_HeatCapacity_DEFINE()                                              \
    real_t T = sycl::max(T0, _DF(200.0)); /*T0;*/ /*sycl::max(T0, _DF(200.0));*/ \
    real_t Cpi = _DF(0.0), _T = _DF(1.0) / T;

// #define MARCO_HeatCapacity_DEFINE()                               \
//     real_t T = T0; /*T0;*/ /*sycl::max(T0, _DF(200.0));*/ \
//     real_t Cpi = _DF(0.0), _T = _DF(1.0) / T;

// =======================================================
//    MARCO argus define for hi
#define MARCO_Enthalpy_DEFINE(MARCO_Enthalpy)                                                            \
    real_t hi = _DF(0.0), TT = T0, T = sycl::max(T0, _DF(200.0)); /*TT;*/ /*sycl::max(T0, _DF(200.0));*/ \
    MARCO_Enthalpy;                                                                                      \
    if (TT < _DF(200.0)) /*take low tempreture into consideration*/                                      \
    {                    /*get_hi at T>200*/                                                             \
        real_t Cpi = HeatCapacity(Hia, _DF(200.0), Ri, n);                                               \
        hi += Cpi * (TT - _DF(200.0));                                                                   \
    }

// #define MARCO_Enthalpy_DEFINE(MARCO_Enthalpy)                                             \
//     real_t hi = _DF(0.0), TT = T0, T = TT; /*TT;*/ /*sycl::max(T0, _DF(200.0));*/ \
//     MARCO_Enthalpy;

// =======================================================
//    MARCO of JANAF for Cpi
#define MARCO_HeatCapacity_JANAF()                                                                                                                                                         \
    MARCO_HeatCapacity_DEFINE();                                                                                                                                                           \
    if (T > _DF(1000.0))                                                                                                                                                                   \
        Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + (Hia[n * 7 * 3 + 1 * 3 + 0] + (Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + Hia[n * 7 * 3 + 4 * 3 + 0] * T) * T) * T) * T); \
    else                                                                                                                                                                                   \
        Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + (Hia[n * 7 * 3 + 1 * 3 + 1] + (Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + Hia[n * 7 * 3 + 4 * 3 + 1] * T) * T) * T) * T);

// #define MARCO_HeatCapacity_JANAF()                                                                                                                                                                           \
//     MARCO_HeatCapacity_DEFINE();                                                                                                                                                                             \
//     if (T > _DF(1000.0))                                                                                                                                                                                     \
//         Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + Hia[n * 7 * 3 + 1 * 3 + 0] * T + Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * T * T * T * T); \
//     else                                                                                                                                                                                                     \
//         Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + Hia[n * 7 * 3 + 1 * 3 + 1] * T + Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * T * T * T * T);

// =======================================================
//    MARCO of JANAF for hi
#define MARCO_Enthalpy_JANAF_BODY()                                                                                                                                                                                                                                  \
    if (T > _DF(1000.0))                                                                                                                                                                                                                                             \
        hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 0] + T * (Hia[n * 7 * 3 + 1 * 3 + 0] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 0] * _OT + T * (Hia[n * 7 * 3 + 3 * 3 + 0] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 0] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 0]); \
    else                                                                                                                                                                                                                                                             \
        hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 1] + T * (Hia[n * 7 * 3 + 1 * 3 + 1] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 1] * _OT + T * (Hia[n * 7 * 3 + 3 * 3 + 1] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 1] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 1]);

// #define MARCO_Enthalpy_JANAF_BODY()                                                                                                                                                                                                                                      \
//     if (T > _DF(1000.0)) /*H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T*/                                                                                                                                                                                  \
//         hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 0] + T * (Hia[n * 7 * 3 + 1 * 3 + 0] / _DF(2.0) + T * (Hia[n * 7 * 3 + 2 * 3 + 0] * _OT + T * (Hia[n * 7 * 3 + 3 * 3 + 0] / _DF(4.0) + Hia[n * 7 * 3 + 4 * 3 + 0] * T / _DF(5.0))))) + Hia[n * 7 * 3 + 5 * 3 + 0]); \
//     else                                                                                                                                                                                                                                                                 \
//         hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 1] + T * (Hia[n * 7 * 3 + 1 * 3 + 1] / _DF(2.0) + T * (Hia[n * 7 * 3 + 2 * 3 + 1] * _OT + T * (Hia[n * 7 * 3 + 3 * 3 + 1] / _DF(4.0) + Hia[n * 7 * 3 + 4 * 3 + 1] * T / _DF(5.0))))) + Hia[n * 7 * 3 + 5 * 3 + 1]);

#define MARCO_Enthalpy_JANAF() \
    MARCO_Enthalpy_DEFINE(MARCO_Enthalpy_JANAF_BODY());

// =======================================================
//    MARCO of NASA for Cpi
#define MARCO_HeatCapacity_NASA()                                                                                                                                                                                                                                \
    MARCO_HeatCapacity_DEFINE();                                                                                                                                                                                                                                 \
    if (T >= (_DF(1000.0)) && T < (_DF(6000.0)))                                                                                                                                                                                                                 \
        Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 1] * _T + Hia[n * 7 * 3 + 1 * 3 + 1]) * _T + Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] + (Hia[n * 7 * 3 + 5 * 3 + 1] + Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T); \
    else if (T < (_DF(1000.0)))                                                                                                                                                                                                                                  \
        Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 0] * _T + Hia[n * 7 * 3 + 1 * 3 + 0]) * _T + Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] + (Hia[n * 7 * 3 + 5 * 3 + 0] + Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T); \
    else if (T >= _DF(6000.0))                                                                                                                                                                                                                                   \
        Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 2] * _T + Hia[n * 7 * 3 + 1 * 3 + 2]) * _T + Hia[n * 7 * 3 + 2 * 3 + 2] + (Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] + (Hia[n * 7 * 3 + 5 * 3 + 2] + Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T);

// =======================================================
//    MARCO of NASA for hi
#define MARCO_Enthalpy_NASA_BODY()                                                                                                                                                                                                                                                                                                                 \
    if (T >= _DF(1000.0) && T < _DF(6000.0))                                                                                                                                                                                                                                                                                                       \
        hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 1] / T + Hia[n * 7 * 3 + 1 * 3 + 1] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 1] + (_DF(0.5) * Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] * _OT + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 1] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 1]); \
    else if (T < _DF(1000.0))                                                                                                                                                                                                                                                                                                                      \
        hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 0] / T + Hia[n * 7 * 3 + 1 * 3 + 0] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 0] + (_DF(0.5) * Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] * _OT + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 0] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 0]); \
    else if (T >= _DF(6000.0))                                                                                                                                                                                                                                                                                                                     \
        hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 2] / T + Hia[n * 7 * 3 + 1 * 3 + 2] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 2] + (_DF(0.5) * Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] * _OT + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 2] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 2]);

#define MARCO_Enthalpy_NASA() \
    MARCO_Enthalpy_DEFINE(MARCO_Enthalpy_NASA_BODY());
