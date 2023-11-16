#pragma once

#include "global_setup.h"
// =======================================================
// repeated code definitions
// =======================================================

// =======================================================
//    Global sycL_reduction
#if defined(DEFINED_ONEAPI)
#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<>())
#else
#define sycl_reduction_plus(argus) sycl::reduction(&(argus), sycl::plus<real_t>())
#define sycl_reduction_max(argus) sycl::reduction(&(argus), sycl::maximum<real_t>())
#define sycl_reduction_min(argus) sycl::reduction(&(argus), sycl::minimum<real_t>())
#endif

// =======================================================
//    Set Domain size
#define MARCO_DOMAIN()        \
    int Xmax = bl.Xmax;       \
    int Ymax = bl.Ymax;       \
    int Zmax = bl.Zmax;       \
    int X_inner = bl.X_inner; \
    int Y_inner = bl.Y_inner; \
    int Z_inner = bl.Z_inner;

#define MARCO_DOMAIN_GHOST()    \
    int Xmax = bl.Xmax;         \
    int Ymax = bl.Ymax;         \
    int Zmax = bl.Zmax;         \
    int X_inner = bl.X_inner;   \
    int Y_inner = bl.Y_inner;   \
    int Z_inner = bl.Z_inner;   \
    int Bwidth_X = bl.Bwidth_X; \
    int Bwidth_Y = bl.Bwidth_Y; \
    int Bwidth_Z = bl.Bwidth_Z;

// =======================================================
//    get Roe values insde Reconstructflux
#define MARCO_ROE()                                       \
    real_t D = sycl::sqrt(rho[id_r] / rho[id_l]); \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));                \
    real_t _u = (u[id_l] + D * u[id_r]) * D1;             \
    real_t _v = (v[id_l] + D * v[id_r]) * D1;             \
    real_t _w = (w[id_l] + D * w[id_r]) * D1;             \
    real_t _H = (H[id_l] + D * H[id_r]) * D1;             \
    real_t _P = (p[id_l] + D * p[id_r]) * D1;             \
    real_t _rho = sycl::sqrt(rho[id_r] * rho[id_l]);

#ifdef ESTIM_NAN
#define MARCO_ERROR_OUT()                   \
    eb1[id_l] = b1;                         \
    eb3[id_l] = b3;                         \
    ec2[id_l] = c2;                         \
    for (size_t nn = 0; nn < NUM_COP; nn++) \
    {                                       \
        ezi[nn + NUM_COP * id_l] = z[nn];   \
    }
#else
#define MARCO_ERROR_OUT() ;
#endif

// =======================================================
//    get c2 #ifdef COP inside Reconstructflux
#define MARCO_COPC2()                                                                                                                                                                                                                              \
    real_t _yi[MAX_SPECIES], hi_l[MAX_SPECIES], hi_r[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0); /*yi_l[NUM_SPECIES], yi_r[NUM_SPECIES],_hi[NUM_SPECIES],*/                                                           \
    for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                                                                                                                       \
    {                                                                                                                                                                                                                                              \
        hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);                                                                                                                                                               \
        hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);                                                                                                                                                               \
    }                                                                                                                                                                                                                                              \
    real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]); /*get_yi(y, yi_l, id_l);*/ /*get_yi(y, yi_r, id_r);*/                                                                                                               \
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                              \
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                                                                                                                                  \
        /*_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;*/                                                                                                                                                                                              \
    }                                                                                                                                                                                                                                              \
    real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];                                                                                                                                                                                                      \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                                                                                                                         \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                                                                                                                         \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                                                                                                                       \
    real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];                                                                                                                                                                       \
    real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];                                                                                                                                                                       \
    real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l];                                                                                                                                                                                  \
    real_t e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];                                                                                                                                                                                  \
    real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                                                                                                                               \
    real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                                                                                                                               \
    real_t R_l = get_CopR(thermal._Wi, yi_l);                                                                                                                                                                                                      \
    real_t R_r = get_CopR(thermal._Wi, yi_r);                                                                                                                                                                                                      \
    real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], thermal.Ri[NUM_COP], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),                                                                                                                        \
                                    get_DpDrho(hi_r[NUM_COP], thermal.Ri[NUM_COP], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);                                                                                                                \
    for (size_t nn = 0; nn < NUM_COP; nn++)                                                                                                                                                                                                        \
    {                                                                                                                                                                                                                                              \
        _dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal.Ri[nn], hi_l[NUM_COP], thermal.Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),                                                                                                      \
                                      get_DpDrhoi(hi_r[nn], thermal.Ri[nn], hi_r[NUM_COP], thermal.Ri[NUM_COP], T[id_r], Cp_r, R_r, gamma_r), D, D1);                                                                                              \
        drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                              \
    real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * ((u[id_r] - u[id_l]) * (u[id_r] - u[id_l]) + (v[id_r] - v[id_l]) * (v[id_r] - v[id_l]) + (w[id_r] - w[id_l]) * (w[id_r] - w[id_l])); \
    real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);                                                                                                                                                                  \
    real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);                                                                                                                                      \
    real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);                                                                                  \
    /*add support while c2<0 use c2 Refed in https://doi.org/10.1006/jcph.1996.5622 */                                                                                                                                                             \
    real_t c2w = sycl::step(c2, _DF(0.0)); /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/                                                                                                                                        \
    c2 = Gamma0 * _P * _rho * c2w + (_DF(1.0) - c2w) * c2;                                                                                                                                                                                         \
    MARCO_ERROR_OUT();

// =======================================================
//    get c2 #ifdef COP inside Reconstructflux
#define MARCO_COPC2_ROB()                                                                                                                       \
    real_t _yi[NUM_SPECIES], Ri[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], b1 = _DF(0.0), b3 = _DF(0.0); \
    for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                    \
    {                                                                                                                                           \
        hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);                                                            \
        hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);                                                            \
        Ri[n] = Ru * thermal._Wi[n];                                                                                                            \
    }                                                                                                                                           \
    real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]);                                                                  \
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                 \
    {                                                                                                                                           \
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                               \
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;                                                                                               \
    }                                                                                                                                           \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                      \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                      \
    real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                            \
    real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                            \
    real_t R_l = get_CopR(thermal._Wi, yi_l);                                                                                                   \
    real_t R_r = get_CopR(thermal._Wi, yi_r);                                                                                                   \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                    \
    real_t _R = get_RoeAverage(R_l, R_r, D, D1);                                                                                                \
    real_t _Cp = get_RoeAverage(Cp_l, Cp_r, D, D1);                                                                                             \
    real_t _T = get_RoeAverage(T[id_l], T[id_r], D, D1);                                                                                        \
    real_t c2 = get_CopC2(z, b1, b3, Ri, _yi, _hi, Gamma0, _R, _Cp, _T);                                                                        \
    MARCO_ERROR_OUT();

// =======================================================
//    get c2 #else COP
// #define MARCO_NOCOPC2()                                                                                                  \
//     real_t yi_l[NUM_SPECIES] = {_DF(1.0)}, yi_r[NUM_SPECIES] = {_DF(1.0)}, _yi[] = {_DF(1.0)}, b3 = _DF(0.0), z[] = {0}; \
//     real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                               \
//     real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                               \
//     real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                             \
//     real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */       \
//     real_t b1 = (Gamma0 - _DF(1.0)) / c2;

#define MARCO_NOCOPC2()                                                                                            \
    real_t _yi[NUM_SPECIES] = {_DF(1.0)}, b3 = _DF(0.0), z[] = {_DF(0.0)};                                         \
    real_t Gamma0 = NCOP_Gamma;                                                                                    \
    real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */ \
    real_t b1 = (Gamma0 - _DF(1.0)) / c2;

#ifdef COP
#define MARCO_GETC2() MARCO_COPC2()
// MARCO_COPC2() //MARCO_COPC2_ROB()
#else
#define MARCO_GETC2() MARCO_NOCOPC2()
#endif // end COP

// =======================================================
//    Pre get eigen_martix
#define MARCO_PREEIGEN()                      \
    real_t q2 = _u * _u + _v * _v + _w * _w;  \
    real_t _c = sycl::sqrt(c2);       \
    real_t b2 = _DF(1.0) + b1 * q2 - b1 * _H; \
    real_t _c1 = _DF(1.0) / _c;

// =======================================================
//    Loop in Output
#define MARCO_OUTLOOP                                        \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)                \
        for (int j = VTI.minY; j < VTI.maxY; j++)            \
            for (int i = VTI.minX; i < VTI.maxX; i++)

#define MARCO_POUTLOOP(BODY)                          \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)         \
        for (int j = VTI.minY; j < VTI.maxY; j++)     \
        {                                             \
            for (int i = VTI.minX; i < VTI.maxX; i++) \
                out << BODY << " ";                   \
            out << "\n";                              \
        }

#define MARCO_COUTLOOP                                       \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = minZ; k < maxZ; k++)                        \
        for (int j = minY; j < maxY; j++)                    \
            for (int i = minX; i < maxX; i++)

// =======================================================
// end repeated code definitions
// =======================================================
