#pragma once
// =======================================================
// repeated code definitions
// =======================================================
/**
 * set Domain size
 */
#define MARCO_DOMAIN()        \
    int Xmax = bl.Xmax;       \
    int Ymax = bl.Ymax;       \
    int Zmax = bl.Zmax;       \
    int X_inner = bl.X_inner; \
    int Y_inner = bl.Y_inner; \
    int Z_inner = bl.Z_inner;
/**
 * set Domain size
 */
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
/**
 * get Roe values insde Reconstructflux
 */
#define MARCO_ROE()                           \
    real_t D = sqrt(rho[id_r] / rho[id_l]);   \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));    \
    real_t _u = (u[id_l] + D * u[id_r]) * D1; \
    real_t _v = (v[id_l] + D * v[id_r]) * D1; \
    real_t _w = (w[id_l] + D * w[id_r]) * D1; \
    real_t _H = (H[id_l] + D * H[id_r]) * D1; \
    real_t _P = (p[id_l] + D * p[id_r]) * D1; \
    real_t _rho = sqrt(rho[id_r] * rho[id_l]);

/**
 * get c2 #ifdef COP inside Reconstructflux
 */
#define MARCO_COPC2()                                                                                                                                   \
    real_t _T = (T[id_l] + D * T[id_r]) * D1;                                                                                                           \
    real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], Ri[NUM_SPECIES]; \
    for (size_t i = 0; i < NUM_SPECIES; i++)                                                                                                            \
    {                                                                                                                                                   \
        Ri[i] = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);                                                                                         \
        hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri[i], i);                                                                          \
        hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri[i], i);                                                                          \
    }                                                                                                                                                   \
    get_yi(y, yi_l, id_l);                                                                                                                              \
    get_yi(y, yi_r, id_r);                                                                                                                              \
    real_t _h = _DF(0.0);                                                                                                                               \
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                         \
    {                                                                                                                                                   \
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                                       \
        _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;                                                                                                       \
        _h += _hi[ii] * _yi[ii];                                                                                                                        \
    }                                                                                                                                                   \
    real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];                                                                                                           \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                              \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                              \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                            \
    real_t e_l = H[id_l] - _DF(0.5) * (u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l]) - p[id_l] / rho[id_l];                                \
    real_t e_r = H[id_r] - _DF(0.5) * (u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r]) - p[id_r] / rho[id_r];                                \
    real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], Ri[NUM_COP], T[id_l], e_l, gamma_l),                                                      \
                                    get_DpDrho(hi_r[NUM_COP], Ri[NUM_COP], T[id_r], e_r, gamma_r), D, D1);                                              \
    real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                                    \
    real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                                    \
    real_t R_l = get_CopR(thermal->species_chara, yi_l);                                                                                                \
    real_t R_r = get_CopR(thermal->species_chara, yi_r);                                                                                                \
    for (size_t i = 0; i < NUM_COP; i++)                                                                                                                \
    {                                                                                                                                                   \
        drhoi[i] = rho[id_r] * yi_r[i] - rho[id_l] * yi_l[i];                                                                                           \
        _dpdrhoi[i] = get_RoeAverage(get_DpDrhoi(hi_l[i], Ri[i], hi_l[NUM_COP], Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),                              \
                                     get_DpDrhoi(hi_l[i], Ri[i], hi_l[NUM_COP], Ri[NUM_COP], T[id_l], Cp_r, R_r, gamma_r), D, D1);                      \
    }                                                                                                                                                   \
    real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1);                                                                     \
    real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);                                                                       \
    real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);                                           \
    real_t c2 = SoundSpeedMultiSpecies(z, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);
/* TODO: real_t c2 = get_CopC2(z, thermal->Ri, _yi, _hi, Gamma0, _h, _T); z[NUM_SPECIES] 是一个在该函数中同时计算的数组变量 */

// real_t _T = (T[id_l] + D * T[id_r]) * D1;
// real_t _yi[NUM_SPECIES], yi_l[NUM_SPECIES], yi_r[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], Ri[NUM_SPECIES];
// for (size_t i = 0; i < NUM_SPECIES; i++)
// {
//     Ri[i] = Ru / (thermal->species_chara[i * SPCH_Sz + 6]);
//     hi_l[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_l], Ri[i], i);
//     hi_r[i] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id_r], Ri[i], i);
// }
// get_yi(y, yi_l, id_l);
// get_yi(y, yi_r, id_r);
// real_t _h = _DF(0.0);
// for (size_t ii = 0; ii < NUM_SPECIES; ii++)
// {
//     _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;
//     _hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;
//     _h += _hi[ii] * _yi[ii];
// }
// // real_t Gamma0 = get_CopGamma(thermal, _yi, _T);              // out from RoeAverage_x , 使用半点的数据计算出半点处的Gamma
// real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];
// real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);
// real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);
// real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);
// real_t e_l = H[id_l] - _DF(0.5) * (u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l]) - p[id_l] / rho[id_l];
// real_t e_r = H[id_r] - _DF(0.5) * (u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r]) - p[id_r] / rho[id_r];
// real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], Ri[NUM_COP], T[id_l], e_l, gamma_l),
//                                 get_DpDrho(hi_r[NUM_COP], Ri[NUM_COP], T[id_r], e_r, gamma_r), D, D1);
// real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);
// real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);
// real_t R_l = get_CopR(thermal->species_chara, yi_l);
// real_t R_r = get_CopR(thermal->species_chara, yi_r);
// for (size_t i = 0; i < NUM_COP; i++)
// {
//     drhoi[i] = rho[id_r] * yi_r[i] - rho[id_l] * yi_l[i];
//     _dpdrhoi[i] = get_RoeAverage(get_DpDrhoi(hi_l[i], Ri[i], hi_l[NUM_COP], Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),
//                                  get_DpDrhoi(hi_l[i], Ri[i], hi_l[NUM_COP], Ri[NUM_COP], T[id_l], Cp_r, R_r, gamma_r), D, D1);
// }
// real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1);
// real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);
// real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);
// real_t c2 = SoundSpeedMultiSpecies(z, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);
// TODO: real_t c2 = get_CopC2(z, thermal->Ri, _yi, _hi, Gamma0, _h, _T); // z[NUM_SPECIES] 是一个在该函数中同时计算的数组变量

/**
 * get c2 #else COP
 */
#define MARCO_NOCOPC2()                                                                                            \
    real_t yi_l[NUM_SPECIES] = {_DF(1.0)}, yi_r[NUM_SPECIES] = {_DF(1.0)};                                         \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                         \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                         \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                       \
    real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */ \
    real_t z[] = {0}, _yi[] = {1};
// real_t yi_l[NUM_SPECIES] = {_DF(1.0)}, yi_r[NUM_SPECIES] = {_DF(1.0)};
// real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);
// real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);
// real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);
// real_t c2 = Gamma0 * _P / _rho; //(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x
// real_t z[] = {0}, _yi[] = {1};

/**
 * prepare for getting viscous flux
 */
#define MARCO_PREVISCFLUX()                                                                                                                               \
    real_t F_wall_v[Emax], f_x, f_y, f_z, u_hlf, v_hlf, w_hlf;                                                                                            \
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0); /*mue at wall*/ \
    real_t lamada = -_DF(2.0) / _DF(3.0) * mue;

/**
 * get viscous flux
 */
#ifdef COP
const bool _COP = true;
#else
const bool _COP = false;
#endif
#ifdef Heat
const bool _Heat = true;
#else
const bool _Heat = false;
#endif
#ifdef Diffu
const bool _Diffu = true;
#else
const bool _Diffu = false;
#endif
#define MARCO_VISCFLUX()                                                                                                                                                                               \
    F_wall_v[0] = _DF(0.0);                                                                                                                                                                            \
    F_wall_v[1] = f_x;                                                                                                                                                                                 \
    F_wall_v[2] = f_y;                                                                                                                                                                                 \
    F_wall_v[3] = f_z;                                                                                                                                                                                 \
    F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;                                                                                                                                             \
    if (_Heat) /* Fourier thermal conductivity*/                                                                                                                                                       \
    {                                                                                                                                                                                                  \
        real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); /* thermal conductivity at wall*/ \
        kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dl / _DF(24.0);                                                                             /* temperature gradient at wall*/ \
        F_wall_v[4] += kk;                                                                                                                                                                             \
    }                                                                                                                                                                                                  \
    if (_Diffu) /* energy fiffusion depends on mass diffusion*/                                                                                                                                        \
    {                                                                                                                                                                                                  \
        real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);                                                                                                 \
        real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];                                                                                               \
        for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                          \
        {                                                                                                                                                                                              \
            hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);                              \
            Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);     \
            if (_COP)                                                                                                                                                                                  \
            {                                                                                                                                                                                          \
                Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);                                                                                      \
                Yil_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dl / _DF(24.0); /* temperature gradient at wall*/                                             \
            }                                                                                                                                                                                          \
            else                                                                                                                                                                                       \
            {                                                                                                                                                                                          \
                Yil_wall[l] = _DF(0.0);                                                                                                                                                                \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
        if (_Heat)                                                                                                                                                                                     \
            for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                      \
            {                                                                                                                                                                                          \
                F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];                                                                                                                      \
            }                                                                                                                                                                                          \
        if (_COP) /* visc flux for cop equations*/                                                                                                                                                     \
        {                                                                                                                                                                                              \
            real_t CorrectTerm = _DF(0.0);                                                                                                                                                             \
            for (int l = 0; l < NUM_SPECIES; l++)                                                                                                                                                      \
            {                                                                                                                                                                                          \
                CorrectTerm += Dim_wall[l] * Yil_wall[l];                                                                                                                                              \
            }                                                                                                                                                                                          \
            CorrectTerm *= rho_wall;                                                                                                                                                                   \
            for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                                                                                                                     \
            {                                                                                                                                                                                          \
                F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;                                                                                             \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
    }                                                                                                                                                                                                  \
    for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/                                                                                                                                \
    {                                                                                                                                                                                                  \
        Flux_wall[n + Emax * id] -= F_wall_v[n];                                                                                                                                                       \
    }

/**
 * Pre get eigen_martix
 */
#define MARCO_PREEIGEN()                               \
    real_t q2 = _u * _u + _v * _v + _w * _w;           \
    real_t _c = sqrt(c2);                              \
    real_t b1 = (Gamma - _DF(1.0)) / c2;               \
    real_t b2 = _DF(1.0) + b1 * q2 - b1 * _H;          \
    real_t b3 = _DF(0.0);                              \
    for (size_t i = 0; i < NUM_COP; i++)               \
    {                                                  \
        b3 += yi[i] * z[i]; /* NOTE: related with yi*/ \
    }                                                  \
    b3 *= b1;                                          \
    real_t _c1 = _DF(1.0) / _c;
// =======================================================
// end repeated code definitions
// =======================================================
