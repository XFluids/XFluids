#pragma once
// =======================================================
// repeated code definitions
// =======================================================
#if 1 == Artificial_type // ROE
#define Roe_type _DF(1.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(0.0)
#elif 2 == Artificial_type // LLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(1.0)
#define GLF_type _DF(0.0)
#elif 3 == Artificial_type // GLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(1.0)
#endif
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
#define MARCO_ROE()                                       \
    real_t D = sycl::sqrt<real_t>(rho[id_r] / rho[id_l]); \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));                \
    real_t _u = (u[id_l] + D * u[id_r]) * D1;             \
    real_t _v = (v[id_l] + D * v[id_r]) * D1;             \
    real_t _w = (w[id_l] + D * w[id_r]) * D1;             \
    real_t _H = (H[id_l] + D * H[id_r]) * D1;             \
    real_t _P = (p[id_l] + D * p[id_r]) * D1;             \
    real_t _rho = sycl::sqrt<real_t>(rho[id_r] * rho[id_l]);

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

/**
 * get c2 #ifdef COP inside Reconstructflux
 */
#define MARCO_COPC2()                                                                                                                                                                                                                        \
    real_t _yi[NUM_SPECIES], /*yi_l[NUM_SPECIES], yi_r[NUM_SPECIES],_hi[NUM_SPECIES],*/ hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], b1 = _DF(0.0), b3 = _DF(0.0);                                                                      \
    for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                        \
        hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);                                                                                                                                                         \
        hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);                                                                                                                                                         \
    }                                                                                                                                                                                                                                        \
    real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]); /*get_yi(y, yi_l, id_l);*/ /*get_yi(y, yi_r, id_r);*/                                                                                                         \
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                                                                                                              \
    {                                                                                                                                                                                                                                        \
        _yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                                                                                                                            \
        /*_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;*/                                                                                                                                                                                        \
    }                                                                                                                                                                                                                                        \
    real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];                                                                                                                                                                                                \
    real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                                                                                                                   \
    real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                                                                                                                   \
    real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                                                                                                                 \
    real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];                                                                                                                                                                 \
    real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];                                                                                                                                                                 \
    real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l];                                                                                                                                                                            \
    real_t e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];                                                                                                                                                                            \
    real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                                                                                                                         \
    real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                                                                                                                         \
    real_t R_l = get_CopR(thermal._Wi, yi_l);                                                                                                                                                                                                \
    real_t R_r = get_CopR(thermal._Wi, yi_r);                                                                                                                                                                                                \
    real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], thermal.Ri[NUM_COP], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),                                                                                                                  \
                                    get_DpDrho(hi_r[NUM_COP], thermal.Ri[NUM_COP], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);                                                                                                          \
    for (size_t nn = 0; nn < NUM_COP; nn++)                                                                                                                                                                                                  \
    {                                                                                                                                                                                                                                        \
        _dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal.Ri[nn], hi_l[NUM_COP], thermal.Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),                                                                                                \
                                      get_DpDrhoi(hi_r[nn], thermal.Ri[nn], hi_r[NUM_COP], thermal.Ri[NUM_COP], T[id_r], Cp_r, R_r, gamma_r), D, D1);                                                                                        \
        drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];                                                                                                                                                                             \
    }                                                                                                                                                                                                                                        \
    real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * (sycl::pow<real_t>(u[id_r] - u[id_l], 2) + sycl::pow<real_t>(v[id_r] - v[id_l], 2) + sycl::pow<real_t>(w[id_r] - w[id_l], 2)); \
    real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);                                                                                                                                                            \
    real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);                                                                                                                                \
    real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);                                                                            \
    /*add support while c2<0 use c2 Refed in https://doi.org/10.1006/jcph.1996.5622 */                                                                                                                                                       \
    real_t c2w = sycl::step(c2, _DF(0.0)); /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/                                                                                                                                  \
    c2 = Gamma0 * _P * _rho * c2w + (_DF(1.0) - c2w) * c2;                                                                                                                                                                                   \
    MARCO_ERROR_OUT();

/**
 * get c2 #ifdef COP inside Reconstructflux
 */
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

/**                                                                                                                                                                                                                                          \
 * get c2 #else COP                                                                                                                                                                                                                          \
 */
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
#define MARCO_GETC2() MARCO_COPC2();
// MARCO_COPC2();//MARCO_COPC2_ROB();
#else
#define MARCO_GETC2() MARCO_NOCOPC2();
#endif // end COP

/**
 * Caculate flux_wall
 */
#if 0 == EIGEN_ALLOC
// RoeAverage_Left and RoeAverage_Right for each DIR
#define MARCO_ROEAVERAGE_LEFTX \
    RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTY \
    RoeAverageLeft_y(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTZ \
    RoeAverageLeft_z(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

#define MARCO_ROEAVERAGE_RIGHTX \
    RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTY \
    RoeAverageRight_y(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTZ \
    RoeAverageRight_z(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

// device function ptr
#define MARCO_ROEAVERAGE_LEFTNX \
    RoeAverageLeftX[n](n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTNY \
    RoeAverageLeftY[n](n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTNZ \
    RoeAverageLeftZ[n](n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

#define MARCO_ROEAVERAGE_RIGHTNX \
    RoeAverageRightX[n](n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTNY \
    RoeAverageRightY[n](n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTNZ \
    RoeAverageRightZ[n](n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

// WENO 7 // used by MARCO_FLUXWALL_WENO7(i + m, j, k, i + m - stencil_P, j, k); in x
#define MARCO_FLUXWALL_WENO7(MARCO_ROE_LEFT, MARCO_ROE_RIGHT, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                               \
    real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value, artificial_viscosity;                                                           \
    for (int n = 0; n < Emax; n++)                                                                                                                                              \
    {                                                                                                                                                                           \
        real_t eigen_local_max = _DF(0.0);                                                                                                                                      \
        MARCO_ROE_LEFT; /* eigen_r actually */ /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */                  \
        eigen_local_max = eigen_value;                                                                                                                                          \
        real_t lambda_l = eigen_local[Emax * id_l + n];                                                                                                                         \
        real_t lambda_r = eigen_local[Emax * id_r + n];                                                                                                                         \
        if (lambda_l * lambda_r < 0.0)                                                                                                                                          \
        {                                                                                                                                                                       \
            for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                                                         \
            { /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */                                                                                    \
                int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                                                                 \
                eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/                                 \
            }                                                                                                                                                                   \
        }                                                                                                                                                                       \
        artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                                                                 \
        for (size_t m = 0; m < stencil_size; m++)                                                                                                                               \
        { /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */                                                                            \
            int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                                                                     \
            uf[m] = _DF(0.0);                                                                                                                                                   \
            ff[m] = _DF(0.0);                                                                                                                                                   \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                   \
            {                                                                                                                                                                   \
                uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];                                                                                                      \
                ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];                                                                                                      \
            } /* for local speed*/                                                                                                                                              \
            pp[m] = _DF(0.5) * (ff[m] + artificial_viscosity * uf[m]);                                                                                                          \
            mm[m] = _DF(0.5) * (ff[m] - artificial_viscosity * uf[m]);                                                                                                          \
        }                                                                   /* calculate the scalar numerical flux at x direction*/                                             \
        f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/                                                                                         \
        MARCO_ROE_RIGHT; /* eigen_r actually */                             /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */ \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                       \
        {                                                                                                                                                                       \
            _p[n][n1] = f_flux * eigen_lr[n1];                                                                                                                                  \
        }                                                                                                                                                                       \
    }                                                                                                                                                                           \
    for (int n = 0; n < Emax; n++)                                                                                                                                              \
    { /* reconstruction the F-flux terms*/                                                                                                                                      \
        real_t fluxl = _DF(0.0);                                                                                                                                                \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                       \
        {                                                                                                                                                                       \
            fluxl += _p[n1][n];                                                                                                                                                 \
        }                                                                                                                                                                       \
        Fwall[Emax * id_l + n] = fluxl;                                                                                                                                         \
    }

#define MARCO_WENO5 weno5old_GPU(&pp[3], &mm[3])
#define MARCO_WENOCU6 WENOCU6_GPU(&pp[3], &mm[3], dl)
#if SCHEME_ORDER == 5
#define WENO_GPU MARCO_WENO5
#elif SCHEME_ORDER == 6
#define WENO_GPU MARCO_WENOCU6
#endif 

// WENO 5 //used by: MARCO_FLUXWALL_WENO5(i + m, j, k, i + m, j, k);
#define MARCO_FLUXWALL_WENO5(MARCO_ROE_LEFT, MARCO_ROE_RIGHT, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                                                                                                                                                                                                                       \
    real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value, artificial_viscosity;                                                                                                                                                                                                                                                   \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                                                                                                                                                                                      \
    {                                                                                                                                                                                                                                                                                                                                                                   \
        real_t eigen_local_max = _DF(0.0);                                                                                                                                                                                                                                                                                                                              \
        MARCO_ROE_LEFT; /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */                                                                                                                                                                                                                                 \
        for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                                                                                                                                                                                                                                                     \
        { /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/                                                                                                                                                                                                                                                                                  \
            int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                                                                                                                                                                                                                                                             \
            eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                                                                                                               \
        artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                                                                                                                                                                                                                                                         \
        for (int m = -3; m <= 4; m++)                                                                                                                                                                                                                                                                                                                                   \
        {                                                                                                                                                                                                                                                                                                                                                               \
            /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i; 3rd oder and can be modified */                                                                                                                                                                                                                                                \
            int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                                                                                                                                                                                                                                                               \
            uf[m + 3] = _DF(0.0);                                                                                                                                                                                                                                                                                                                                       \
            ff[m + 3] = _DF(0.0);                                                                                                                                                                                                                                                                                                                                       \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                                                                                                                                                                                           \
            {                                                                                                                                                                                                                                                                                                                                                           \
                uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; /* eigen_l actually */                                                                                                                                                                                                                                                                 \
                ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];                                                                                                                                                                                                                                                                                        \
            } /*  for local speed*/                                                                                                                                                                                                                                                                                                                                     \
            pp[m + 3] = _DF(0.5) * (ff[m + 3] + artificial_viscosity * uf[m + 3]);                                                                                                                                                                                                                                                                                      \
            mm[m + 3] = _DF(0.5) * (ff[m + 3] - artificial_viscosity * uf[m + 3]);                                                                                                                                                                                                                                                                                      \
        }                                                                                                                                                                                                                                                           /* calculate the scalar numerical flux at x direction*/                                             \
        f_flux = WENO_GPU; /* WENOCU6_GPU(&pp[3], &mm[3], dl) WENO_GPU WENOCU6_P(&pp[3], dl) + WENOCU6_P(&mm[3], dl);*/ /*(weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)) / _DF(6.0);*/ /* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/ /* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/                                              \
        MARCO_ROE_RIGHT;                                                                                                                                                                                                                                            /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */ \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                                                                                                                                                                                               \
        {                                      /* get Fp */                                                                                                                                                                                                                                                                                                             \
            _p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */                                                                                                                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                                                                                                                                                               \
    }                                                                                                                                                                                                                                                                                                                                                                   \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                                                                                                                                                                                      \
    { /* reconstruction the F-flux terms*/                                                                                                                                                                                                                                                                                                                              \
        real_t fluxl = _DF(0.0);                                                                                                                                                                                                                                                                                                                                        \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                               \
            fluxl += _p[n1][n];                                                                                                                                                                                                                                                                                                                                         \
        }                                                                                                                                                                                                                                                                                                                                                               \
        Fwall[Emax * id_l + n] = fluxl;                                                                                                                                                                                                                                                                                                                                 \
    }
#endif // end EIGEN_ALLOC

#if 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
// RoeAverage for each DIR
#define MARCO_ROEAVERAGE_X \
    RoeAverage_x(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_Y \
    RoeAverage_y(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_Z \
    RoeAverage_z(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

// WENO 7 // used by MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_X, i + m, j, k, i + m - stencil_P, j, k); in x
#define MARCO_FLUXWALL_WENO7(ROE_AVERAGE, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                   \
    ROE_AVERAGE;                                                                                                                                \
    real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux;                                                                              \
    for (int n = 0; n < Emax; n++)                                                                                                              \
    {                                                                                                                                           \
        real_t eigen_local_max = _DF(0.0);                                                                                                      \
        eigen_local_max = eigen_value[n];                                                                                                       \
        real_t lambda_l = eigen_local[Emax * id_l + n];                                                                                         \
        real_t lambda_r = eigen_local[Emax * id_r + n];                                                                                         \
        if (lambda_l * lambda_r < 0.0)                                                                                                          \
        {                                                                                                                                       \
            for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                         \
            { /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */                                                    \
                int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                                 \
                eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/ \
            }                                                                                                                                   \
        }                                                                                                                                       \
        for (size_t m = 0; m < stencil_size; m++)                                                                                               \
        { /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */                                            \
            int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                                     \
            uf[m] = _DF(0.0);                                                                                                                   \
            ff[m] = _DF(0.0);                                                                                                                   \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                   \
            {                                                                                                                                   \
                uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                                    \
                ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                                    \
            }                                                                                                                                   \
            /* for local speed*/                                                                                                                \
            pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);                                                                               \
            mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);                                                                               \
        }                                                                                                                                       \
        /* calculate the scalar numerical flux at x direction*/                                                                                 \
        f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl);                                                                     \
        /* get Fp*/                                                                                                                             \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                       \
        {                                                                                                                                       \
            _p[n][n1] = f_flux * eigen_r[n1][n];                                                                                                \
        }                                                                                                                                       \
    }                                                                                                                                           \
    for (int n = 0; n < Emax; n++)                                                                                                              \
    { /* reconstruction the F-flux terms*/                                                                                                      \
        real_t fluxl = _DF(0.0);                                                                                                                \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                       \
        {                                                                                                                                       \
            fluxl += _p[n1][n];                                                                                                                 \
        }                                                                                                                                       \
        Fwall[Emax * id_l + n] = fluxl;                                                                                                         \
    }

// WENO 5 //used by: MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_X, i + m, j, k, i + m, j, k);
#define MARCO_FLUXWALL_WENO5(ROE_AVERAGE, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                                                                                         \
    ROE_AVERAGE;                                                                                                                                                                                      \
    real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax];                                                                                                                                    \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                    \
    {                                                                                                                                                                                                 \
        real_t eigen_local_max = _DF(0.0);                                                                                                                                                            \
        for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                                                                                   \
        { /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/                                                                                                                \
            int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                                                                                           \
            eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/                                                           \
        }                                                                                                                                                                                             \
        for (int m = -3; m <= 4; m++)                                                                                                                                                                 \
        { /* int _i_2 = i + m, _j_2 = j, _k_2 = k; 3rd oder and can be modified*/ /* 3rd oder and can be modified*/                                                                                   \
            int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                                                                                             \
            uf[m + 3] = _DF(0.0);                                                                                                                                                                     \
            ff[m + 3] = _DF(0.0);                                                                                                                                                                     \
            for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                         \
            {                                                                                                                                                                                         \
                uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];                                                                                                                    \
                ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_l[n][n1];                                                                                                                    \
            } /*  for local speed*/                                                                                                                                                                   \
            pp[m + 3] = _DF(0.5) * (ff[m + 3] + eigen_local_max * uf[m + 3]);                                                                                                                         \
            mm[m + 3] = _DF(0.5) * (ff[m + 3] - eigen_local_max * uf[m + 3]);                                                                                                                         \
        }                                                                                                                                     /* calculate the scalar numerical flux at x direction*/ \
        f_flux = (weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)); /* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/ /* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/  \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                             \
        { /* get Fp*/                                                                                                                                                                                 \
            _p[n][n1] = f_flux * eigen_r[n1][n];                                                                                                                                                      \
        }                                                                                                                                                                                             \
    }                                                                                                                                                                                                 \
    for (int n = 0; n < Emax; n++)                                                                                                                                                                    \
    { /* reconstruction the F-flux terms*/                                                                                                                                                            \
        real_t fluxl = _DF(0.0);                                                                                                                                                                      \
        for (int n1 = 0; n1 < Emax; n1++)                                                                                                                                                             \
        {                                                                                                                                                                                             \
            fluxl += _p[n1][n];                                                                                                                                                                       \
        }                                                                                                                                                                                             \
        Fwall[Emax * id_l + n] = fluxl;                                                                                                                                                               \
    }
#endif // end EIGEN_ALLOC

/**
 * prepare for getting viscous flux new
 */
#define MARCO_PREVISCFLUX()                                                                                                                               \
    real_t F_wall_v[Emax], f_x, f_y, f_z, u_hlf, v_hlf, w_hlf;                                                                                            \
    real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0); /*mue at wall*/ \
    real_t lamada = -_DF(2.0) * _OT * mue;

const real_t _sxtn = _DF(1.0) / _DF(16.0);
const real_t _twfr = _DF(1.0) / _DF(24.0);
const real_t _twle = _DF(1.0) / _DF(12.0);

#ifdef COP
#define MARCO_VIS_COP_IN_DIFFU1()                                                                                                       \
    Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/ \
    Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;                                           \
    CorrectTerm += Dim_wall[l] * Yil_wall[l];
/**
 * NOTE: CorrectTerm for diffusion to Average the error from the last species to
 *  all species according to the mass fraction
 */
#else
#define MARCO_VIS_COP_IN_DIFFU1() Yil_wall[l] = _DF(0.0);
#endif // COP

// for error out of Vis
#ifdef ESTIM_NAN
#define MARCO_Err_Dffu()        \
    ErDimw[g_id] = Dim_wall[l]; \
    Erhiw[g_id] = hi_wall[l];   \
    ErYiw[g_id] = Yi_wall[l];   \
    ErYilw[g_id] = Yil_wall[l];

#define MARCO_Err_VisFw() \
    ErvisFw[n + Emax * id] = F_wall_v[n];

#else
#define MARCO_Err_Dffu() ;
#define MARCO_Err_VisFw() ;
#endif

#ifdef Visc_Diffu
#define MARCO_VIS_Diffu()                                                                                                                           \
    real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn, CorrectTerm = _DF(0.0);                              \
    real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];                                                \
    for (int l = 0; l < NUM_SPECIES; l++)                                                                                                           \
    {                                                                                                                                               \
        int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;   \
        hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;                                                   \
        Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;                          \
        MARCO_VIS_COP_IN_DIFFU1();                                                                                                                  \
        /* visc flux for heat of diffusion */                                                                                                       \
        F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];                                                                           \
        MARCO_Err_Dffu();                                                                                                                           \
    }                                                                                                                                               \
    /* visc flux for cop equations*/                                                                                                                \
    CorrectTerm *= rho_wall;                                                                                                                        \
    for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                                                                          \
    {                                                                                                                                               \
        int p_temp = p - 5;                                                                                                                         \
        F_wall_v[p] = rho_wall * Dim_wall[p_temp] * Yil_wall[p_temp] - Yi_wall[p_temp] * CorrectTerm; /*CorrectTerm = 0.0 while not added in loop*/ \
    }
#else // Visc_Diffu
#define MARCO_VIS_Diffu()                                \
    for (int p = 5; p < Emax; p++) /* Avoid NAN error */ \
        F_wall_v[p] = 0.0;
#endif // Visc_Diffu

#ifdef Visc_Heat
#define MARCO_VIS_HEAT() /* thermal conductivity at wall*/                                                                                                   \
    real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; \
    kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr; /* temperature gradient at wall*/                                          \
    F_wall_v[4] += kk;
#else // else Visc_Heat
#define MARCO_VIS_HEAT() ;
#endif // end Visc_Heat

/**
 * get viscous flux
 */
#define MARCO_VISCFLUX()                                   \
    F_wall_v[0] = _DF(0.0);                                \
    F_wall_v[1] = f_x;                                     \
    F_wall_v[2] = f_y;                                     \
    F_wall_v[3] = f_z;                                     \
    F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf; \
    /* Fourier thermal conductivity*/                      \
    MARCO_VIS_HEAT();                                      \
    /* energy fiffusion depends on mass diffusion*/        \
    MARCO_VIS_Diffu();                                     \
    /* add viscous flux to fluxwall*/                      \
    for (size_t n = 0; n < Emax; n++)                      \
    {                                                      \
        Flux_wall[n + Emax * id] -= F_wall_v[n];           \
        MARCO_Err_VisFw();                                 \
    }

/**
 * prepare for getting viscous flux new
 */
// #ifdef COP
// const bool _COP = true;
// #define MARCO_VIS_COP_IN_DIFFU1()                                                                                                       \
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;                                           \
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/ \
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];

// #define MARCO_VIS_COP_IN_DIFFU2()                                                                     \
//     CorrectTerm *= rho_wall;                                                                          \
//     for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                            \
//     {                                                                                                 \
//         int p_temp = p - 5;                                                                           \
//         F_wall_v[p] = rho_wall * Dim_wall[p_temp] * Yil_wall[p_temp] - Yi_wall[p_temp] * CorrectTerm; \
//     }

// #else
// const bool _COP = false;
// #define MARCO_VIS_COP_IN_DIFFU1() Yil_wall[l] = _DF(0.0);
// #define MARCO_VIS_COP_IN_DIFFU2() ;
// #endif

// #ifdef Visc_Heat
// const bool _Heat = true;

// #define MARCO_VIS_HEAT()                                                                                                                                                                       \
//     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/ \
//     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/ \
//     F_wall_v[4] += kk;

// #define MARCO_VIS_HEAT_IN_DIFFU() \
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];

// // for (int l = 0; l < NUM_SPECIES; l++)                                 \
//     // {                                                                     \
//     //     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l]; \
//     // }
// #else
// const bool _Heat = false;
// #define MARCO_VIS_HEAT() ;
// #define MARCO_VIS_HEAT_IN_DIFFU() ;
// #endif

// #ifdef Visc_Diffu
// const bool _Diffu = true;

// #define MARCO_VIS_Diffu()                                                                                                                         \
//     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;                                                    \
//     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES], CorrectTerm = _DF(0.0);                      \
//     for (int l = 0; l < NUM_SPECIES; l++)                                                                                                         \
//     {                                                                                                                                             \
//         int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1; \
//         hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;                                                 \
//         Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;                        \
//         MARCO_VIS_COP_IN_DIFFU1();                                                                                                                \
//         /* visc flux for heat of diffusion */                                                                                                     \
//         MARCO_VIS_HEAT_IN_DIFFU();                                                                                                                \
//     }                                                                                                                                             \
//     /* visc flux for cop equations*/                                                                                                              \
//     MARCO_VIS_COP_IN_DIFFU2();

// #else
// const bool _Diffu = false;
// #define MARCO_VIS_Diffu() ;
// #endif

// F_wall_v[0] = _DF(0.0);
// F_wall_v[1] = f_x;
// F_wall_v[2] = f_y;
// F_wall_v[3] = f_z;
// F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;
// int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;                /* Fourier thermal conductivity*/
// real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/
// kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/
// F_wall_v[4] += kk;                                                                                                                                       /* energy fiffusion depends on mass diffusion*/
// real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;
// real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES], CorrectTerm = _DF(0.0);
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;
//     Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];
// } /* visc flux for cop equations*/
// CorrectTerm *= rho_wall;
// for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/
// {
//     F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;
// }
// for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/
// {
//     Flux_wall[n + Emax * id] -= F_wall_v[n];
// }

// F_wall_v[0] = _DF(0.0);
// F_wall_v[1] = f_x;
// F_wall_v[2] = f_y;
// F_wall_v[3] = f_z;
// F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;
// int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;                /* Fourier thermal conductivity*/
// real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/
// kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/
// F_wall_v[4] += kk;
// real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;
// real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;
//     Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/
// } /* visc flux for heat of diffusion */
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];
// }
// real_t for (int l = 0; l < NUM_SPECIES; l++)
// {
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];
// }
// CorrectTerm *= rho_wall;
// for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/
// {
//     F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;
// }
// for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/
// {
//     Flux_wall[n + Emax * id] -= F_wall_v[n];
// }

/**
 * Pre get eigen_martix
 */
#define MARCO_PREEIGEN()                      \
    real_t q2 = _u * _u + _v * _v + _w * _w;  \
    real_t _c = sqrt(c2);                     \
    real_t b2 = _DF(1.0) + b1 * q2 - b1 * _H; \
    real_t _c1 = _DF(1.0) / _c;

/**
 *
 */
#define MARCO_OUTLOOP                                        \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)                \
        for (int j = VTI.minY; j < VTI.maxY; j++)            \
            for (int i = VTI.minX; i < VTI.maxX; i++)

/**
 *
 */
#define MARCO_POUTLOOP(BODY)                          \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)         \
        for (int j = VTI.minY; j < VTI.maxY; j++)     \
        {                                             \
            for (int i = VTI.minX; i < VTI.maxX; i++) \
                out << BODY << " ";                   \
            out << "\n";                              \
        }

/**
 *  *
 */
#define MARCO_COUTLOOP                                       \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = minZ; k < maxZ; k++)                        \
        for (int j = minY; j < maxY; j++)                    \
            for (int i = minX; i < maxX; i++)

// =======================================================
// end repeated code definitions
// =======================================================
