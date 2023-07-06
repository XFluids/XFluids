#pragma once
#include "global_class.h"
#include "marco.h"
#include "device_func.hpp"
#include "ini_sample.hpp"

extern SYCL_EXTERNAL void EstimatePrimitiveVarKernel(int i, int j, int k, Block bl, Thermal thermal, int *error_pos, bool *error1, bool *error2, real_t *UI, real_t *rho,
                                                     real_t *u, real_t *v, real_t *w, real_t *p, real_t *T, real_t *y, real_t *H, real_t *e, real_t *gamma, real_t *c)
{ // numPte: number of Vars need be posoitive; numVars: length of *Vars(numbers of all Vars need to be estimed).
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
#if DIM_X
    if (i >= Xmax - bl.Bwidth_X)
        return;
#endif
#if DIM_Y
    if (j >= Ymax - bl.Bwidth_Y)
        return;
#endif
#if DIM_Z
    if (k >= bl.Zmax - bl.Bwidth_Z)
        return;
#endif

    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    // #ifdef ERROR_PATCH
    real_t theta = _DF(1.0);
#if DIM_X
    int id_xm = (Xmax * Ymax * k + Xmax * j + (i - 1));
    int id_xp = (Xmax * Ymax * k + Xmax * j + (i + 1));
    theta *= _DF(0.5);
#endif
#if DIM_Y
    int id_ym = (Xmax * Ymax * k + Xmax * (j - 1) + i);
    int id_yp = (Xmax * Ymax * k + Xmax * (j + 1) + i);
    theta *= _DF(0.5);
#endif
#if DIM_Z
    int id_zm = (Xmax * Ymax * (k - 1) + Xmax * j + i);
    int id_zp = (Xmax * Ymax * (k + 1) + Xmax * j + i);
    theta *= _DF(0.5);
#endif
    // #endif // end ERROR_PATCH

    bool spc = false, spcnan = false, spcs[NUM_SPECIES], spcnans[NUM_SPECIES];
    real_t *yi = &(y[NUM_SPECIES * id]), *U = &(UI[Emax * id]);
    for (size_t n2 = 0; n2 < NUM_SPECIES; n2++)
    {
        spcs[n2] = (yi[n2] < _DF(1e-20) || yi[n2] > _DF(1.0));
        spcnans[n2] = (sycl::isnan(yi[n2]) || sycl::isinf(yi[n2]));
        spc = spc || spcs[n2], spcnan = spcnan || spcnans[n2];
        if (spc || spcnan)
            error_pos[n2 + 3] = 1;
        if (spcs[n2])
        {
            // #ifdef ERROR_PATCH
            yi[n2] = _DF(0.0);
#if DIM_X
            yi[n2] += theta * (y[n2 + id_xm * NUM_SPECIES] + y[n2 + id_xp * NUM_SPECIES]);
#endif
#if DIM_Y
            yi[n2] += theta * (y[n2 + id_ym * NUM_SPECIES] + y[n2 + id_yp * NUM_SPECIES]);
#endif
#if DIM_Z
            yi[n2] += theta * (y[n2 + id_zm * NUM_SPECIES] + y[n2 + id_zp * NUM_SPECIES]);
#endif
            // #endif // end ERROR_PATCH
        }
    }
    // spcnan = true;
    // spc = true;
    if (spc || spcnan)
    {
        if (spcnan)
        {
#if DIM_X
            u[id] = _DF(0.5) * (u[id_xm] + u[id_xp]);
#endif
#if DIM_Y
            v[id] = _DF(0.5) * (v[id_ym] + v[id_yp]);
#endif
#if DIM_Z
            w[id] = _DF(0.5) * (w[id_zm] + w[id_zp]);
#endif
            // #endif // end ERROR_PATCH
        }
        *error2 = true; // add condition to avoid rewrite by other threads
        real_t sum = _DF(0.0);
        for (size_t nn = 0; nn < NUM_SPECIES; nn++)
            sum += yi[nn];
        sum = _DF(1.0) / sum;
        for (size_t nn = 0; nn < NUM_SPECIES; nn++)
            yi[nn] *= sum;
        ReGetStates(thermal, yi, U, rho[id], u[id], v[id], w[id], p[id], T[id], H[id], c[id], e[id], gamma[id]);
    }

    bool ngatve = false, ngatves[3];
    real_t ngaVs[3] = {rho[id], p[id], T[id]}, *ngaPatch[3] = {rho, p, T};
    for (size_t n1 = 0; n1 < 3; n1++)
    {
        ngatves[n1] = (ngaVs[n1] < 0) || sycl::isnan(ngaVs[n1]) || sycl::isinf(ngaVs[n1]);
        ngatve = ngatve || ngatves[n1];
        if (ngatves[n1])
        {
            error_pos[n1] = 1;
#ifdef ERROR_PATCH // may cause physical unconservative
            ngaVs[n1] = _DF(0.0);
#if DIM_X
            ngaVs[n1] += theta * (ngaPatch[n1][id_xm] + ngaPatch[n1][id_xp]);
#endif
#if DIM_Y
            ngaVs[n1] += theta * (ngaPatch[n1][id_ym] + ngaPatch[n1][id_yp]);
#endif
#if DIM_Z
            ngaVs[n1] += theta * (ngaPatch[n1][id_zm] + ngaPatch[n1][id_zp]);
#endif
#endif // end ERROR_PATCH
        }
    }
    // ngatve = true;
    if (ngatve || spcnan)
    {
        error_pos[3 + NUM_SPECIES] = i, error_pos[4 + NUM_SPECIES] = j, error_pos[5 + NUM_SPECIES] = k;
        if (ngatve)
            *error1 = true;  // add condition to avoid rewrite by other threads
    }                        // *error = true;
}

extern SYCL_EXTERNAL void EstimateFluidNANKernel(int i, int j, int k, int x_offset, int y_offset, int z_offset, Block bl, int *error_pos, real_t *UI, real_t *LUI, bool *error) //, sycl::stream stream_ct1
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int id = (Xmax * Ymax * k + Xmax * j + i) * Emax;
#if DIM_X
    if (i >= Xmax - bl.Bwidth_X)
        return;
#endif
#if DIM_Y
    if (j >= Ymax - bl.Bwidth_Y)
        return;
#endif
#if DIM_Z
    if (k >= bl.Zmax - bl.Bwidth_Z)
        return;
#endif
    // #ifdef ERROR_PATCH
    //     real_t theta = _DF(1.0);
    // #if DIM_X
    //     int id_xm = (Xmax * Ymax * k + Xmax * j + (i - 6)) * Emax;
    //     int id_xp = (Xmax * Ymax * k + Xmax * j + (i + 6)) * Emax;
    //     theta *= _DF(0.5);
    // #endif
    // #if DIM_Y
    //     int id_ym = (Xmax * Ymax * k + Xmax * (j - 6) + i) * Emax;
    //     int id_yp = (Xmax * Ymax * k + Xmax * (j + 6) + i) * Emax;
    //     theta *= _DF(0.5);
    // #endif
    // #if DIM_Z
    //     int id_zm = (Xmax * Ymax * (k - 6) + Xmax * j + i) * Emax;
    //     int id_zp = (Xmax * Ymax * (k + 6) + Xmax * j + i) * Emax;
    //     theta *= _DF(0.5);
    // #endif
    // #endif // end ERROR_PATCH
    bool tempnegv = UI[0 + id] < 0; //|| UI[4 + id] < 0;
    for (size_t ii = 0; ii < Emax; ii++)
    { //(i == 150 && j == 30 && ii == 0) ? true :
        bool tempnan = sycl::isnan(UI[ii + id]) || sycl::isinf(UI[ii + id]);
        if (tempnan || tempnegv)
        {
            *error = true;
            error_pos[ii] = 1;
            // error_pos[Emax] = i;
            // error_pos[Emax + 1] = j;
            // error_pos[Emax + 2] = k;
            // #ifdef ERROR_PATCH
            //             UI[ii + id] = _DF(0.0);
            // #if DIM_X
            //             UI[ii + id] += theta * (UI[ii + id_xm] + UI[ii + id_xp]);
            // #endif
            // #if DIM_Y
            //             UI[ii + id] += theta * (UI[ii + id_ym] + UI[ii + id_yp]);
            // #endif
            // #if DIM_Z
            //             UI[ii + id] += theta * (UI[ii + id_zm] + UI[ii + id_zp]);
            // #endif
            // #endif // end ERROR_PATCH
        }
    }
}

extern SYCL_EXTERNAL void PositivityPreservingKernel(int i, int j, int k, int id_l, int id_r, Block bl, Thermal thermal,
                                                     real_t *UI, real_t *Fl, real_t *Fwall, const real_t T_l, const real_t T_r,
                                                     const real_t lambda_0, const real_t lambda, const real_t *epsilon) // , sycl::stream stream epsilon[NUM_SPECIES+2]={rho, e, y(0), ..., y(n)}
{
#if DIM_X
    if (i >= bl.Xmax - bl.Bwidth_X)
        return;
#endif
#if DIM_Y
    if (j >= bl.Ymax - bl.Bwidth_Y)
        return;
#endif
#if DIM_Z
    if (k >= bl.Zmax - bl.Bwidth_Z)
        return;
#endif
    // stream << "eps: " << epsilon[0] << " " << epsilon[3] << " " << epsilon[5] << " " << epsilon[NUM_SPECIES + 1] << "\n";
    // int id_l = (Xmax * Ymax * k + Xmax * j + i) * Emax;
    // int id_r = (Xmax * Ymax * k + Xmax * j + i + 1) * Emax;
    // Positivity preserving flux limiter form Dr.Hu.https://www.sciencedirect.com/science/article/pii/S0021999113000557, expand to multicomponent, need positivity Initial value
    real_t rho_min, theta, theta_u, theta_p, F_LF[Emax], FF_LF[Emax], FF[Emax], *UU = &(UI[id_l]), *UP = &(UI[id_r]); // UU[Emax], UP[Emax],
    for (int n = 0; n < Emax; n++)
    {
        F_LF[n] = _DF(0.5) * (Fl[n + id_l] + Fl[n + id_r] + lambda_0 * (UI[n + id_l] - UI[n + id_r])); // FF_LF == F_(i+1/2) of Lax-Friedrichs
        FF_LF[n] = _DF(2.0) * lambda * F_LF[n];                                                        // U_i^+ == (U_i^n-FF_LF)
        FF[n] = _DF(2.0) * lambda * Fwall[n + id_l];                                                   // FF from original high accuracy scheme
    }

    // // correct for positive density
    theta_u = _DF(1.0), theta_p = _DF(1.0);
    rho_min = sycl::min<real_t>(UU[0], epsilon[0]);
    if (UU[0] - FF[0] < rho_min)
        theta_u = (UU[0] - FF_LF[0] - rho_min) / (FF[0] - FF_LF[0]);
    rho_min = sycl::min<real_t>(UP[0], epsilon[0]);
    if (UP[0] + FF[0] < rho_min)
        theta_p = (UP[0] + FF_LF[0] - rho_min) / (FF_LF[0] - FF[0]);
    theta = sycl::min<real_t>(theta_u, theta_p);
    for (int n = 0; n < Emax; n++)
    {
        FF[n] = (_DF(1.0) - theta) * FF_LF[n] + theta * FF[n];
        Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
    }

    // // correct for yi // need Ini of yi > 0(1.0e-10 set)
    real_t yi_q[NUM_SPECIES], yi_u[NUM_SPECIES], yi_qp[NUM_SPECIES], yi_up[NUM_SPECIES], _rhoq, _rhou, _rhoqp, _rhoup;
    _rhoq = _DF(1.0) / (UU[0] - FF[0]), _rhou = _DF(1.0) / (UU[0] - FF_LF[0]), yi_q[NUM_COP] = _DF(1.0), yi_u[NUM_COP] = _DF(1.0);
    _rhoqp = _DF(1.0) / (UP[0] + FF[0]), _rhoup = _DF(1.0) / (UP[0] + FF_LF[0]), yi_qp[NUM_COP] = _DF(1.0), yi_up[NUM_COP] = _DF(1.0);
    for (size_t n = 0; n < NUM_COP; n++)
    {
        int tid = n + 5;
        yi_q[n] = (UU[tid] - FF[tid]) * _rhoq, yi_q[NUM_COP] -= yi_q[n];
        yi_u[n] = (UU[tid] - FF_LF[tid]) * _rhou, yi_u[NUM_COP] -= yi_u[n];
        yi_qp[n] = (UP[tid] + FF[tid]) * _rhoqp, yi_qp[NUM_COP] -= yi_qp[n];
        yi_up[n] = (UP[tid] + FF_LF[tid]) * _rhoup, yi_up[NUM_COP] -= yi_up[n];
        // real_t temp = epsilon[n + 2];
        // if (yi_q[n] < temp)
        // {
        //     real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
        //     theta_u = (yi_u[n] - yi_min) / (yi_u[n] - yi_q[n]);
        // }
        // if (yi_qp[n] < temp)
        // {
        //     real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
        //     theta_p = (yi_up[n] - yi_min) / (yi_up[n] - yi_qp[n]);
        // }
        // theta = sycl::min<real_t>(theta_u, theta_p);
        // for (int nn = 0; nn < Emax; nn++)
        //     Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
    }
    // // // correct for yn
    // real_t temp = epsilon[NUM_SPECIES + 1];
    // if (yi_q[NUM_COP] < temp)
    // {
    //     real_t yi_min = sycl::min<real_t>(yi_u[NUM_COP], temp);
    //     theta_u = (yi_u[NUM_COP] - yi_min) / (yi_u[NUM_COP] - yi_q[NUM_COP]);
    // }
    // if (yi_qp[NUM_COP] < temp)
    // {
    //     real_t yi_min = sycl::min<real_t>(yi_up[NUM_COP], temp);
    //     theta_p = (yi_up[NUM_COP] - yi_min) / (yi_up[NUM_COP] - yi_qp[NUM_COP]);
    // }
    // theta = sycl::min<real_t>(theta_u, theta_p);
    // for (int nn = 0; nn < Emax; nn++)
    //     Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];

#ifdef COP_CHEME
    size_t begin = NUM_SPECIES - 2, end = NUM_SPECIES - 1;
#else
    size_t begin = NUM_SPECIES - 2, end = NUM_SPECIES - 1;
#endif // end COP_CHEME

    for (size_t n = 0; n < 2; n++) // NUM_SPECIES - 2
    {
        theta_u = _DF(1.0), theta_p = _DF(1.0);
        real_t temp = epsilon[n + 2];
        if (yi_q[n] < temp)
        {
            real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
            theta_u = (yi_u[n] - yi_min) / (yi_u[n] - yi_q[n]);
        }
        if (yi_qp[n] < temp)
        {
            real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
            theta_p = (yi_up[n] - yi_min) / (yi_up[n] - yi_qp[n]);
        }
        theta = sycl::min<real_t>(theta_u, theta_p);
        for (int nn = 0; nn < Emax; nn++)
            Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
    }
    for (size_t n = begin; n < end; n++) // NUM_SPECIES - 2
    {
        theta_u = _DF(1.0), theta_p = _DF(1.0);
        real_t temp = epsilon[n + 2];
        if (yi_q[n] < temp)
        {
            real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
            theta_u = (yi_u[n] - yi_min) / (yi_u[n] - yi_q[n]);
        }
        if (yi_qp[n] < temp)
        {
            real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
            theta_p = (yi_up[n] - yi_min) / (yi_up[n] - yi_qp[n]);
        }
        theta = sycl::min<real_t>(theta_u, theta_p);
        for (int nn = 0; nn < Emax; nn++)
            Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
    }

    // // correct for positive p, method to get p for multicomponent theory:
    // // e = UI[4]*_rho-_DF(0.5)*_rho*_rho*(UI[1]*UI[1]+UI[2]*UI[2]+UI[3]*UI[3]);
    // // R = get_CopR(thermal._Wi, yi); T = get_T(thermal, yi, e, T); p = rho * R * T;
    // // known that rho and yi has been preserved to be positive, only need to preserve positive T
    real_t e_q, T_q, P_q, theta_pu = 1.0, theta_pp = 1.0;
    theta_u = _DF(1.0), theta_p = _DF(1.0);
    e_q = (UU[4] - FF[4] - _DF(0.5) * ((UU[1] - FF[1]) * (UU[1] - FF[1]) + (UU[2] - FF[2]) * (UU[2] - FF[2]) + (UU[3] - FF[3]) * (UU[3] - FF[3])) * _rhoq) * _rhoq;
    T_q = get_T(thermal, yi_q, e_q, T_l);
    P_q = T_q * get_CopR(thermal._Wi, yi_q);
    if (T_q < epsilon[1])
    {
        real_t e_u = (UU[4] - FF_LF[4] - _DF(0.5) * ((UU[1] - FF_LF[1]) * (UU[1] - FF_LF[1]) + (UU[2] - FF_LF[2]) * (UU[2] - FF_LF[2]) + (UU[3] - FF_LF[3]) * (UU[3] - FF_LF[3])) * _rhou) * _rhou;
        real_t T_u = get_T(thermal, yi_u, e_u, T_l);
        real_t T_min = sycl::min<real_t>(T_u, epsilon[1]);
        theta_u = (T_u - T_min) / (T_u - T_q);
    }
    if (P_q < epsilon[1])
    {
        real_t e_u = (UU[4] - FF_LF[4] - _DF(0.5) * ((UU[1] - FF_LF[1]) * (UU[1] - FF_LF[1]) + (UU[2] - FF_LF[2]) * (UU[2] - FF_LF[2]) + (UU[3] - FF_LF[3]) * (UU[3] - FF_LF[3])) * _rhou) * _rhou;
        real_t P_u = get_T(thermal, yi_u, e_u, T_l) * get_CopR(thermal._Wi, yi_u);
        real_t P_min = sycl::min<real_t>(P_u, epsilon[1]);
        theta_pu = (P_u - P_min) / (P_u - P_q);
    }

    e_q = (UP[4] + FF[4] - _DF(0.5) * ((UP[1] + FF[1]) * (UP[1] + FF[1]) + (UP[2] + FF[2]) * (UP[2] + FF[2]) + (UP[3] + FF[3]) * (UP[3] + FF[3])) * _rhoqp) * _rhoqp;
    T_q = get_T(thermal, yi_qp, e_q, T_r);
    P_q = T_q * get_CopR(thermal._Wi, yi_qp);
    if (T_q < epsilon[1])
    {
        real_t e_p = (UP[4] + FF_LF[4] - _DF(0.5) * ((UP[1] + FF_LF[1]) * (UP[1] + FF_LF[1]) + (UP[2] + FF_LF[2]) * (UP[2] + FF_LF[2]) + (UP[3] + FF_LF[3]) * (UP[3] + FF_LF[3])) * _rhoup) * _rhoup;
        real_t T_p = get_T(thermal, yi_up, e_p, T_r);
        real_t T_min = sycl::min<real_t>(T_p, epsilon[1]);
        theta_p = (T_p - T_min) / (T_p - T_q);
    }
    if (P_q < epsilon[1])
    {
        real_t e_p = (UP[4] + FF_LF[4] - _DF(0.5) * ((UP[1] + FF_LF[1]) * (UP[1] + FF_LF[1]) + (UP[2] + FF_LF[2]) * (UP[2] + FF_LF[2]) + (UP[3] + FF_LF[3]) * (UP[3] + FF_LF[3])) * _rhoup) * _rhoup;
        real_t P_p = get_T(thermal, yi_up, e_p, T_r) * get_CopR(thermal._Wi, yi_qp);
        real_t P_min = sycl::min<real_t>(P_p, epsilon[1]);
        theta_pp = (P_p - P_min) / (P_p - P_q);
    }
    theta = sycl::min<real_t>(theta_u, theta_p);
    for (int n = 0; n < Emax; n++)
        Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
    theta = sycl::min<real_t>(theta_pu, theta_pp);
    for (int n = 0; n < Emax; n++)
        Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
}

#if DIM_X
extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
                                           real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H)
{
    MARCO_DOMAIN_GHOST();
    if (i >= X_inner + Bwidth_X)
        return;
    if (j >= Y_inner + Bwidth_Y)
        return;
    if (k >= Z_inner + Bwidth_Z)
        return;

    int id_l = Xmax * Ymax * k + Xmax * j + i;
    int id_r = Xmax * Ymax * k + Xmax * j + i + 1;
    real_t dl = bl.dx;

    // preparing some interval value for roe average
    MARCO_ROE();

    MARCO_GETC2()

#if 1 == EIGEN_ALLOC
    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
    real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
    for (size_t n = 0; n < Emax; n++)
    {
        eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
        eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
    }
#endif // end EIGEN_ALLOC
//     // RoeAverage_x(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

//     // construct the right value & the left value scalar equations by characteristic reduction
//     // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m, j, k);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNX, MARCO_ROEAVERAGE_RIGHTNX, i + m, j, k, i + m - stencil_P, j, k);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNX, MARCO_ROEAVERAGE_RIGHTNX, i + m, j, k, i + m, j, k);
// #endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_X, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_X, i + m, j, k, i + m, j, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

    //     // real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value;
    //     // for (int n = 0; n < Emax; n++)
    //     // {
    //     //     real_t eigen_local_max = _DF(0.0);
    //     //     RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); /*get eigen_l*/
    //     //     for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
    //     //     {
    //     //         int _i_1 = i + m, _j_1 = j, _k_1 = k;
    //     //         int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/
    //     //         eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
    //     //     }
    //     //     for (int m = -3; m <= 4; m++)
    //     //     {
    //     //         int _i_2 = i + m, _j_2 = j, _k_2 = k;                         /* int _i_2 = i + m, _j_2 = j, _k_2 = k; 3rd oder and can be modified*/
    //     //         int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /*Xmax * Ymax * k + Xmax * j + m + i;*/
    //     //         uf[m + 3] = _DF(0.0);
    //     //         ff[m + 3] = _DF(0.0);
    //     //         for (int n1 = 0; n1 < Emax; n1++)
    //     //         {
    //     //             uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; /* eigen_l actually */
    //     //             ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];
    //     //         } /*  for local speed*/
    //     //         pp[m + 3] = _DF(0.5) * (ff[m + 3] + eigen_local_max * uf[m + 3]);
    //     //         mm[m + 3] = _DF(0.5) * (ff[m + 3] - eigen_local_max * uf[m + 3]);
    //     //     }                                                                                                                                     /* calculate the scalar numerical flux at x direction*/
    //     //     f_flux = (weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)); /* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/ /* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/
    //     //     RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);                                                     /* get eigen_r */
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {                                      /* get Fp */
    //     //         _p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */
    //     //     }
    //     // }
    //     // for (int n = 0; n < Emax; n++)
    //     // { /* reconstruction the F-flux terms*/
    //     //     Fwall[Emax * id_l + n] = _DF(0.0);
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         Fwall[Emax * id_l + n] += _p[n1][n];
    //     //     }
    //     // }

    //     // real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value;
    //     // for (int n = 0; n < Emax; n++)
    //     // {
    //     //     real_t eigen_local_max = _DF(0.0);
    //     //     MARCO_ROEAVERAGE_LEFTX; // MARCO_ROE_LEFT; /* eigen_r actually */ /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */
    //     //     eigen_local_max = eigen_value;
    //     //     real_t lambda_l = eigen_local[Emax * id_l + n];
    //     //     real_t lambda_r = eigen_local[Emax * id_r + n];
    //     //     if (lambda_l * lambda_r < 0.0)
    //     //     {
    //     //         for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
    //     //         {
    //     //             int _i_1 = i + m, _j_1 = j, _k_1 = k;
    //     //             int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */
    //     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
    //     //         }
    //     //     }
    //     //     for (size_t m = 0; m < stencil_size; m++)
    //     //     {
    //     //         int _i_2 = i + m - stencil_P, _j_2 = j, _k_2 = k;
    //     //         int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */
    //     //         uf[m] = _DF(0.0);
    //     //         ff[m] = _DF(0.0);
    //     //         for (int n1 = 0; n1 < Emax; n1++)
    //     //         {
    //     //             uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];
    //     //             ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];
    //     //         } /* for local speed*/
    //     //         pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);
    //     //         mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);
    //     //     }                                                                   /* calculate the scalar numerical flux at x direction*/
    //     //     f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/
    //     //     MARCO_ROEAVERAGE_RIGHTX;                                            // MARCO_ROE_RIGHT; /* eigen_r actually */                                                    /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         _p[n][n1] = f_flux * eigen_lr[n1];
    //     //     }
    //     // } /* reconstruction the F-flux terms*/
    //     // for (int n = 0; n < Emax; n++)
    //     // {
    //     //     real_t temp_flux = _DF(0.0);
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         temp_flux += _p[n1][n];
    //     //     }
    //     //     Fwall[Emax * id_l + n] = temp_flux;
    //     // }

    //     // real_t de_fw[Emax];
    //     // get_Array(Fwall, de_fw, Emax, id_l);
    //     // real_t de_fx[Emax];
}
#endif // end DIM_X

#if DIM_Y
extern SYCL_EXTERNAL void ReconstructFluxY(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
                                           real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H)
{
    MARCO_DOMAIN_GHOST();
    if (i >= X_inner + Bwidth_X)
        return;
    if (j >= Y_inner + Bwidth_Y)
        return;
    if (k >= Z_inner + Bwidth_Z)
        return;

    int id_l = Xmax * Ymax * k + Xmax * j + i;
    int id_r = Xmax * Ymax * k + Xmax * (j + 1) + i;
    real_t dl = bl.dy;

    // preparing some interval value for roe average
    MARCO_ROE();

    MARCO_GETC2()

#if 1 == EIGEN_ALLOC
    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
    real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
    for (size_t n = 0; n < Emax; n++)
    {
        eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
        eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
    }
#endif // end EIGEN_ALLOC
    //     // RoeAverage_y(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

    //     // // construct the right value & the left value scalar equations by characteristic reduction
    //     // // at i+1/2 in x direction

#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m, k);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNY, MARCO_ROEAVERAGE_RIGHTNY, i, j + m, k, i, j + m - stencil_P, k);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNY, MARCO_ROEAVERAGE_RIGHTNY, i, j + m, k, i, j + m, k);
// #endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

    //     // real_t de_fw[Emax];
    //     // get_Array(Fwall, de_fw, Emax, id_l);
    //     // real_t de_fx[Emax];
}
#endif // end DIM_Y

#if DIM_Z
extern SYCL_EXTERNAL void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
                                           real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H)
{
    MARCO_DOMAIN_GHOST();
    if (i >= X_inner + Bwidth_X)
        return;
    if (j >= Y_inner + Bwidth_Y)
        return;
    if (k >= Z_inner + Bwidth_Z)
        return;

    int id_l = Xmax * Ymax * k + Xmax * j + i;
    int id_r = Xmax * Ymax * (k + 1) + Xmax * j + i;
    real_t dl = bl.dz;

    // preparing some interval value for roe average
    MARCO_ROE();

    MARCO_GETC2()

#if 1 == EIGEN_ALLOC
    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
    real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
    for (size_t n = 0; n < Emax; n++)
    {
        eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
        eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
    }
#endif // end EIGEN_ALLOC
    //     // RoeAverage_z(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

    //     // // construct the right value & the left value scalar equations by characteristic reduction
    //     // // at i+1/2 in x direction

#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNZ, MARCO_ROEAVERAGE_RIGHTNZ, i, j, k + m, i, j, k + m - stencil_P);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNZ, MARCO_ROEAVERAGE_RIGHTNZ, i, j, k + m, i, j, k + m);
// #endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

    //     // real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value, eigen_l[Emax][Emax], eigen_r[Emax][Emax];
    //     // for (int n = 0; n < Emax; n++)
    //     // {
    //     //     real_t eigen_local_max = _DF(0.0);
    //     //     MARCO_ROEAVERAGE_LEFTZ; /* eigen_r actually */ /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */
    //     //     eigen_local_max = eigen_value;
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         eigen_l[n][n1] = eigen_lr[n1];
    //     //     }
    //     //     real_t lambda_l = eigen_local[Emax * id_l + n];
    //     //     real_t lambda_r = eigen_local[Emax * id_r + n];
    //     //     if (lambda_l * lambda_r < 0.0)
    //     //     {
    //     //         for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
    //     //         {
    //     //             int _i_1 = i, _j_1 = j, _k_1 = k + m;
    //     //             int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */
    //     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
    //     //         }
    //     //     }
    //     //     for (size_t m = 0; m < stencil_size; m++)
    //     //     {
    //     //         int _i_2 = i, _j_2 = j, _k_2 = k + m - stencil_P;
    //     //         int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */
    //     //         uf[m] = _DF(0.0);
    //     //         ff[m] = _DF(0.0);
    //     //         for (int n1 = 0; n1 < Emax; n1++)
    //     //         {
    //     //             uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];
    //     //             ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];
    //     //         } /* for local speed*/
    //     //         pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);
    //     //         mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);
    //     //     }                                                                   /* calculate the scalar numerical flux at x direction*/
    //     //     f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/
    //     //     MARCO_ROEAVERAGE_RIGHTZ; /* eigen_r actually */                     /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         eigen_r[n1][n] = eigen_lr[n1];
    //     //     }
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         _p[n][n1] = f_flux * eigen_lr[n1];
    //     //     }
    //     // }
    //     // for (int n = 0; n < Emax; n++)
    //     // { /* reconstruction the F-flux terms*/
    //     //     Fwall[Emax * id_l + n] = _DF(0.0);
    //     //     for (int n1 = 0; n1 < Emax; n1++)
    //     //     {
    //     //         Fwall[Emax * id_l + n] += _p[n1][n];
    //     //     }
    //     // }

    //     // real_t de_fw[Emax];
    //     // get_Array(Fwall, de_fw, Emax, id_l);
    //     // real_t de_fx[Emax];
}
#endif // end DIM_Z

extern SYCL_EXTERNAL void GetLocalEigen(int i, int j, int k, Block bl, real_t AA, real_t BB, real_t CC, real_t *eigen_local, real_t *u, real_t *v, real_t *w, real_t *c)
{
    MARCO_DOMAIN();
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    if(i >= Xmax)
    return;
#endif
#if DIM_Y
    if(j >= Ymax)
    return;
#endif
#if DIM_Z
    if(k >= Zmax)
    return;
#endif
#if SCHEME_ORDER <= 6
    real_t uu = AA * u[id] + BB * v[id] + CC * w[id];
    real_t uuPc = uu + c[id];
    real_t uuMc = uu - c[id];

    // local eigen values
    eigen_local[Emax * id + 0] = uuMc;
    for (size_t ii = 1; ii < Emax - 1; ii++)
    {
            eigen_local[Emax * id + ii] = uu;
    }
    eigen_local[Emax * id + Emax - 1] = uuPc;
#elif SCHEME_ORDER == 7
    for (size_t ii = 0; ii < Emax; ii++)
        eigen_local[Emax * id + ii] = 0.0;
#endif // end FLUX_method

    // real_t de_fw[Emax];
    // get_Array(eigen_local, de_fw, Emax, id);
    // real_t de_fx[Emax];
}

extern SYCL_EXTERNAL void UpdateFluidLU(int i, int j, int k, Block bl, real_t *LU, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
{
    MARCO_DOMAIN();
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    int id_im = Xmax * Ymax * k + Xmax * j + i - 1;
#endif
#if DIM_Y
    int id_jm = Xmax * Ymax * k + Xmax * (j - 1) + i;
#endif
#if DIM_Z
    int id_km = Xmax * Ymax * (k - 1) + Xmax * j + i;
#endif

    for (int n = 0; n < Emax; n++)
    {
    real_t LU0 = _DF(0.0);
#if DIM_X
    LU0 += (FluxFw[Emax * id_im + n] - FluxFw[Emax * id + n]) * bl._dx;
#endif
#if DIM_Y
    LU0 += (FluxGw[Emax * id_jm + n] - FluxGw[Emax * id + n]) * bl._dy;
#endif
#if DIM_Z
    LU0 += (FluxHw[Emax * id_km + n] - FluxHw[Emax * id + n]) * bl._dz;
#endif
    LU[Emax * id + n] = LU0;
    }

    // real_t de_LU[Emax];
    // get_Array(LU, de_LU, Emax, id);
}

extern SYCL_EXTERNAL void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
                                                 real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *_y,
                                                 real_t *gamma, real_t *T, real_t *e, real_t const Gamma) //, const sycl::stream &stream_ct1
{
    MARCO_DOMAIN_GHOST();
    int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
    if (i >= Xmax)
    return;
#endif
#if DIM_Y
    if (j >= Ymax)
    return;
#endif
#if DIM_Z
    if (k >= Zmax)
    return;
#endif

    real_t *U = &(UI[Emax * id]), *yi = &(_y[NUM_SPECIES * id]); //, yi[NUM_SPECIES];

    GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], gamma[id], T[id], e[id], thermal, yi);

    // if (i == 80 && j == 25)
    // {
    // yi[0] += 20.0;
    // yi[NUM_COP] += -20.0;
    // // p[id] = -10000.0;
    // // c[id] = sycl::sqrt<real_t>(-1.0);
    // ReGetStates(thermal, yi, U[4], rho[id], p[id], T[id], H[id], c[id], e[id], gamma[id]);
    // }

    // // real_t x = DIM_X ? (i - Bwidth_X + bl.myMpiPos_x * X_inner + _DF(0.5)) * bl.dx + bl.Domain_xmin : _DF(0.0);
    // // real_t y = DIM_Y ? (j - Bwidth_Y + bl.myMpiPos_y * Y_inner + _DF(0.5)) * bl.dy + bl.Domain_ymin : _DF(0.0);
    // // real_t z = DIM_Z ? (k - Bwidth_Z + bl.myMpiPos_z * Z_inner + _DF(0.5)) * bl.dz + bl.Domain_zmin : _DF(0.0);

    real_t *Fx = &(FluxF[Emax * id]);
    real_t *Fy = &(FluxG[Emax * id]);
    real_t *Fz = &(FluxH[Emax * id]);

    GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);

    // // for (size_t n = 0; n < NUM_SPECIES; n++)
    // //     _y[n][id] = yi[n];

    // // real_t de_fx[Emax], de_fy[Emax], de_fz[Emax];
    // // get_Array(FluxF, de_fx, Emax, id);
    // // get_Array(FluxG, de_fy, Emax, id);
    // // get_Array(FluxH, de_fz, Emax, id);
}

extern SYCL_EXTERNAL void UpdateURK3rdKernel(int i, int j, int k, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
    MARCO_DOMAIN();
    int id = Xmax * Ymax * k + Xmax * j + i;

    // real_t de_U[Emax], de_U1[Emax], de_LU[Emax];
    switch (flag)
    {
    case 1:
        for (int n = 0; n < Emax; n++)
            U1[Emax * id + n] = U[Emax * id + n] + dt * LU[Emax * id + n];
        break;
    case 2:
        for (int n = 0; n < Emax; n++)
            U1[Emax * id + n] = _DF(0.75) * U[Emax * id + n] + _DF(0.25) * U1[Emax * id + n] + _DF(0.25) * dt * LU[Emax * id + n];
        break;
    case 3:
        for (int n = 0; n < Emax; n++)
            U[Emax * id + n] = (U[Emax * id + n] + _DF(2.0) * U1[Emax * id + n] + _DF(2.0) * dt * LU[Emax * id + n]) * _OT;
        break;
    }
    // get_Array(U, de_U, Emax, id);
    // get_Array(U1, de_U1, Emax, id);
    // get_Array(LU, de_LU, Emax, id);
}

extern SYCL_EXTERNAL void FluidBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
    MARCO_DOMAIN_GHOST();
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_Y
    if(j >= Ymax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

    switch(BC) {
        case Symmetry:
        {
        int offset = 2 * (Bwidth_X + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_X + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
        for (int n = Emax - NUM_COP; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
        }
        break;
#ifdef USE_MPI
        case BC_COPY:
        break;
        case BC_UNDEFINED:
        break;
#endif
        }
}

extern SYCL_EXTERNAL void FluidBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
        MARCO_DOMAIN_GHOST();
        int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
        if (i >= Xmax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

    switch(BC) {
        case Symmetry:
        {
        int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
        int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
        for (int n = Emax - NUM_COP; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
        }
        break;
#ifdef USE_MPI
        case BC_COPY:
        break;
        case BC_UNDEFINED:
        break;
#endif
        }
}

extern SYCL_EXTERNAL void FluidBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
        MARCO_DOMAIN_GHOST();
        int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
        if (i >= Xmax)
        return;
#endif
#if DIM_Y
        if (j >= Ymax)
        return;
#endif

        switch (BC)
        {
        case Symmetry:
        {
        int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
        int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        }
        break;

        case Periodic:
        {
        int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Inflow:
        break;

        case Outflow:
        {
        int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
        for (int n = 0; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
        }
        break;

        case Wall:
        {
        int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
        int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
        d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
        d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
        d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
        d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
        d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
        for (int n = Emax - NUM_COP; n < Emax; n++)
            d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
        }
        break;
#ifdef USE_MPI
        case BC_COPY:
        break;
        case BC_UNDEFINED:
        break;
#endif
        }
}

#if USE_MPI
extern SYCL_EXTERNAL void FluidMpiCopyKernelX(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
                                              const int Bwidth_Xset, const MpiCpyType Cpytype)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;
    int id = Xmax * Ymax * k + Xmax * j + i;
    int tid = sycl::abs(Bwidth_Xset) * Ymax * k + sycl::abs(Bwidth_Xset) * j + (i - index_offset);
    int fid = Xmax * Ymax * k + Xmax * j + (i - Bwidth_Xset);

#if DIM_Y
    if (j >= Ymax)
        return;
#endif
#if DIM_Z
    int Zmax = bl.Zmax;
    if (k >= Zmax)
        return;
#endif
    for (size_t n = 0; n < Emax; n++)
    {
        if (BorToBuf == Cpytype)
            d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
        if (BufToBC == Cpytype)
            d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
    }
}

extern SYCL_EXTERNAL void FluidMpiCopyKernelY(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
                                              const int Bwidth_Yset, const MpiCpyType Cpytype)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;

    int id = Xmax * Ymax * k + Xmax * j + i;
    int tid = Xmax * sycl::abs(Bwidth_Yset) * k + Xmax * (j - index_offset) + i;
    int fid = Xmax * Ymax * k + Xmax * (j - Bwidth_Yset) + i;

#if DIM_Y
    if (j >= Ymax)
        return;
#endif
#if DIM_Z
    if (k >= bl.Zmax)
        return;
#endif
    for (size_t n = 0; n < Emax; n++)
    {
        if (BorToBuf == Cpytype)
            d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
        if (BufToBC == Cpytype)
            d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
    }
}

extern SYCL_EXTERNAL void FluidMpiCopyKernelZ(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
                                              const int Bwidth_Zset, const MpiCpyType Cpytype)
{
    int Xmax = bl.Xmax;
    int Ymax = bl.Ymax;

    int id = Xmax * Ymax * k + Xmax * j + i;
    int tid = Xmax * Ymax * (k - index_offset) + Xmax * j + i;
    int fid = Xmax * Ymax * (k - Bwidth_Zset) + Xmax * j + i;

#if DIM_Y
    if (j >= Ymax)
        return;
#endif
#if DIM_Z
    if (k >= bl.Zmax)
        return;
#endif
    for (size_t n = 0; n < Emax; n++)
    {
        if (BorToBuf == Cpytype)
            d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
        if (BufToBC == Cpytype)
            d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
    }
}
#endif // end USE_MPI

extern SYCL_EXTERNAL void GetInnerCellCenterDerivativeKernel(int i, int j, int k, Block bl, real_t *u, real_t *v, real_t *w, real_t *const *Vde)
{
#if DIM_X
    if (i > bl.Xmax - bl.Bwidth_X + 1)
        return;
#endif // DIM_X
#if DIM_Y
    if (j > bl.Ymax - bl.Bwidth_Y + 1)
        return;
#endif // DIM_Y
#if DIM_Z
    if (k > bl.Zmax - bl.Bwidth_Z + 1)
        return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;

    Vde[ducx][id] = _DF(0.0);
    Vde[dvcx][id] = _DF(0.0);
    Vde[dwcx][id] = _DF(0.0);
    Vde[ducy][id] = _DF(0.0);
    Vde[dvcy][id] = _DF(0.0);
    Vde[dwcy][id] = _DF(0.0);
    Vde[ducz][id] = _DF(0.0);
    Vde[dvcz][id] = _DF(0.0);
    Vde[dwcz][id] = _DF(0.0);

#if DIM_X
    real_t _dx = bl._dx;
    int id_m1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
    int id_m2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
    int id_p1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
    int id_p2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;

    Vde[ducx][id] = (_DF(8.0) * (u[id_p1_x] - u[id_m1_x]) - (u[id_p2_x] - u[id_m2_x])) * _dx * _twle;
#if DIM_Y
    Vde[dvcx][id] = (_DF(8.0) * (v[id_p1_x] - v[id_m1_x]) - (v[id_p2_x] - v[id_m2_x])) * _dx * _twle;
#endif // DIM_Y
#if DIM_Z
    Vde[dwcx][id] = (_DF(8.0) * (w[id_p1_x] - w[id_m1_x]) - (w[id_p2_x] - w[id_m2_x])) * _dx * _twle;
#endif // DIM_Z
#endif // end DIM_X

#if DIM_Y
    real_t _dy = bl._dy;
    int id_m1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
    int id_m2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
    int id_p1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
    int id_p2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;

#if DIM_X
    Vde[ducy][id] = (_DF(8.0) * (u[id_p1_y] - u[id_m1_y]) - (u[id_p2_y] - u[id_m2_y])) * _dy * _twle;
#endif // DIM_X
    Vde[dvcy][id] = (_DF(8.0) * (v[id_p1_y] - v[id_m1_y]) - (v[id_p2_y] - v[id_m2_y])) * _dy * _twle;
#if DIM_Z
    Vde[dwcy][id] = (_DF(8.0) * (w[id_p1_y] - w[id_m1_y]) - (w[id_p2_y] - w[id_m2_y])) * _dy * _twle;
#endif // DIM_Z

#endif // end DIM_Y

#if DIM_Z
    real_t _dz = bl._dz;
    int id_m1_z = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
    int id_m2_z = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
    int id_p1_z = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
    int id_p2_z = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;

#if DIM_X
    Vde[ducz][id] = (_DF(8.0) * (u[id_p1_z] - u[id_m1_z]) - (u[id_p2_z] - u[id_m2_z])) * _dz * _twle;
#endif // DIM_X
#if DIM_Y
    Vde[dvcz][id] = (_DF(8.0) * (v[id_p1_z] - v[id_m1_z]) - (v[id_p2_z] - v[id_m2_z])) * _dz * _twle;
#endif // DIM_Y
    Vde[dwcz][id] = (_DF(8.0) * (w[id_p1_z] - w[id_m1_z]) - (w[id_p2_z] - w[id_m2_z])) * _dz * _twle;

#endif // end DIM_Z
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
    MARCO_DOMAIN_GHOST();
    int id = Xmax * Ymax * k + Xmax * j + i;
    real_t *Vde_x[] = {Vde[ducy], Vde[ducz], Vde[dvcy], Vde[dwcz]};
#if DIM_Y
    if (j >= Ymax)
            return;
#endif
#if DIM_Z
    if (k >= Zmax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_X + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
            for (int n = 0; n < 4; n++)
            Vde_x[n][id] = Vde_x[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
            for (int n = 0; n < 4; n++)
            Vde_x[n][id] = Vde_x[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde_x[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
            for (int n = 0; n < 4; n++)
            Vde_x[n][id] = Vde_x[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_X + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
            for (int n = 0; n < 4; n++)
            Vde_x[n][id] = -Vde_x[n][target_id];
    }
    break;
#ifdef USE_MPI
    case BC_COPY:
            break;
    case BC_UNDEFINED:
            break;
#endif
    }
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
    MARCO_DOMAIN_GHOST();
    int id = Xmax * Ymax * k + Xmax * j + i;
    real_t *Vde_y[4] = {Vde[dvcx], Vde[dvcz], Vde[ducx], Vde[dwcz]};
#if DIM_X
    if (i >= Xmax)
            return;
#endif
#if DIM_Z
    if (k >= Zmax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
            for (int n = 0; n < 4; n++)
            Vde_y[n][id] = Vde_y[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
            for (int n = 0; n < 4; n++)
            Vde_y[n][id] = Vde_y[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde_y[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
            for (int n = 0; n < 4; n++)
            Vde_y[n][id] = Vde_y[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
            int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
            for (int n = 0; n < 4; n++)
            Vde_y[n][id] = -Vde_y[n][target_id];
    }
    break;
#ifdef USE_MPI
    case BC_COPY:
            break;
    case BC_UNDEFINED:
            break;
#endif
    }
}

extern SYCL_EXTERNAL void CenterDerivativeBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
    MARCO_DOMAIN_GHOST();
    int id = Xmax * Ymax * k + Xmax * j + i;
    real_t *Vde_z[4] = {Vde[dwcx], Vde[dwcy], Vde[ducx], Vde[dvcy]};
#if DIM_X
    if (i >= Xmax)
            return;
#endif
#if DIM_Y
    if (j >= Ymax)
            return;
#endif

    switch (BC)
    {
    case Symmetry:
    {
            int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
            int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde_z[n][id] = Vde_z[n][target_id];
    }
    break;

    case Periodic:
    {
            int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde_z[n][id] = Vde_z[n][target_id];
    }
    break;

    case Inflow:
            for (int n = 0; n < 4; n++)
            Vde_z[n][id] = real_t(0.0f);
            break;

    case Outflow:
    {
            int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde_z[n][id] = Vde_z[n][target_id];
    }
    break;

    case Wall:
    {
            int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
            int target_id = Xmax * Ymax * (k - offset) + Xmax * j + i;
            for (int n = 0; n < 4; n++)
            Vde_z[n][id] = -Vde_z[n][target_id];
    }
    break;
#ifdef USE_MPI
    case BC_COPY:
            break;
    case BC_UNDEFINED:
            break;
#endif
    }
}

extern SYCL_EXTERNAL void Gettransport_coeff_aver(int i, int j, int k, Block bl, Thermal thermal, real_t *viscosity_aver, real_t *thermal_conduct_aver,
                                                  real_t *Dkm_aver, real_t *y, real_t *hi, real_t *rho, real_t *p, real_t *T, real_t *Ertemp1, real_t *Ertemp2)
{
#ifdef DIM_X
    if (i >= bl.Xmax)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (j >= bl.Ymax)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (k >= bl.Zmax)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    // get mole fraction of each specie
    real_t X[NUM_SPECIES] = {_DF(0.0)}; //, yi[NUM_SPECIES] = {_DF(0.0)};
#ifdef Diffu
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
            hi[ii + NUM_SPECIES * id] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id], thermal.Ri[ii], ii);
#endif // end Diffu
    real_t *yi = &(y[NUM_SPECIES * id]); // get_yi(y, yi, id);
    real_t C_total = get_xi(X, yi, thermal._Wi, rho[id]);
    //  real_t *temp = &(Dkm_aver[NUM_SPECIES * id]);
    //  real_t *temp = &(hi[NUM_SPECIES * id]);
    Get_transport_coeff_aver(i, j, k, thermal, &(Dkm_aver[NUM_SPECIES * id]), viscosity_aver[id], thermal_conduct_aver[id],
                             X, rho[id], p[id], T[id], C_total, &(Ertemp1[NUM_SPECIES * id]), &(Ertemp2[NUM_SPECIES * id]));
}

#if DIM_X
extern SYCL_EXTERNAL void GetWallViscousFluxX(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
                                              real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (j >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (k >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
    int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
    int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
    int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;
    real_t *Ducy = Vde[ducy];
    real_t *Ducz = Vde[ducz];
    real_t *Dvcy = Vde[dvcy];
    real_t *Dwcz = Vde[dwcz];
    real_t _dl = bl._dx;

    MARCO_PREVISCFLUX();
    // real_t F_x_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) * _OT * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

    f_x = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
    f_x += lamada * (_DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) * _sxtn;
#if DIM_Y
    f_y = mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
    f_y += mue * (_DF(9.0) * (Ducy[id_p1] + Ducy[id]) - (Ducy[id_p2] + Ducy[id_m1])) * _sxtn;
#else
    f_y = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
    f_z = mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
    f_z += mue * (_DF(9.0) * (Ducz[id_p1] + Ducz[id]) - (Ducz[id_p2] + Ducz[id_m1])) * _sxtn;
#else
    f_z = _DF(0.0);
#endif // DIM_Z

    u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#if DIM_Y
    v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#else
    v_hlf = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
    w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;
#else
    w_hlf = _DF(0.0);
#endif // DIM_Z

    MARCO_VISCFLUX();

    //     F_x_wall_v[0] = _DF(0.0);
    //     F_x_wall_v[1] = f_x;
    //     F_x_wall_v[2] = f_y;
    //     F_x_wall_v[3] = f_z;
    //     F_x_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

    // #ifdef Heat // Fourier thermal conductivity; // thermal conductivity at wall
    //     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0);
    //     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dx / _DF(24.0);                                                // temperature gradient at wall
    //     F_x_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
    // #endif                                                                                                                              // end Heat
    // #ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    //     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    //     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yix_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //     {
    //             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    //             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    // #ifdef COP
    //             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
    //             Yix_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dx / _DF(24.0); // temperature gradient at wall
    // #else
    //             Yix_wall[l] = _DF(0.0);
    // #endif // end COP
    //     }
    // #ifdef Heat
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             F_x_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yix_wall[l];
    // #endif     // end Heat
    // #ifdef COP // visc flux for cop equations
    //     real_t CorrectTermX = _DF(0.0);
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             CorrectTermX += Dim_wall[l] * Yix_wall[l];
    //     CorrectTermX *= rho_wall;
    //     // ADD Correction Term in X-direction
    //     for (int p = 5; p < Emax; p++)
    //             F_x_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yix_wall[p - 5] - Yi_wall[p - 5] * CorrectTermX;
    // #endif // end COP
    // #endif // end Diffu
    //     for (size_t n = 0; n < Emax; n++)
    //     { // add viscous flux to fluxwall
    //             FluxFw[n + Emax * id] -= F_x_wall_v[n];
    //     }
}
#endif // end DIM_X

#if DIM_Y
extern SYCL_EXTERNAL void GetWallViscousFluxY(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
                                              real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (j >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (k >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
    int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
    int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
    int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;
    real_t *Dvcx = Vde[dvcx];
    real_t *Dvcz = Vde[dvcz];
    real_t *Ducx = Vde[ducx];
    real_t *Dwcz = Vde[dwcz];
    real_t _dl = bl._dy;

    MARCO_PREVISCFLUX();
    // // mue at wall
    // real_t F_y_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) * _OT * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

#if DIM_X
    f_x = mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
    f_x += mue * (_DF(9.0) * (Dvcx[id_p1] + Dvcx[id]) - (Dvcx[id_p2] + Dvcx[id_m1])) * _sxtn;
#else
    f_x = _DF(0.0);
#endif // DIM_X
    f_y = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
    f_y += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) * _sxtn;
#if DIM_Z
    f_z = mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
    f_z += mue * (_DF(9.0) * (Dvcz[id_p1] + Dvcz[id]) - (Dvcz[id_p2] + Dvcz[id_m1])) * _sxtn;
#else
    f_z = _DF(0.0);
#endif // DIM_Z

#if DIM_X
    u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#else
    u_hlf = _DF(0.0);
#endif // DIM_X
    v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#if DIM_Z
    w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;
#else
    w_hlf = _DF(0.0);
#endif // DIMZ
    MARCO_VISCFLUX();

    //     F_y_wall_v[0] = _DF(0.0);
    //     F_y_wall_v[1] = f_x;
    //     F_y_wall_v[2] = f_y;
    //     F_y_wall_v[3] = f_z;
    //     F_y_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

    // #ifdef Heat    // Fourier thermal conductivity
    //     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); // thermal conductivity at wall
    //     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dy / _DF(24.0);                                                // temperature gradient at wall
    //     F_y_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
    // #endif                                                                                                                              // end Heat
    // #ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    //     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    //     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiy_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //     {
    //             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    //             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    // #ifdef COP
    //             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
    //             Yiy_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dy / _DF(24.0); // temperature gradient at wal
    // #else
    //             Yiy_wall[l] = _DF(0.0);
    // #endif // end COP
    //     }
    // #ifdef Heat
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             F_y_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiy_wall[l];
    // #endif     // end Heat
    // #ifdef COP // visc flux for cop equations
    //     real_t CorrectTermY = _DF(0.0);
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             CorrectTermY += Dim_wall[l] * Yiy_wall[l];
    //     CorrectTermY *= rho_wall;
    //     // ADD Correction Term in X-direction
    //     for (int p = 5; p < Emax; p++)
    //             F_y_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiy_wall[p - 5] - Yi_wall[p - 5] * CorrectTermY;
    // #endif // end COP
    // #endif // end Diffu
    //     for (size_t n = 0; n < Emax; n++)
    //     { // add viscous flux to fluxwall
    //             FluxGw[n + Emax * id] -= F_y_wall_v[n];
    //     }
}
#endif // end DIM_Y

#if DIM_Z
extern SYCL_EXTERNAL void GetWallViscousFluxZ(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
                                              real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Heat、Diffu viscity in this function
#ifdef DIM_X
    if (i >= bl.X_inner + bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (j >= bl.Y_inner + bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (k >= bl.Z_inner + bl.Bwidth_Z)
            return;
#endif // DIM_Z
    int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
    int id_m1 = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
    int id_m2 = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
    int id_p1 = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
    int id_p2 = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;
    real_t *Dwcx = Vde[dwcx];
    real_t *Dwcy = Vde[dwcy];
    real_t *Ducx = Vde[ducx];
    real_t *Dvcy = Vde[dvcy];
    real_t _dl = bl._dz;

    MARCO_PREVISCFLUX();
    // real_t F_z_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) * _OT * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

#if DIM_X
    f_x = mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
    f_x += mue * (_DF(9.0) * (Dwcx[id_p1] + Dwcx[id]) - (Dwcx[id_p2] + Dwcx[id_m1])) * _sxtn;
#else
    f_x = _DF(0.0);
#endif // DIM_X
#if DIM_Y
    f_y = mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
    f_y += mue * (_DF(9.0) * (Dwcy[id_p1] + Dwcy[id]) - (Dwcy[id_p2] + Dwcy[id_m1])) * _sxtn;
#else
    f_y = _DF(0.0);
#endif
    f_z = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
    f_z += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1])) * _sxtn;

#if DIM_X
    u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#else
    u_hlf = _DF(0.0);
#endif // DIM_X
#if DIM_Y
    v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#else
    v_hlf = _DF(0.0);
#endif // DIM_Y
    w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;

    MARCO_VISCFLUX();

    //     F_z_wall_v[0] = _DF(0.0);
    //     F_z_wall_v[1] = f_x;
    //     F_z_wall_v[2] = f_y;
    //     F_z_wall_v[3] = f_z;
    //     F_z_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

    // #ifdef Heat    // Fourier thermal conductivity
    //     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); // thermal conductivity at wall
    //     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dz / _DF(24.0);                                                // temperature gradient at wall
    //     F_z_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
    // #endif                                                                                                                              // end Heat
    // #ifdef Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
    //     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
    //     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiz_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //     {
    //             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    //             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
    // #ifdef COP
    //             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
    //             Yiz_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dz / _DF(24.0); // temperature gradient at wall
    // #else
    //             Yiz_wall[l] = _DF(0.0);
    // #endif // end COP
    //     }
    // #ifdef Heat
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             F_z_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiz_wall[l];
    // #endif     // end Heat
    // #ifdef COP // visc flux for cop equations
    //     real_t CorrectTermZ = _DF(0.0);
    //     for (int l = 0; l < NUM_SPECIES; l++)
    //             CorrectTermZ += Dim_wall[l] * Yiz_wall[l];
    //     CorrectTermZ *= rho_wall;
    //     // ADD Correction Term in X-direction
    //     for (int p = 5; p < Emax; p++)
    //             F_z_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiz_wall[p - 5] - Yi_wall[p - 5] * CorrectTermZ;
    // #endif // end COP
    // #endif // end Diffu
    //     for (size_t n = 0; n < Emax; n++)
    //     { // add viscous flux to fluxwall
    //             FluxHw[n + Emax * id] -= F_z_wall_v[n];
    //     }
}
#endif // end DIM_Z

#ifdef COP_CHEME
extern SYCL_EXTERNAL void ChemeODEQ2SolverKernel(int i, int j, int k, Block bl, Thermal thermal, Reaction react, real_t *UI, real_t *y, real_t *rho, real_t *T, real_t *e, const real_t dt)
{
    MARCO_DOMAIN();
#ifdef DIM_X
    if (i >= Xmax - bl.Bwidth_X)
            return;
#endif // DIM_X
#ifdef DIM_Y
    if (j >= Ymax - bl.Bwidth_Y)
            return;
#endif // DIM_Y
#ifdef DIM_Z
    if (k >= Zmax - bl.Bwidth_Z)
            return;
#endif // DIM_Z

    int id = Xmax * Ymax * k + Xmax * j + i;

    real_t Kf[NUM_REA], Kb[NUM_REA], U[Emax - NUM_COP], *yi = &(y[NUM_SPECIES * id]);          // yi[NUM_SPECIES],//get_yi(y, yi, id);
    get_KbKf(Kf, Kb, react.Rargus, thermal._Wi, thermal.Hia, thermal.Hib, react.Nu_d_, T[id]); // get_e
    // for (size_t n = 0; n < Emax - NUM_COP; n++)
    // {
    //         U[n] = UI[Emax * id + n];
    // }
    // real_t rho1 = _DF(1.0) / U[0];
    // real_t u = U[1] * rho1;
    // real_t v = U[2] * rho1;
    // real_t w = U[3] * rho1;
    // real_t e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
    Chemeq2(id, thermal, Kf, Kb, react.React_ThirdCoef, react.Rargus, react.Nu_b_, react.Nu_f_, react.Nu_d_, react.third_ind,
            react.reaction_list, react.reactant_list, react.product_list, react.rns, react.rts, react.pls, yi, dt, T[id], rho[id], e[id]);
    // update partial density according to C0
    for (int n = 0; n < NUM_COP; n++)
    {
            // if (bool(sycl::isnan(yi[n])))
            // {
            // yi[n] = _DF(1.0e-20);
            // }
            UI[Emax * id + n + 5] = yi[n] * rho[id];
    }
}
#endif // end COP_CHEME