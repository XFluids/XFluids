#pragma once
#include "global_class.h"
#include "marco.h"
#include "device_func.hpp"

/**
 * @brief  Initialize Fluid states espically primitive quantity;
 * @return void
 */
extern SYCL_EXTERNAL void
InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal *thermal,
                    real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *const *_y, real_t *T)
{
    MARCO_DOMAIN_GHOST();
    real_t dx = bl.dx;
    real_t dy = bl.dy;
    real_t dz = bl.dz;
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
    int id = Xmax * Ymax * k + Xmax * j + i;

    real_t x = DIM_X ? (i - Bwidth_X + bl.myMpiPos_x * (Xmax - Bwidth_X - Bwidth_X)) * dx + _DF(0.5) * dx + bl.Domain_xmin : _DF(0.0);
    real_t y = DIM_Y ? (j - Bwidth_Y + bl.myMpiPos_y * (Ymax - Bwidth_Y - Bwidth_Y)) * dy + _DF(0.5) * dy + bl.Domain_ymin : _DF(0.0);
    real_t z = DIM_Z ? (k - Bwidth_Z + bl.myMpiPos_z * (Zmax - Bwidth_Z - Bwidth_Z)) * dz + _DF(0.5) * dz + bl.Domain_zmin : _DF(0.0);

    real_t d2;
    switch (ini.blast_type)
    {
    case 0:
        d2 = _DF(0.0);
        break;
    case 1:
        d2 = x;
        break;
    case 2:
        d2 = (sycl::pow<real_t>(x - ini.blast_center_x, 2) + sycl::pow<real_t>(y - ini.blast_center_y, 2));
        break;
    case 3:
        d2 = (sycl::pow<real_t>(x - ini.blast_center_x, 2) + sycl::pow<real_t>(y - ini.blast_center_y, 2) + sycl::pow<real_t>(z - ini.blast_center_z, 2));
        break;
    }
#ifdef COP
    real_t dy2;
    real_t copBin = int(ini.cop_radius / bl.dl) * bl.dl;
    real_t copBout = copBin + 2 * bl.dl;
    switch (ini.cop_type)
    { // 可以选择组分不同区域，圆形或类shock-wave
    case 0:
        dy2 = _DF(0.0);
        break;
    case 1:
        dy2 = x;
        break;
    case 2:
        dy2 = (sycl::pow<real_t>(x - ini.cop_center_x, 2) + sycl::pow<real_t>(y - ini.cop_center_y, 2));
        copBin = copBin * copBin;
        copBout = copBout * copBout;
        break;
    case 3: // for 3D shock-bubble interactive
        dy2 = (sycl::pow<real_t>(x - ini.cop_center_x, 2) + sycl::pow<real_t>(y - ini.cop_center_y, 2) + sycl::pow<real_t>(z - ini.cop_center_z, 2));
        copBin = copBin * copBin;
        copBout = copBout * copBout;
        break;
    }
#endif

#if 1 == NumFluid
    if (d2 < ini.blast_center_x) // 1d shock tube case: no bubble // upstream of the shock
    {
        rho[id] = ini.blast_density_in;
        p[id] = ini.blast_pressure_in;
        u[id] = ini.blast_u_in;
        v[id] = ini.blast_v_in;
        w[id] = ini.blast_w_in;
#ifdef COP // to be 1d shock without define React
        for (size_t i = 0; i < NUM_SPECIES; i++)
            _y[i][id] = thermal->species_ratio_out[i];
#endif // end COP
    }
    else
    {
        rho[id] = ini.blast_density_out;
        p[id] = ini.blast_pressure_out;
        u[id] = ini.blast_u_out;
        v[id] = ini.blast_v_out;
        w[id] = ini.blast_w_out;
#ifdef COP
        if (dy2 < copBin)                 //|| dy2 == (n - 1) * (n - 1) * dx * dx
        {                                 // in bubble
            rho[id] = ini.cop_density_in; // 气泡内单独赋值密度以和气泡外区分
            p[id] = ini.cop_pressure_in;  // 气泡内单独赋值压力以和气泡外区分
            for (size_t i = 0; i < NUM_SPECIES; i++)
                _y[i][id] = thermal->species_ratio_in[i];
        }
        else if (dy2 > copBout)
        { // out of bubble
            for (size_t i = 0; i < NUM_SPECIES; i++)
                _y[i][id] = thermal->species_ratio_out[i];
        }
        else
        { // boundary of bubble && shock
            rho[id] = _DF(0.5) * (ini.cop_density_in + ini.blast_density_out);
            p[id] = _DF(0.5) * (ini.cop_pressure_in + ini.blast_pressure_out);
            for (size_t i = 0; i < NUM_SPECIES; i++)
                _y[i][id] = _DF(0.5) * (thermal->species_ratio_in[i] + thermal->species_ratio_out[i]);
        }
#endif // end COP
    }
#endif // 2==NumFluid
}

/**
 * @brief  Initialize conservative quantity;
 * @return void
 */
extern SYCL_EXTERNAL void InitialUFKernel(int i, int j, int k, Block bl, MaterialProperty material, Thermal *thermal, real_t *U, real_t *U1, real_t *LU,
                                          real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                                          real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *const *_y, real_t *T, real_t *H, real_t *c)
{
    MARCO_DOMAIN_GHOST();
    real_t dx = bl.dx;
    real_t dy = bl.dy;
    real_t dz = bl.dz;
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
    int id = Xmax * Ymax * k + Xmax * j + i;

    real_t x = DIM_X ? (i - Bwidth_X + bl.myMpiPos_x * (Xmax - Bwidth_X - Bwidth_X)) * dx + _DF(0.5) * dx + bl.Domain_xmin : _DF(0.0);
    real_t y = DIM_Y ? (j - Bwidth_Y + bl.myMpiPos_y * (Ymax - Bwidth_Y - Bwidth_Y)) * dy + _DF(0.5) * dy + bl.Domain_ymin : _DF(0.0);
    real_t z = DIM_Z ? (k - Bwidth_Z + bl.myMpiPos_z * (Zmax - Bwidth_Z - Bwidth_Z)) * dz + _DF(0.5) * dz + bl.Domain_zmin : _DF(0.0);

    // Ini yi
    real_t yi[NUM_SPECIES];
    get_yi(_y, yi, id);

    // // TODO: for debug
    // //  1d debug set
    // x
    //  p[id] = 101325.0 + (32000.0 - 101325.0) * (1 - sycl::exp(-(x - 0.45) * (x - 0.45))); //* (Length - fabs(x));
    //  u[id] = 100.0 * sycl::fabs<real_t>(x);                                               // sycl::sin<real_t>(x * 0.5 * M_PI);
    //  T[id] = 1350.0 + (320.0 - 1350.0) * (1 - sycl::exp(-(x - 0.35) * (x - 0.35) / 0.07 / 0.07));
    // y
    //  p[id] = 101325.0 + (32000.0 - 101325.0) * (1 - sycl::exp(-(y - 0.45) * (y - 0.45))); //* (Length - fabs(x));
    //  v[id] = 100.0 * sycl::fabs<real_t>(y);                                               // sycl::sin<real_t>(x * 0.5 * M_PI);
    //  T[id] = 1350.0 + (320.0 - 1350.0) * (1 - sycl::exp(-(y - 0.35) * (y - 0.35) / 0.07 / 0.07));
    //  rho[id] = p[id] / R / T[id];

    // // GUASS-WAVE
    // p[id] = 101325.0;
    // u[id] = 0.0;
    // v[id] = 0.0;
    // w[id] = 0.0;
    // T[id] = 1350.0 + (320.0 - 1350.0) * (1 - 0.5 * sycl::exp<real_t>(-(x - 0.5) / 0.05 * (x - 0.5) / 0.05)); // - (y - 0.5) / 0.05 * (y - 0.5) / 0.05
    // rho[id] = p[id] / R / T[id];

    // // 1D multicomponent insert shock tube
    // // x
    // T[id] = x < 0.05 ? 400 : 1200;
    // p[id] = x < 0.05 ? 8000 : 80000;
    /// // y
    // T[id] = y < 0.05 ? 400 : 1200;
    // p[id] = y < 0.05 ? 8000 : 80000;
    // // z
    // T[id] = z < 0.05 ? 400 : 1200;
    // p[id] = z < 0.05 ? 8000 : 80000;
    // rho[id] = p[id] / R / T[id];

    // // 1D reactive shock tube
    // // x
    rho[id] = x < 0.06 ? 0.072 : 0.18075;
    u[id] = x < 0.06 ? 0.0 : -487.34;
    p[id] = x < 0.06 ? 7173 : 35594;
    // y
    // rho[id] = y < 0.06 ? 0.072 : 0.18075;
    // v[id] = y < 0.06 ? 0.0 : -487.34;
    // p[id] = y < 0.06 ? 7173 : 35594;
    // // z
    // rho[id] = z < 0.06 ? 0.072 : 0.18075;
    // w[id] = z < 0.06 ? 0.0 : -487.34;
    // p[id] = z < 0.06 ? 7173 : 35594;
    // T[id] = p[id] / rho[id] / R; //

    // // 2D Riemann problem
    // if (y > 0.5)
    // { // dimensionlize ini settings from the non-dimensional settings from the Ref
    //     if (x < 0.5)
    //     {
    //         // case 3 // left upper
    //         p[id] = 0.3 * 160000;
    //         rho[id] = 0.5323;
    //     }
    //     else
    //     {
    //         // case 3 // right upper
    //         p[id] = 1.5 * 160000;
    //         rho[id] = 1.5;
    //     }
    // }
    // else
    // {
    //     if (x < 0.5)
    //     {
    //         // case 3 // left lower
    //         p[id] = 0.029 * 160000;
    //         rho[id] = 0.138;
    //     }
    //     else
    //     {
    //         // case 3 // right lower
    //         p[id] = 0.3 * 160000;
    //         rho[id] = 0.5323;
    //     }
    // }

    // if (y > 0.5)
    // {
    //     if (x < 0.5)
    //     {
    //         // case 3 // left upper
    //         u[id] = 1.206 * sycl::sqrt<real_t>(160000);
    //         v[id] = 0.0 * sycl::sqrt<real_t>(160000);
    //     }
    //     else
    //     {
    //         // case 3 // right upper
    //         u[id] = 0.0 * sycl::sqrt<real_t>(160000);
    //         v[id] = 0.0 * sycl::sqrt<real_t>(160000);
    //     }
    // }
    // else
    // {
    //     if (x < 0.5)
    //     {
    //         // case 3 // left lower
    //         u[id] = 1.206 * sycl::sqrt<real_t>(160000);
    //         v[id] = 1.206 * sycl::sqrt<real_t>(160000);
    //     }
    //     else
    //     {
    //         // case 3 // right lower
    //         u[id] = 0.0 * sycl::sqrt<real_t>(160000);
    //         v[id] = 1.206 * sycl::sqrt<real_t>(160000);
    //     }
    // }

    // // 2D under-expanded jet
    // if (i <= 3)
    // {
    //     if (-0.015 < y && y < 0.015)
    //     {
    //         p[id] = 10.0 * 101325.0;
    //         T[id] = 1000.0;
    //         yi[0] = 0.0087;
    //         yi[1] = 0.2329;
    //         yi[2] = 0.7584;
    //     }
    //     else if (-0.015 * 25 < y && y < 0.015 * 25)
    //     {
    //         p[id] = 1.0 * 101325.0;
    //         T[id] = 300.0;
    //         yi[0] = 0.0;
    //         yi[1] = 0.233;
    //         yi[2] = 0.767;
    //     }
    //     else
    //     {
    //         p[id] = 1.0 * 101325.0;
    //         T[id] = 300.0;
    //         yi[0] = 0.0;
    //         yi[1] = 0.233;
    //         yi[2] = 0.767;
    //     }
    // }
    // else
    // {
    //     p[id] = 1.0 * 101325.0;
    //     T[id] = 300.0;
    //     yi[0] = 0.0;
    //     yi[1] = 0.233;
    //     yi[2] = 0.767;
    // }

    // Get R of mixture
    real_t R = get_CopR(thermal->species_chara, yi);
    T[id] = p[id] / R / rho[id]; // rho[id] = p[id] / R / T[id]; //
    real_t Gamma_m = get_CopGamma(thermal, yi, T[id]);
    c[id] = sqrt(p[id] / rho[id] * Gamma_m);

    // if (i <= 3)
    // {
    //     if (-0.015 < y && y < 0.015)
    //     {
    //         u[id] = c[id];
    //     }
    //     else if (-0.015 * 25 < y && y < 0.015 * 25)
    //     {
    //         u[id] = 0.0575 * c[id];
    //     }
    //     else
    //     {
    //         u[id] = 0.0 * c[id];
    //     }
    // }
    // else
    // {
    //     u[id] = 0.0 * c[id];
    // }

    // U[4] of mixture differ from pure gas
    real_t h = get_Coph(thermal, yi, T[id]);
    U[Emax * id + 4] = rho[id] * (h + _DF(0.5) * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id])) - p[id];
#if 1 != NumFluid
    //  for both singlephase && multiphase
    c[id] = material.Mtrl_ind == 0 ? sqrt(material.Gamma * p[id] / rho[id]) : sqrt(material.Gamma * (p[id] + material.B - material.A) / rho[id]);
    if (material.Mtrl_ind == 0)
        U[Emax * id + 4] = p[id] / (material.Gamma - 1.0) + 0.5 * rho[id] * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id]);
    else
        U[Emax * id + 4] = (p[id] + material.Gamma * (material.B - material.A)) / (material.Gamma - 1.0) + 0.5 * rho[id] * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id]);
#endif // COP
    H[id] = (U[Emax * id + 4] + p[id]) / rho[id];
    // initial U[0]-U[3]
    U[Emax * id + 0] = rho[id];
    U[Emax * id + 1] = rho[id] * u[id];
    U[Emax * id + 2] = rho[id] * v[id];
    U[Emax * id + 3] = rho[id] * w[id];

    // initial flux terms F, G, H
    FluxF[Emax * id + 0] = U[Emax * id + 1];
    FluxF[Emax * id + 1] = U[Emax * id + 1] * u[id] + p[id];
    FluxF[Emax * id + 2] = U[Emax * id + 1] * v[id];
    FluxF[Emax * id + 3] = U[Emax * id + 1] * w[id];
    FluxF[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * u[id];

    FluxG[Emax * id + 0] = U[Emax * id + 2];
    FluxG[Emax * id + 1] = U[Emax * id + 2] * u[id];
    FluxG[Emax * id + 2] = U[Emax * id + 2] * v[id] + p[id];
    FluxG[Emax * id + 3] = U[Emax * id + 2] * w[id];
    FluxG[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * v[id];

    FluxH[Emax * id + 0] = U[Emax * id + 3];
    FluxH[Emax * id + 1] = U[Emax * id + 3] * u[id];
    FluxH[Emax * id + 2] = U[Emax * id + 3] * v[id];
    FluxH[Emax * id + 3] = U[Emax * id + 3] * w[id] + p[id];
    FluxH[Emax * id + 4] = (U[Emax * id + 4] + p[id]) * w[id];

#ifdef COP
    for (size_t ii = 5; ii < Emax; ii++)
    { // equations of species
        U[Emax * id + ii] = rho[id] * yi[ii - 5];
        FluxF[Emax * id + ii] = rho[id] * u[id] * yi[ii - 5];
        FluxG[Emax * id + ii] = rho[id] * v[id] * yi[ii - 5];
        FluxH[Emax * id + ii] = rho[id] * w[id] * yi[ii - 5];
    }
#endif // end COP

#if NumFluid != 1
    real_t fraction = material.Rgn_ind > 0.5 ? vof[id] : 1.0 - vof[id];
#endif
    // give intial value for the interval matrixes
    for (int n = 0; n < Emax; n++)
    {
        LU[Emax * id + n] = _DF(0.0);         // incremental of one time step
        U1[Emax * id + n] = U[Emax * id + n]; // intermediate conwervatives
        FluxFw[Emax * id + n] = _DF(0.0);     // numerical flux F
        FluxGw[Emax * id + n] = _DF(0.0);     // numerical flux G
        FluxHw[Emax * id + n] = _DF(0.0);     // numerical flux H
#if NumFluid != 1
        CnsrvU[Emax * id + n] = U[Emax * id + n] * fraction;
        CnsrvU1[Emax * id + n] = CnsrvU[Emax * id + n];
#endif // NumFluid
    }
}

#if DIM_X
extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *const *y, real_t *T, real_t *H)
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

#ifdef COP
    MARCO_COPC2();
#else
    MARCO_NOCOPC2();
#endif

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
    RoeAverage_x(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

    // // construct the right value & the left value scalar equations by characteristic reduction
    // // at i+1/2 in x direction
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER == 5
    MARCO_FLUXWALL_WENO5(i + m, j, k, i + m, j, k);
#endif
    // real_t de_fw[Emax];
    // get_Array(Fwall, de_fw, Emax, id_l);
    // real_t de_fx[Emax];
}
#endif // end DIM_X

#if DIM_Y
extern SYCL_EXTERNAL void ReconstructFluxY(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *const *y, real_t *T, real_t *H)
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

#ifdef COP
    MARCO_COPC2();
#else
    MARCO_NOCOPC2();
#endif

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
    RoeAverage_y(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

    // // construct the right value & the left value scalar equations by characteristic reduction
    // // at i+1/2 in x direction
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER == 5
    MARCO_FLUXWALL_WENO5(i, j + m, k, i, j + m, k);
#endif
    // real_t ug[10], gg[10], pp[10], mm[10], g_flux, _p[Emax][Emax];
    // for(int n=0; n<Emax; n++){
    //     real_t eigen_local_max = _DF(0.0);

    //     for(int m=-2; m<=3; m++){
    //         int id_local = Xmax*Ymax*k + Xmax*(j + m) + i;
    //         eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local + n])); // local lax-friedrichs
    //     }

    // 	for(int m=j-3; m<=j+4; m++){	// 3rd oder and can be modified
    //         int id_local = Xmax * Ymax * k + Xmax * m + i;

    //         ug[m - j + 3] = _DF(0.0);
    //         gg[m - j + 3] = _DF(0.0);

    //         for (int n1 = 0; n1 < Emax; n1++)
    //         {
    //             ug[m - j + 3] = ug[m - j + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];
    //             gg[m - j + 3] = gg[m - j + 3] + Fy[Emax * id_local + n1] * eigen_l[n][n1];
    //         }
    //         // for local speed
    //         pp[m - j + 3] = _DF(0.5) * (gg[m - j + 3] + eigen_local_max * ug[m - j + 3]);
    //         mm[m - j + 3] = _DF(0.5) * (gg[m - j + 3] - eigen_local_max * ug[m - j + 3]);
    //     }
    // 	// calculate the scalar numerical flux at y direction
    //     g_flux = (weno5old_P(&pp[3], dy) + weno5old_M(&mm[3], dy));
    //     // get Gp
    //     for (int n1 = 0; n1 < Emax; n1++)
    //         _p[n][n1] = g_flux * eigen_r[n1][n];
    // }
    // // reconstruction the G-flux terms
    // for(int n=0; n<Emax; n++){
    //     real_t fluxy = _DF(0.0);
    //     for (int n1 = 0; n1 < Emax; n1++)
    //     {
    //         fluxy += _p[n1][n];
    //     }
    //     Fywall[Emax*id_l+n] = fluxy;
    // }

    // real_t de_fw[Emax];
    // get_Array(Fwall, de_fw, Emax, id_l);
    // real_t de_fx[Emax];
}
#endif // end DIM_Y

#if DIM_Z
extern SYCL_EXTERNAL void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *Fl, real_t *Fwall, real_t *eigen_local,
                                           real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *const *y, real_t *T, real_t *H)
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

#ifdef COP
    MARCO_COPC2();
#else
    MARCO_NOCOPC2();
#endif

    real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
    RoeAverage_z(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

    // // construct the right value & the left value scalar equations by characteristic reduction
    // // at i+1/2 in x direction
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER == 5
    MARCO_FLUXWALL_WENO5(i, j, k + m, i, j, k + m);
#endif
    // real_t uh[10], hh[10], pp[10], mm[10], real_t h_flux, _p[Emax][Emax];

    // //construct the right value & the left value scalar equations by characteristic reduction
    // // at k+1/2 in z direction
    // for(int n=0; n<Emax; n++){
    //     real_t eigen_local_max = _DF(0.0);

    //     for(int m=-2; m<=3; m++){
    //         int id_local = Xmax*Ymax*(k + m) + Xmax*j + i;
    //         eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local + n])); // local lax-friedrichs
    //     }
    // 	for(int m=k-3; m<=k+4; m++){
    //         int id_local = Xmax*Ymax*m + Xmax*j + i;
    //         uh[m - k + 3] = _DF(0.0);
    //         hh[m - k + 3] = _DF(0.0);

    //         for (int n1 = 0; n1 < Emax; n1++)
    //         {
    //             uh[m - k + 3] = uh[m - k + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];
    //             hh[m - k + 3] = hh[m - k + 3] + Fz[Emax * id_local + n1] * eigen_l[n][n1];
    //         }
    //         // for local speed
    //         pp[m - k + 3] = _DF(0.5) * (hh[m - k + 3] + eigen_local_max * uh[m - k + 3]);
    //         mm[m - k + 3] = _DF(0.5) * (hh[m - k + 3] - eigen_local_max * uh[m - k + 3]);
    //     }
    // 	// calculate the scalar numerical flux at y direction
    //     h_flux = (weno5old_P(&pp[3], dz) + weno5old_M(&mm[3], dz));

    //     // get Gp
    //     for (int n1 = 0; n1 < Emax; n1++)
    //         _p[n][n1] = h_flux*eigen_r[n1][n];
    // }
    // // reconstruction the H-flux terms
    // for(int n=0; n<Emax; n++){
    //     real_t fluxz = _DF(0.0);
    //     for (int n1 = 0; n1 < Emax; n1++)
    //     {
    //         fluxz +=  _p[n1][n];
    //     }
    //     Fzwall[Emax*id_l+n]  = fluxz;
    // }

    real_t de_fw[Emax];
    get_Array(Fwall, de_fw, Emax, id_l);
    real_t de_fx[Emax];
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
#if SCHEME_ORDER == 5
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
    LU0 += (FluxFw[Emax * id_im + n] - FluxFw[Emax * id + n]) / bl.dx;
#endif
#if DIM_Y
    LU0 += (FluxGw[Emax * id_jm + n] - FluxGw[Emax * id + n]) / bl.dy;
#endif
#if DIM_Z
    LU0 += (FluxHw[Emax * id_km + n] - FluxHw[Emax * id + n]) / bl.dz;
#endif
    LU[Emax * id + n] = LU0;
    }

    // real_t de_LU[Emax];
    // get_Array(LU, de_LU, Emax, id);
}

extern SYCL_EXTERNAL void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal *thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
                                                 real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *const *_y,
                                                 real_t *gamma, real_t *T, real_t const Gamma, bool *error, const sycl::stream &stream_ct1)
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

    real_t *U = &(UI[Emax * id]), yi[NUM_SPECIES];

    *error = GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], gamma[id], T[id], thermal, yi);
    if (*error)
        return;

    real_t *Fx = &(FluxF[Emax * id]);
    real_t *Fy = &(FluxG[Emax * id]);
    real_t *Fz = &(FluxH[Emax * id]);

    GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);

    for (size_t n = 0; n < NUM_SPECIES; n++)
        _y[n][id] = yi[n];

    // real_t de_fx[Emax], de_fy[Emax], de_fz[Emax];
    // get_Array(FluxF, de_fx, Emax, id);
    // get_Array(FluxG, de_fy, Emax, id);
    // get_Array(FluxH, de_fz, Emax, id);
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
            U[Emax * id + n] = (U[Emax * id + n] + _DF(2.0) * U1[Emax * id + n] + _DF(2.0) * dt * LU[Emax * id + n]) / _DF(3.0);
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
    int tid = abs(Bwidth_Xset) * Ymax * k + abs(Bwidth_Xset) * j + (i - index_offset);
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
    int tid = Xmax * abs(Bwidth_Yset) * k + Xmax * (j - index_offset) + i;
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

#ifdef Visc
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

#if DIM_X
    real_t dx = bl.dx;
    int id_m1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
    int id_m2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
    int id_p1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
    int id_p2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;

    Vde[ducx][id] = (_DF(8.0) * (u[id_p1_x] - u[id_m1_x]) - (u[id_p2_x] - u[id_m2_x])) / dx / _DF(12.0);
    Vde[dvcx][id] = DIM_Y ? (_DF(8.0) * (v[id_p1_x] - v[id_m1_x]) - (v[id_p2_x] - v[id_m2_x])) / dx / _DF(12.0) : _DF(0.0);
    Vde[dwcx][id] = DIM_Z ? (_DF(8.0) * (w[id_p1_x] - w[id_m1_x]) - (w[id_p2_x] - w[id_m2_x])) / dx / _DF(12.0) : _DF(0.0);
#else
    Vde[ducx][id] = _DF(0.0);
    Vde[dvcx][id] = _DF(0.0);
    Vde[dwcx][id] = _DF(0.0);
#endif // end DIM_X
#if DIM_Y
    real_t dy = bl.dy;
    int id_m1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
    int id_m2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
    int id_p1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
    int id_p2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;

    Vde[ducy][id] = DIM_X ? (_DF(8.0) * (u[id_p1_y] - u[id_m1_y]) - (u[id_p2_y] - u[id_m2_y])) / dy / _DF(12.0) : _DF(0.0);
    Vde[dvcy][id] = (_DF(8.0) * (v[id_p1_y] - v[id_m1_y]) - (v[id_p2_y] - v[id_m2_y])) / dy / _DF(12.0);
    Vde[dwcy][id] = DIM_Z ? (_DF(8.0) * (w[id_p1_y] - w[id_m1_y]) - (w[id_p2_y] - w[id_m2_y])) / dy / _DF(12.0) : _DF(0.0);
#else
    Vde[ducy][id] = _DF(0.0);
    Vde[dvcy][id] = _DF(0.0);
    Vde[dwcy][id] = _DF(0.0);
#endif // end DIM_Y
#if DIM_Z
    real_t dz = bl.dz;
    int id_m1_z = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
    int id_m2_z = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
    int id_p1_z = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
    int id_p2_z = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;

    Vde[ducz][id] = DIM_X ? (_DF(8.0) * (u[id_p1_z] - u[id_m1_z]) - (u[id_p2_z] - u[id_m2_z])) / dz / _DF(12.0) : _DF(0.0);
    Vde[dvcz][id] = DIM_Y ? (_DF(8.0) * (v[id_p1_z] - v[id_m1_z]) - (v[id_p2_z] - v[id_m2_z])) / dz / _DF(12.0) : _DF(0.0);
    Vde[dwcz][id] = (_DF(8.0) * (w[id_p1_z] - w[id_m1_z]) - (w[id_p2_z] - w[id_m2_z])) / dz / _DF(12.0);
#else
    Vde[ducz][id] = _DF(0.0);
    Vde[dvcz][id] = _DF(0.0);
    Vde[dwcz][id] = _DF(0.0);
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

extern SYCL_EXTERNAL void Gettransport_coeff_aver(int i, int j, int k, Block bl, Thermal *thermal, real_t *viscosity_aver, real_t *thermal_conduct_aver,
                                                  real_t *Dkm_aver, real_t *const *y, real_t *hi, real_t *rho, real_t *p, real_t *T)
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
    real_t X[NUM_SPECIES] = {_DF(0.0)}, yi[NUM_SPECIES] = {_DF(0.0)};
#ifdef Diffu
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
            hi[ii + NUM_SPECIES * id] = get_Enthalpy(thermal->Hia, thermal->Hib, T[id], thermal->Ri[ii], ii);
#endif // end Diffu
    get_yi(y, yi, id);
    real_t C_total = get_xi(X, yi, thermal->Wi, rho[id]);
    //  real_t *temp = &(Dkm_aver[NUM_SPECIES * id]);
    //  real_t *temp = &(hi[NUM_SPECIES * id]);
    Get_transport_coeff_aver(thermal, &(Dkm_aver[NUM_SPECIES * id]), viscosity_aver[id], thermal_conduct_aver[id], X, rho[id], p[id], T[id], C_total);
}

#if DIM_X
extern SYCL_EXTERNAL void GetWallViscousFluxX(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde)
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
    real_t dl = bl.dx;

    MARCO_PREVISCFLUX();
    // real_t F_x_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

    f_x = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dl / _DF(24.0);
    f_y = DIM_Y ? mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dl / _DF(24.0) : _DF(0.0);
    f_z = DIM_Z ? mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dl / _DF(24.0) : _DF(0.0);

    f_x += lamada * (_DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) / _DF(16.0);
    f_y += DIM_Y ? mue * (_DF(9.0) * (Ducy[id_p1] + Ducy[id]) - (Ducy[id_p2] + Ducy[id_m1])) / _DF(16.0) : _DF(0.0);
    f_z += DIM_Z ? mue * (_DF(9.0) * (Ducz[id_p1] + Ducz[id]) - (Ducz[id_p2] + Ducz[id_m1])) / _DF(16.0) : _DF(0.0);

    u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0);
    v_hlf = DIM_Y ? (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0) : _DF(0.0);
    w_hlf = DIM_Z ? (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0) : _DF(0.0);

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
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde)
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
    real_t dl = bl.dy;

    MARCO_PREVISCFLUX();
    // // mue at wall
    // real_t F_y_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

    f_x = DIM_X ? mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dl / _DF(24.0) : _DF(0.0);
    f_y = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dl / _DF(24.0);
    f_z = DIM_Z ? mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dl / _DF(24.0) : _DF(0.0);

    f_x += DIM_X ? mue * (_DF(9.0) * (Dvcx[id_p1] + Dvcx[id]) - (Dvcx[id_p2] + Dvcx[id_m1])) / _DF(16.0) : _DF(0.0);
    f_y += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) / _DF(16.0);
    f_z += DIM_Z ? mue * (_DF(9.0) * (Dvcz[id_p1] + Dvcz[id]) - (Dvcz[id_p2] + Dvcz[id_m1])) / _DF(16.0) : _DF(0.0);

    u_hlf = DIM_X ? (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0) : _DF(0.0);
    v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0);
    w_hlf = DIM_Z ? (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0) : _DF(0.0);

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
                                              real_t *T, real_t *rho, real_t *hi, real_t *const *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde)
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
    real_t dl = bl.dz;

    MARCO_PREVISCFLUX();
    // real_t F_z_wall_v[Emax];
    // real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
    // real_t lamada = -_DF(2.0) / _DF(3.0) * mue;
    // real_t f_x, f_y, f_z;
    // real_t u_hlf, v_hlf, w_hlf;

    f_x = DIM_X ? mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) / dl / _DF(24.0) : _DF(0.0);
    f_y = DIM_Y ? mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) / dl / _DF(24.0) : _DF(0.0);
    f_z = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) / dl / _DF(24.0);

    f_x += DIM_X ? mue * (_DF(9.0) * (Dwcx[id_p1] + Dwcx[id]) - (Dwcx[id_p2] + Dwcx[id_m1])) / _DF(16.0) : _DF(0.0);
    f_y += DIM_Y ? mue * (_DF(9.0) * (Dwcy[id_p1] + Dwcy[id]) - (Dwcy[id_p2] + Dwcy[id_m1])) / _DF(16.0) : _DF(0.0);
    f_z += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1])) / _DF(16.0);
    u_hlf = DIM_X ? (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) / _DF(16.0) : _DF(0.0);
    v_hlf = DIM_Y ? (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) / _DF(16.0) : _DF(0.0);
    w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) / _DF(16.0);

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
#endif // Visc

#ifdef COP_CHEME
extern SYCL_EXTERNAL void ChemeODEQ2SolverKernel(int i, int j, int k, Block bl, Thermal *thermal, Reaction *react, real_t *UI, real_t *const *y, real_t *rho, real_t *T, const real_t dt)
{
    MARCO_DOMAIN();
    int id = Xmax * Ymax * k + Xmax * j + i;

    real_t yi[NUM_SPECIES], Kf[NUM_REA], Kb[NUM_REA], U[Emax - NUM_COP];
    get_yi(y, yi, id);
    get_KbKf(Kf, Kb, react->Rargus, thermal->species_chara, thermal->Hia, thermal->Hib, react->Nu_d_, T[id]); // get_e
    for (size_t n = 0; n < Emax - NUM_COP; n++)
    {
            U[n] = UI[Emax * id + n];
    }
    real_t rho1 = _DF(1.0) / U[0];
    real_t u = U[1] * rho1;
    real_t v = U[2] * rho1;
    real_t w = U[3] * rho1;
    real_t e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
    Chemeq2(thermal, Kf, Kb, react->React_ThirdCoef, react->Rargus, react->Nu_b_, react->Nu_f_, react->Nu_d_, react->third_ind,
            react->reaction_list, react->reactant_list, react->product_list, react->rns, react->rts, react->pls, yi, dt, T[id], rho[id], e);
    // update partial density according to C0
    for (int n = 0; n < NUM_COP; n++)
    { // NOTE: related with yi[n]
            UI[Emax * id + n + 5] = yi[n] * rho[id];
    }
}
#endif // end COP_CHEME
