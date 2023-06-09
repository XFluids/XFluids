#pragma once
#include "../../../global_class.h"
#include "../../../device_func.hpp"

/**
 * @brief  Initialize Fluid states espically primitive quantity;
 * @return void
 */
extern SYCL_EXTERNAL void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal thermal,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T)
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
            _y[i][id] = thermal.species_ratio_out[i];
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
                _y[i][id] = thermal.species_ratio_in[i];
        }
        else if (dy2 > copBout)
        { // out of bubble
            for (size_t i = 0; i < NUM_SPECIES; i++)
                _y[i][id] = thermal.species_ratio_out[i];
        }
        else
        { // boundary of bubble && shock
            rho[id] = _DF(0.5) * (ini.cop_density_in + ini.blast_density_out);
            p[id] = _DF(0.5) * (ini.cop_pressure_in + ini.blast_pressure_out);
            for (size_t i = 0; i < NUM_SPECIES; i++)
                _y[i][id] = _DF(0.5) * (thermal.species_ratio_in[i] + thermal.species_ratio_out[i]);
        }
#endif // end COP
    }
#endif // 2==NumFluid
}

/**
 * @brief  Initialize conservative quantity;
 * @return void
 */
extern SYCL_EXTERNAL void InitialUFKernel(int i, int j, int k, Block bl, MaterialProperty material, Thermal thermal, real_t *U, real_t *U1, real_t *LU,
                                          real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                                          real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T, real_t *H, real_t *c)
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
    // p[id] = 101325.0 + (32000.0 - 101325.0) * (1 - sycl::exp(-(x - 0.45) * (x - 0.45))); //* (Length - fabs(x));
    // u[id] = 100.0 * sycl::fabs<real_t>(x);                                               // sycl::sin<real_t>(x * 0.5 * M_PI);
    // T[id] = 1350.0 + (320.0 - 1350.0) * (1 - sycl::exp(-(x - 0.55) * (x - 0.55)));
    // y
    // p[id] = 101325.0 + (32000.0 - 101325.0) * (1 - sycl::exp(-(y - 0.45) * (y - 0.45)));
    // v[id] = 100.0 * sycl::fabs<real_t>(y);
    // T[id] = 1350.0 + (320.0 - 1350.0) * (1 - sycl::exp(-(y - 0.55) * (y - 0.55)));
    // z
    // p[id] = 101325.0 + (32000.0 - 101325.0) * (1 - sycl::exp(-(z - 0.45) * (z - 0.45)));
    // w[id] = 100.0 * sycl::fabs<real_t>(z);
    // T[id] = 1350.0 + (320.0 - 1350.0) * (1 - sycl::exp(-(z - 0.55) * (z - 0.55)));

    // // GUASS-WAVE
    // p[id] = 101325.0;
    // u[id] = 0.0;
    // v[id] = 0.0;
    // w[id] = 0.0;
    // T[id] = 1350.0 + (320.0 - 1350.0) * (1 - 0.5 * sycl::exp<real_t>(-(x - 0.5) / 0.05 * (x - 0.5) / 0.05)); // - (y - 0.5) / 0.05 * (y - 0.5) / 0.05
    // rho[id] = p[id] / R / T[id];

    // // 1D multicomponent insert shock tube
    // // x
    T[id] = x < 0.05 ? 400 : 1200;
    p[id] = x < 0.05 ? 8000 : 80000;
    /// // y
    // T[id] = y < 0.05 ? 400 : 1200;
    // p[id] = y < 0.05 ? 8000 : 80000;
    // // z
    // T[id] = z < 0.05 ? 400 : 1200;
    // p[id] = z < 0.05 ? 8000 : 80000;
    // rho[id] = p[id] / R / T[id];

    // // 1D reactive shock tube
    // // x
    // rho[id] = x < 0.06 ? 0.072 : 0.18075;
    // u[id] = x < 0.06 ? 0.0 : -487.34;
    // p[id] = x < 0.06 ? 7173 : 35594;
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
    real_t R = get_CopR(thermal._Wi, yi);
    rho[id] = p[id] / R / T[id]; // T[id] = p[id] / R / rho[id]; //
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
