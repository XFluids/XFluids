#pragma once

#include "sycl_devices.hpp"

/**
 * @brief  Initialize Fluid states espically primitive quantity;
 * @return void
 */
extern void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal thermal,
                                real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T)
{
    MARCO_DOMAIN_GHOST();
    if (i >= Xmax)
        return;
    if (j >= Ymax)
        return;
    if (k >= Zmax)
        return;

    int id = Xmax * Ymax * k + Xmax * j + i;
    real_t dx = bl.dx, dy = bl.dy, dz = bl.dz;
    real_t x = bl.DimX ? (i - Bwidth_X + bl.myMpiPos_x * (Xmax - Bwidth_X - Bwidth_X)) * dx + _DF(0.5) * dx + bl.Domain_xmin : _DF(0.0);
    real_t y = bl.DimY ? (j - Bwidth_Y + bl.myMpiPos_y * (Ymax - Bwidth_Y - Bwidth_Y)) * dy + _DF(0.5) * dy + bl.Domain_ymin : _DF(0.0);
    real_t z = bl.DimZ ? (k - Bwidth_Z + bl.myMpiPos_z * (Zmax - Bwidth_Z - Bwidth_Z)) * dz + _DF(0.5) * dz + bl.Domain_zmin : _DF(0.0);

    rho[id] = _DF(0.0);
    p[id] = _DF(94232.25);
    T[id] = _DF(0.0);
    u[id] = _DF(0.0);
    v[id] = _DF(0.0);
    w[id] = _DF(0.0);

    // // USE Species:H2, O2, O, H, OH, HO2, H2O2, H2O
    real_t Tf = 545.0, To = 1475.0, fx = _DF(1.0);
    real_t *yi = &(_y[NUM_SPECIES * id]), Yio[NUM_SPECIES], Yif[NUM_SPECIES] = {0.05, 0, 0, 0, 0, 0, 0, 0, 1.0};
    if (bl.RSources)
    {
        Yio[0] = 0, Yio[1] = 0.278, Yio[2] = 1.55E-4;
        Yio[3] = 5.6E-7, Yio[4] = 1.83E-3, Yio[5] = 5.1E-6;
        Yio[6] = 2.5E-7, Yio[7] = 0.17, Yio[8] = 1.0;
    }
    else
    {
        Yio[0] = 0, Yio[1] = 0.278, Yio[2] = 0;
        Yio[3] = 0, Yio[4] = 0, Yio[5] = 0;
        Yio[6] = 0, Yio[7] = 0.17, Yio[8] = 1.0;
    }

#if defined(GhostSpecies)
    int Last_SID = NUM_SPECIES - 2;
    yi[Last_SID + 1] = _DF(0.0);
#else
    int Last_SID = NUM_SPECIES - 1;
#endif

    yi[Last_SID] = _DF(1.0);
    if (bl.DimX)
        fx *= sycl::tanh(_DF(2.0) * (x - _DF(0.5) * bl.Domain_length) / _DF(1.44E-4));

    if (bl.DimY)
        fx *= sycl::tanh(_DF(2.0) * (y - _DF(0.5) * bl.Domain_width) / _DF(1.44E-4));

    if (bl.DimZ)
        fx *= sycl::tanh(_DF(2.0) * (z - _DF(0.5) * bl.Domain_height) / _DF(1.44E-4));

    T[id] = _DF(0.5) * (Tf + To + (Tf - To) * fx);
    for (size_t ii = 0; ii < Last_SID; ii++)
        yi[ii] = sycl::max(_DF(0.5) * (Yif[ii] + Yio[ii] + (Yif[ii] - Yio[ii]) * fx), _DF(1.0E-10)), yi[Last_SID] -= yi[ii];
}

/**
 * @brief  Initialize conservative quantity;
 * @return void
 */
extern void InitialUFKernel(int i, int j, int k, Block bl, MaterialProperty material, Thermal thermal, real_t *U, real_t *U1, real_t *LU,
                            real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
                            real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T, real_t *H, real_t *c)
{
    MARCO_DOMAIN_GHOST();
    if (i >= Xmax)
        return;
    if (j >= Ymax)
        return;
    if (k >= Zmax)
        return;

    int id = Xmax * Ymax * k + Xmax * j + i;
    real_t dx = bl.dx, dy = bl.dy, dz = bl.dz;

    real_t *yi = &(_y[NUM_SPECIES * id]);
    // Get R of mixture
    real_t R = get_CopR(thermal._Wi, yi);
    rho[id] = p[id] / R / T[id]; // T[id] = p[id] / R / rho[id];
    real_t Gamma_m = get_CopGamma(thermal, yi, T[id]);
    c[id] = sycl::sqrt(p[id] / rho[id] * Gamma_m);

    // U[4] of mixture differ from pure gas
    real_t h = get_Coph(thermal, yi, T[id]);
    U[Emax * id + 4] = rho[id] * (h + _DF(0.5) * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id])) - p[id];
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

    // give intial value for the interval matrixes
    for (int n = 0; n < Emax; n++)
    {
        LU[Emax * id + n] = _DF(0.0);         // incremental of one time step
        U1[Emax * id + n] = U[Emax * id + n]; // intermediate conwervatives
        FluxFw[Emax * id + n] = _DF(0.0);     // numerical flux F
        FluxGw[Emax * id + n] = _DF(0.0);     // numerical flux G
        FluxHw[Emax * id + n] = _DF(0.0);     // numerical flux H
    }
}