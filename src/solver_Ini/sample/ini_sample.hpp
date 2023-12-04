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
    p[id] = _DF(0.0);
    T[id] = _DF(0.0);
    u[id] = _DF(0.0);
    v[id] = _DF(0.0);
    w[id] = _DF(0.0);
    real_t *yi = &(_y[Emax * id]);
    for (size_t i = 0; i < NUM_SPECIES; i++)
        yi[i] = _DF(0.0);
    for (size_t i = 0; i < count; i++)
    {
        /* code */
    }
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
    real_t x = bl.DimX ? (i - Bwidth_X + bl.myMpiPos_x * (Xmax - Bwidth_X - Bwidth_X)) * dx + _DF(0.5) * dx + bl.Domain_xmin : _DF(0.0);
    real_t y = bl.DimY ? (j - Bwidth_Y + bl.myMpiPos_y * (Ymax - Bwidth_Y - Bwidth_Y)) * dy + _DF(0.5) * dy + bl.Domain_ymin : _DF(0.0);
    real_t z = bl.DimZ ? (k - Bwidth_Z + bl.myMpiPos_z * (Zmax - Bwidth_Z - Bwidth_Z)) * dz + _DF(0.5) * dz + bl.Domain_zmin : _DF(0.0);

    rho[id] = _DF(0.0);
    p[id] = _DF(0.0);
    T[id] = _DF(0.0);
    u[id] = _DF(0.0);
    v[id] = _DF(0.0);
    w[id] = _DF(0.0);

    // // GUASS-WAVE
    p[id] = _DF(101325.0);
    real_t tmt = _DF(0.0);

    if (bl.DimX)
        tmt += -(x - _DF(0.025)) / _DF(0.0025) * (x - _DF(0.025)) / _DF(0.0025);

    if (bl.DimY)
        tmt += -(y - _DF(0.025)) / _DF(0.0025) * (y - _DF(0.025)) / _DF(0.0025);

    if (bl.DimZ)
        tmt += -(z - _DF(0.025)) / _DF(0.0025) * (z - _DF(0.025)) / _DF(0.0025);

    real_t fx = _DF(1.0) - _DF(0.5) * sycl::exp(tmt);
    real_t *yi = &(_y[NUM_SPECIES * id]), Yif[4] = {_DF(0.195), _DF(0.591), _DF(0.0), _DF(0.214)}, Yio[4] = {_DF(0.142), _DF(0.758), _DF(0.1), _DF(0.0)};

    T[id] = _DF(1350.0) + (_DF(320.0) - _DF(1350.0)) * fx;
    for (size_t n = 0; n < NUM_SPECIES; n++)
    {
#ifdef DiffuReverse
        yi[NUM_COP - n] = Yio[n] + (Yif[n] - Yio[n]) * fx;
#else
        yi[n] = Yio[n] + (Yif[n] - Yio[n]) * fx;
#endif // end DiffuReverse
    }
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