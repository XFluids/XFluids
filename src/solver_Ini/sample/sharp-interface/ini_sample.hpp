#pragma once
 
#include "sycl_devices.hpp"

/**
 * @brief  Initialize Fluid states espically primitive quantity;
 * @return void
 */
extern void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal thermal,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T) {}

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

    // // x-dir shock
    T[id] = x < _DF(0.005) ? _DF(1200.0) : _DF(400.0);
    p[id] = x < _DF(0.005) ? _DF(80000.0) : _DF(8000.0);
    u[id] = x < _DF(0.005) ? _DF(750.0) : _DF(0.0);

    real_t dy_ = _DF(0.0), tmp = _DF(0.0);
    // a circular bubble
    if (bl.DimX)
        dy_ += (x - _DF(0.02)) * (x - _DF(0.02));

    if (bl.DimY)
        dy_ += (y - _DF(0.0)) * (y - _DF(0.0));

    if (bl.DimZ)
        dy_ += (z - _DF(0.0)) * (z - _DF(0.0));

    // // Ini interface
    dy_ = sycl::sqrt(dy_) / _DF(0.01) - _DF(1.0); // not actually the same as that in Ref: https://doi.org/10.1016/j.combustflame.2022.112085
    real_t xrest = _DF(1.0), ff = _DF(1.0e-10), dd = _DF(0.5) * (xrest - _DF(2.0) * ff);
    real_t *yi = &(_y[NUM_SPECIES * id]);
    yi[NUM_SPECIES - 1] = dd * (sycl::tanh(dy_ * _DF(0.01) / dx)) + _DF(0.5); // molecular fraction
    yi[0] = 0.30 * (1 - yi[NUM_SPECIES - 1]);
    yi[1] = 0.15 * (1 - yi[NUM_SPECIES - 1]);
    yi[2] = 0.55 * (1 - yi[NUM_SPECIES - 1]);

    get_yi(yi, thermal.Wi); // get yi: mass fraction
    // Get R of mixture
    real_t R = get_CopR(thermal._Wi, yi);
    rho[id] = p[id] / R / T[id];
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