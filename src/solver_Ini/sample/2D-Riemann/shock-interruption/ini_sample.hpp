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

    rho[id] = _DF(0.0), p[id] = _DF(0.0), T[id] = _DF(0.0), u[id] = _DF(0.0), v[id] = _DF(0.0), w[id] = _DF(0.0);

    real_t *yi = &(_y[NUM_SPECIES * id]);
    for (size_t n = 0; n < NUM_SPECIES; n++)
        yi[n] = thermal.species_ratio_out[n];

    real_t rho1 = _DF(1.0), p1 = _DF(1.0), u1 = _DF(0.1), v1 = _DF(0.0);
    real_t rho2 = _DF(0.5313), p2 = _DF(0.4), u2 = _DF(0.8276), v2 = _DF(0.0);
    real_t rho3 = _DF(0.8), p3 = _DF(0.4), u3 = _DF(0.1), v3 = _DF(0.0);
    real_t rho4 = _DF(0.5313), p4 = _DF(0.4), u4 = _DF(0.1), v4 = _DF(0.7276);

    if (y > _DF(0.5) * bl.Domain_height)
    {
        if (x > _DF(0.5) * bl.Domain_length) // zone 1
            rho[id] = rho1, p[id] = rho1, u[id] = u1, v[id] = v1;
        else // x < _DF(0.5) * bl.Domain_length // zone 2
            rho[id] = rho2, p[id] = rho2, u[id] = u2, v[id] = v2;
    }
    else // y < _DF(0.5) * bl.Domain_height
    {
        if (x < _DF(0.5) * bl.Domain_length) // zone 3
            rho[id] = rho3, p[id] = rho3, u[id] = u3, v[id] = v3;
        else // x > _DF(0.5) * bl.Domain_length // zone 4
            rho[id] = rho4, p[id] = rho4, u[id] = u4, v[id] = v4;
    }

    c[id] = sycl::sqrt(p[id] / rho[id] * NCOP_Gamma);
    // initial U[0]-U[3]
    U[Emax * id + 0] = rho[id];
    U[Emax * id + 1] = rho[id] * u[id];
    U[Emax * id + 2] = rho[id] * v[id];
    U[Emax * id + 3] = rho[id] * w[id];
    U[Emax * id + 4] = p[id] / (NCOP_Gamma - _DF(1.0)) + _DF(0.5) * rho[id] * (u[id] * u[id] + v[id] * v[id] + w[id] * w[id]);

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