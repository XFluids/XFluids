#pragma once

#include "sycl_devices.hpp"

/**
 * @brief  Initialize Fluid states espically primitive quantity;
 * @return void
 */
extern void InitialStatesKernel(int i, int j, int k, Block bl, IniShape ini, MaterialProperty material, Thermal thermal,
                                              real_t *u, real_t *v, real_t *w, real_t *rho, real_t *p, real_t *_y, real_t *T) {}

// isentropic vortex problem
// the periodic time is 10s with the [0,10]x[0,10] domain
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

    real_t beta = _DF(5.0);
    real_t beta2 = beta*beta;
    real_t _x_vortex = (x - _DF(0.5)*bl.Domain_length);
    real_t _y_vortex = (y - _DF(0.5)*bl.Domain_width);
    real_t r2 = (_x_vortex * _x_vortex + _y_vortex * _y_vortex);
    real_t inter1 = beta2 * sycl::exp(_DF(1.0) - r2) / _DF(28.0) / pi / pi;
    real_t inter2 = _DF(0.5) * beta * sycl::exp(_DF(0.5) * (_DF(1.0) - r2)) / pi;

    rho[id] = _DF(0.0);
    p[id] = _DF(0.0);
    T[id] = _DF(0.0);
    u[id] = _DF(0.0);
    v[id] = _DF(0.0);
    w[id] = _DF(0.0);

    // initial
    rho[id] = sycl::pow(_DF(1.0) - inter1, _DF(2.5));
    p[id] = sycl::pow(rho[id], _DF(1.4));
    u[id] = _DF(1.0) - inter2 * _y_vortex;
    v[id] = _DF(1.0) + inter2 * _x_vortex;
    
    // initial U[0]-U[3]
    U[Emax * id + 0] = rho[id];
    U[Emax * id + 1] = rho[id] * u[id];
    U[Emax * id + 2] = rho[id] * v[id];
    U[Emax * id + 3] = rho[id] * w[id];
    // EOS included, gamma gas
    real_t Gamma_tmp = _DF(1.4);
    U[Emax * id + 4] = p[id] /(Gamma_tmp-_DF(1.0)) + _DF(0.5)*rho[id]*(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);

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