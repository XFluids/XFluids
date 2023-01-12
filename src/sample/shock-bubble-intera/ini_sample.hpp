#pragma once
#include "../../include/global_class.h"
#include "../../sycl_devices/device_func.hpp"

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

#ifdef COP
    real_t *xi = &(_y[NUM_SPECIES * id]); //[NUM_SPECIES] = {0.0};
    // for 2D/3D shock-bubble interactive
    real_t dy_ = _DF(0.0), tmp = _DF(0.0);
    // real_t dy_in = _DF(0.0), dy_out = _DF(0.0);
    // real_t dy_in = -_DF(1.0), dy_out = -_DF(1.0);
#if DIM_X
    tmp = (x - ini.cop_center_x) * (x - ini.cop_center_x);
    dy_ += tmp * ini._xa2;
    // dy_in += tmp * ini._xa2_in;
    // dy_out += tmp * ini._xa2_out;
#endif // end DIM_X
#if DIM_Y
    tmp = (y - ini.cop_center_y) * (y - ini.cop_center_y);
    dy_ += tmp * ini._yb2;
    // dy_in += tmp * ini._yb2_in;
    // dy_out += tmp * ini._yb2_out;
#endif // end DIM_Y
    // for 3D shock-bubble interactive
#if DIM_Z
    tmp = (z - ini.cop_center_z) * (z - ini.cop_center_z);
    dy_ += tmp * ini._zc2;
    // dy_in += tmp * ini._zc2_in;
    // dy_out += tmp * ini._zc2_out;
#endif // end DIM_Z

    // // Ini bubble
    dy_ = sqrt(dy_) - _DF(1.0); // not actually the same as that in Ref: https://doi.org/10.1016/j.combustflame.2022.112085
    real_t xrest = _DF(1.0), ff = _DF(1.0e-4), dd = _DF(0.5) * (xrest - _DF(2.0) * ff);
    xi[NUM_SPECIES - 1] = dd * (sycl::tanh<real_t>(dy_ * ini.C)) + _DF(0.5); // increase ini.C for a sharper boundary //[-1,1]--0.5(1-ff)*[-1,1]+0.5-->[ff,1-ff]
    xi[0] = _DF(0.3) * (xrest - xi[NUM_SPECIES - 1]);                        // H2
    xi[1] = _DF(0.15) * (xrest - xi[NUM_SPECIES - 1]);                       // O2
    xi[NUM_SPECIES - 2] = _DF(0.55) * (xrest - xi[NUM_SPECIES - 1]);         // Xe

    get_yi(xi, thermal.Wi);

#ifdef COP_CHEME
    real_t xre = _DF(1.0e-15), ratios = xre * real_t(NUM_COP - 3) * _DF(0.25);
    for (size_t n1 = 0; n1 < NUM_SPECIES; n1++)
        xi[n1] -= ratios;
    for (size_t nn = 2; nn < NUM_COP - 1; nn++)
        xi[nn] = xre;
#endif // end COP_CHEME

#endif // end COP

    if (x > ini.blast_center_x) //(i > 3)
    {
        T[id] = ini.blast_T_out;
        p[id] = ini.blast_pressure_out;
        u[id] = ini.blast_u_out;
        v[id] = ini.blast_v_out;
        w[id] = ini.blast_w_out;
    }
    else
    {
        T[id] = ini.blast_T_in;
        p[id] = ini.blast_pressure_in;
        u[id] = ini.blast_u_in;
        v[id] = ini.blast_v_in;
        w[id] = ini.blast_w_in;
    }

    // if (x > ini.blast_center_x) // downstream of the shock
    //     {
    //         T[id] = ini.blast_T_out;
    //         rho[id] = ini.blast_density_out;
    //         p[id] = ini.blast_pressure_out;
    //         u[id] = ini.blast_u_out;
    //         v[id] = ini.blast_v_out;
    //         w[id] = ini.blast_w_out;
    // #ifdef COP
    //         if (dy_in < 0 && dy_out < 0)
    //         {                                 // in bubble
    //             rho[id] = ini.cop_density_in; // 气泡内单独赋值密度以和气泡外区分
    //             p[id] = ini.cop_pressure_in;  // 气泡内单独赋值压力以和气泡外区分
    //             for (size_t i = 0; i < NUM_SPECIES; i++)
    //                 _y[i][id] = thermal.species_ratio_in[i];
    //         }
    //         else if (dy_in > 0 && dy_out < 0)
    //         { // boundary of bubble && shock
    //             rho[id] = _DF(0.5) * (ini.cop_density_in + ini.blast_density_out);
    //             p[id] = _DF(0.5) * (ini.cop_pressure_in + ini.blast_pressure_out);
    //             for (size_t i = 0; i < NUM_SPECIES; i++)
    //                 _y[i][id] = _DF(0.5) * (thermal.species_ratio_in[i] + thermal.species_ratio_out[i]);
    //         }
    //         else
    //         { // out of bubble
    //             for (size_t i = 0; i < NUM_SPECIES; i++)
    //                 _y[i][id] = thermal.species_ratio_out[i];
    //         }
    // #endif // end COP
    //     }
    //     else // upstream of the shock
    //     {
    //         T[id] = ini.blast_T_in;
    //         rho[id] = ini.blast_density_in;
    //         p[id] = ini.blast_pressure_in;
    //         u[id] = ini.blast_u_in;
    //         v[id] = ini.blast_v_in;
    //         w[id] = ini.blast_w_in;
    // #ifdef COP // to be 1d shock without define React
    //         for (size_t i = 0; i < NUM_SPECIES; i++)
    //             _y[i][id] = thermal.species_ratio_out[i];
    // #endif // end COP
    //     }
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

    // // Ini yi
    // get_yi(_y, yi, id);
    // Get R of mixture
    real_t *yi = &(_y[NUM_SPECIES * id]); //[NUM_SPECIES];

    real_t R = get_CopR(thermal._Wi, yi);
    rho[id] = p[id] / R / T[id];

    // real_t Gamma_m = get_CopGamma(thermal, yi, T[id]);
    // c[id] = sqrt(p[id] / rho[id] * Gamma_m);

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
