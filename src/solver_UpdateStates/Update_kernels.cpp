#include "timer/timer.h"
#include "Update_kernels.hpp"
#include "Estimate_kernels.hpp"
#include "kattribute/attribute.h"

void UpdateFuidStatesSPKernel(int i, int j, int k, MeshSize bl, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
											   real_t *rho, real_t *p, real_t *u, real_t *v, real_t *w, real_t *c, real_t *H, real_t const gamma)
{
	MARCO_DOMAIN_GHOST();
	if (i >= Xmax)
		return;
	if (j >= Ymax)
		return;
	if (k >= Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	real_t *U = &(UI[Emax * id]);

	GetStatesSP(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], NCOP_Gamma);

	real_t *Fx = &(FluxF[Emax * id]);
	real_t *Fy = &(FluxG[Emax * id]);
	real_t *Fz = &(FluxH[Emax * id]);

	// GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);
	Fx[0] = U[1];
	Fx[1] = U[1] * u[id] + p[id];
	Fx[2] = U[1] * v[id];
	Fx[3] = U[1] * w[id];
	Fx[4] = (U[4] + p[id]) * u[id];

	Fy[0] = U[2];
	Fy[1] = U[2] * u[id];
	Fy[2] = U[2] * v[id] + p[id];
	Fy[3] = U[2] * w[id];
	Fy[4] = (U[4] + p[id]) * v[id];

	Fz[0] = U[3];
	Fz[1] = U[3] * u[id];
	Fz[2] = U[3] * v[id];
	Fz[3] = U[3] * w[id] + p[id];
	Fz[4] = (U[4] + p[id]) * w[id];
}