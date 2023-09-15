#pragma once

#include "Eigen_callback.h"

extern void GetLocalEigen(int i, int j, int k, Block bl, real_t AA, real_t BB, real_t CC, real_t *eigen_local, real_t *u, real_t *v, real_t *w, real_t *c)
{
	MARCO_DOMAIN();
	if (i >= Xmax)
		return;
	if (j >= Ymax)
		return;
	if (k >= Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;

	real_t uu = AA * u[id] + BB * v[id] + CC * w[id];
	real_t uuPc = uu + c[id];
	real_t uuMc = uu - c[id];

	// local eigen values
	for (size_t ii = 1; ii < Emax; ii++)
		eigen_local[Emax * id + ii] = uu;
	eigen_local[Emax * id + 0] = uuMc;
	eigen_local[Emax * id + 2] = uuPc;

	// real_t de_fw[Emax];
	// get_Array(eigen_local, de_fw, Emax, id);
	// real_t de_fx[Emax];
}