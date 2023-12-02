#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"
#include "../include/sycl_devices.hpp"
#include "../schemes/schemes_device.hpp"

/**
 * @param left_id is the array for right high-order primitive variables reconstruction.
 * for x-dir WENO reconstruction: left_id[]={id-3, id-2, id-1, id, id+1}, right_id[]={id+2, id+1, id, id-1, id-2}
 * @param right_id is the array for right high-order primitive variables reconstruction.
 * @param energy_dir is the id of prim[] to decide which direction's flux to be reconstructed.
 * energy_dir==2 for x-dir energy flux reconstruction, energy_dir==3 for y-dir, energy_dir==4 for z-dir.
 * @param bl is the block_size of mesh.
 * @param thermal is the thermodynamics properties of each component of gas.
 * @param Fwall is where the solution Flux stored.
 * @param rho, p, u, v, w, y, T is all the primitive variables used to high-order reconstruction and flux consruction.
 * @param face_vector is the area face vector and magnitude of face area.
 */
extern void ReconstructFlux(int *left_id, int *right_id, int id, int energy_dir, Block bl, Thermal thermal, real_t *Fwall,
										  real_t *rho, real_t *p, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *face_vector)
{
	int Xmax = bl.Xmax, Ymax = bl.Ymax;
	real_t *PVar[6] = {rho, p, u, v, w, T}, RePVar_l[6 + NUM_SPECIES], RePVar_r[6 + NUM_SPECIES];
	for (size_t n = 0; n < 6; n++)
	{
		RePVar_l[n] = weno5old_BODY(PVar[n][left_id[0]], PVar[n][left_id[1]], PVar[n][left_id[2]], PVar[n][left_id[3]], PVar[n][left_id[4]]);
		RePVar_r[n] = weno5old_BODY(PVar[n][right_id[0]], PVar[n][right_id[1]], PVar[n][right_id[2]], PVar[n][right_id[3]], PVar[n][right_id[4]]);
	}

	// #if defined(COP)
	for (size_t ns = 0; ns < NUM_SPECIES; ns++)
	{
		RePVar_l[ns + 6] = weno5old_BODY(y[(left_id[0]) * NUM_SPECIES + ns], y[(left_id[1]) * NUM_SPECIES + ns], y[(left_id[2]) * NUM_SPECIES + ns], y[(left_id[3]) * NUM_SPECIES + ns], y[(left_id[4]) * NUM_SPECIES + ns]);
		RePVar_r[ns + 6] = weno5old_BODY(y[(right_id[0]) * NUM_SPECIES + ns], y[(right_id[1]) * NUM_SPECIES + ns], y[(right_id[2]) * NUM_SPECIES + ns], y[(right_id[3]) * NUM_SPECIES + ns], y[(right_id[4]) * NUM_SPECIES + ns]);
	}
	// #endif

	Riemann_solver(energy_dir, RePVar_l, RePVar_r, Fwall, thermal, &(face_vector[id * vector_num]));
}
