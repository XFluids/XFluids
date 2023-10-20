#pragma once

#include "Recon_device.hpp"
#include "Mixing_device.hpp"

#define vector_num 4

/**
 * Riemann solver: given left primitive variables and right primitive variables to Calculate Flux at the middle separation.
 * This is a multi-commponent implementation of HLLC Riemann Solver.
 * Referenced from: The HLLC Riemann Solver(https://doi.org/10.1007/s00193-019-00912-4).
 * @param energy_dir is the id of prim[] to decide which direction's flux to be reconstructed.
 * energy_dir==2 for x-dir energy flux reconstruction, energy_dir==3 for y-dir, energy_dir==4 for z-dir.
 * @param left_prim is the primitive variables at the left side of Riemann interruption.
 * @param right_prim is the primitive variables at the right side of Riemann interruption,
 * prim[6 + NUM_SPECIES] including density(rho), pressure(p), velocity(u,v,w), temperature(T), mass fraction(Yi).
 * @param flux is the calculated flux at middle Riemann interruption.
 * @param thermal is the thermodynamics properties of each component of gas.
 * @param face_vector is the face vector of cell calculated,
 * including x-dir(face_vector[0])/y-dir(face_vector[1])/z-dir(face_vector[2]) projection direction vector and area magnitude of face(face_vector[3]).
 */
void Riemann_solver(const int energy_dir, real_t *left_prim, real_t *right_prim, real_t *flux, Thermal thermal, real_t *face_vector)
{
}