#pragma once

#include "Recon_device.hpp"
#include "Mixing_device.hpp"

#define vector_num 4

/**
 * ConstructFromPrim: Construct Flux from primitive variables.
 * @param energy_dir is the index of prim[] to decide which direction's flux to be reconstructed.
 * energy_dir==2 for x-dir energy flux reconstruction, energy_dir==3 for y-dir, energy_dir==4 for z-dir.
 * @param target_flux is the solution flux calculated from primitive variables.
 * @param prim is the array of primitive variables.
 * prim[6 + NUM_SPECIES] including density(rho), pressure(p), velocity(u,v,w), temperature(T), mass fraction(Yi).
 * @param thermal is the thermodynamics properties of each component of gas.
 * @param face_vector is the face vector of cell calculated.
 */
void ConstructFromPrim(const int energy_dir, real_t *target_flux, real_t *prim, Thermal thermal, real_t *face_vector)
{
	real_t velNorm = prim[2] * face_vector[0] + prim[3] * face_vector[1] + prim[4] * face_vector[2];
	target_flux[0] = prim[0];
	target_flux[1] = prim[0] * velNorm * prim[2] + prim[1] * face_vector[0]; // x-dir-moment term.
	target_flux[2] = prim[0] * velNorm * prim[3] + prim[1] * face_vector[1]; // y-dir-moment term.
	target_flux[3] = prim[0] * velNorm * prim[4] + prim[1] * face_vector[2]; // z-dir-moment term.
	target_flux[4] = prim[0] * velNorm * prim[energy_dir] * (get_Coph(thermal, &(prim[6]), prim[6]) + _DF(0.5) * (prim[2] * prim[2] + prim[3] * prim[3] + prim[4] * prim[4])); // energy term.
	for (size_t ns = 5; ns < Emax; ns++)
	{
		target_flux[ns] = prim[0] * prim[ns + 1] * velNorm;
	}
}

/**
 * ConstructSolutionFlux: Construct Flux from left_flux and right_flux.
 * @return solution flux
 * @param left_flux is the solution flux calculated from primitive variables.
 * @param right_flux is the array of primitive variables.
 * @param diss is the numerical dissipation of flux.
 */
void ConstructSolutionFlux(real_t *solution_flux, real_t *left_flux, real_t *right_flux, real_t *diss)
{
	for (size_t i = 0; i < Emax; i++)
	{
		solution_flux[i] = left_flux[i] + right_flux[i] - diss[i];
		solution_flux[i] *= _DF(0.5);
	}
}

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
	real_t _rho = sycl::sqrt(right_prim[0] / left_prim[0]), _rho1 = _DF(1.0) / (_rho + _DF(1.0)); // _rho: Roe averaged density.
	real_t _p = get_RoeAverage(left_prim[1], right_prim[1], _rho, _rho1);
	real_t _u = get_RoeAverage(left_prim[2], right_prim[2], _rho, _rho1);
	real_t _v = get_RoeAverage(left_prim[3], right_prim[3], _rho, _rho1);
	real_t _w = get_RoeAverage(left_prim[4], right_prim[4], _rho, _rho1);
	real_t _T = get_RoeAverage(left_prim[5], right_prim[5], _rho, _rho1);

	// start calculation of dissipation term - follows procedure in Blazek 4.3.3
	real_t dissipation[4 + NUM_SPECIES] = {0.0};

	// #if defined(COP)
	real_t _Yi[NUM_SPECIES], _RhoYi[NUM_SPECIES], delta_RhoYi[NUM_SPECIES];
	for (size_t ns = 0; ns < NUM_SPECIES; ns++)
	{
		_Yi[ns] = get_RoeAverage(left_prim[ns + 6], right_prim[ns + 6], _rho, _rho1);
		_RhoYi[ns] = get_RoeAverage(left_prim[0] * left_prim[ns + 6], right_prim[0] * right_prim[ns + 6], _rho, _rho1);
		delta_RhoYi[ns] = right_prim[0] * right_prim[ns + 6] - left_prim[0] * left_prim[ns + 6];
	}
	real_t _h = get_Coph(thermal, _Yi, _T); // Roe Enthalpy
	real_t Gamma = get_CopGamma(thermal, _Yi, _T);
	// #else
	// 	real_t Gamma = NCOP_Gamma;
	// 	real_t _h = (_T); // TODO: _rho*(_h - R*_T) = E - 0.5*_rho*(_u*_u+_v*_v+_w*_w)
	// #endif

	real_t _c = sycl::sqrt(Gamma * _p / _rho);
	real_t delta_prim[6 + NUM_SPECIES];
	for (size_t ii = 0; ii < NUM_SPECIES + 6; ii++)
		delta_prim[ii] = right_prim[ii] - left_prim[ii];
	real_t velNormR = _u * face_vector[0] + _v * face_vector[1] + _w * face_vector[2];
	real_t normVelDiff = (delta_prim[2] * face_vector[0]) + (delta_prim[3] * face_vector[1]) + (delta_prim[4] * face_vector[2]);

	// left moving acoustic wave ------------------------------------------------
	real_t waveSpeed = fabs(velNormR - _c);
	// calculate entropy fix (Harten) and adjust wave speed if necessary
	// default setting for entropy fix to kick in
	const real_t entropyFix = _DF(0.1);
	if (waveSpeed < entropyFix)
		waveSpeed = 0.5 * (waveSpeed * waveSpeed / entropyFix + entropyFix);
	real_t waveStrength = (delta_prim[1] - _rho * _c * normVelDiff) / (_DF(2.0) * _c * _c);
	real_t waveSpeedStrength = waveSpeed * waveStrength;
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		dissipation[ii] = waveSpeedStrength * _Yi[ii];

	dissipation[0 + NUM_SPECIES] = waveSpeedStrength * (_u - _c * face_vector[0]);
	dissipation[1 + NUM_SPECIES] = waveSpeedStrength * (_v - _c * face_vector[1]);
	dissipation[2 + NUM_SPECIES] = waveSpeedStrength * (_w - _c * face_vector[2]);
	dissipation[3 + NUM_SPECIES] = waveSpeedStrength * (_h - _c * velNormR);
	// TODO: Add Turbulence Here.

	// entropy and shear waves -------------------------------------------------
	// entropy wave
	waveSpeed = fabs(velNormR);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
	{
		waveStrength = -delta_prim[1] / (_c * _c);
		waveSpeedStrength = waveSpeed * waveStrength;
		dissipation[ii] += waveSpeedStrength * _Yi[ii] + waveSpeed * delta_RhoYi[ii];
	}
	waveStrength = delta_prim[0] - delta_prim[1] / (_c * _c);
	waveSpeedStrength = waveSpeed * waveStrength;
	dissipation[0 + NUM_SPECIES] += waveSpeedStrength * _u;
	dissipation[1 + NUM_SPECIES] += waveSpeedStrength * _v;
	dissipation[2 + NUM_SPECIES] += waveSpeedStrength * _w;
	dissipation[3 + NUM_SPECIES] += waveSpeedStrength * _DF(0.5) * (_u * _u + _v * _v + _w * _w);
	// turbulence values are zero

	// shear wave
	waveStrength = _rho;
	waveSpeedStrength = waveSpeed * waveStrength;
	// species values are zero
	dissipation[0 + NUM_SPECIES] += waveSpeedStrength * (delta_prim[2] - normVelDiff * face_vector[0]);
	dissipation[1 + NUM_SPECIES] += waveSpeedStrength * (delta_prim[3] - normVelDiff * face_vector[1]);
	dissipation[2 + NUM_SPECIES] += waveSpeedStrength * (delta_prim[4] - normVelDiff * face_vector[2]);
	dissipation[3 + NUM_SPECIES] += waveSpeedStrength * (_u * delta_prim[2] + _v * delta_prim[3] + _w * delta_prim[4] - velNormR * normVelDiff);
	// turbulence values are zero

	// right moving acoustic wave ------------------------------------------------
	waveSpeed = fabs(velNormR + _c);
	// calculate entropy fix (Harten) and adjust wave speed if necessary
	if (waveSpeed < entropyFix)
		waveSpeed = 0.5 * (waveSpeed * waveSpeed / entropyFix + entropyFix);
	waveStrength = (delta_prim[1] + _rho * _c * normVelDiff) / (_DF(2.0) * _c * _c);
	waveSpeedStrength = waveSpeed * waveStrength;
	for (auto ii = 0; ii < dissipation.NumSpecies(); ++ii)
		dissipation[ii] += waveSpeedStrength * _Yi[ii];
	dissipation[0 + NUM_SPECIES] += waveSpeedStrength * (_u + _c * face_vector[0]);
	dissipation[1 + NUM_SPECIES] += waveSpeedStrength * (_v + _c * face_vector[1]);
	dissipation[2 + NUM_SPECIES] += waveSpeedStrength * (_w + _c * face_vector[2]);
	dissipation[3 + NUM_SPECIES] += waveSpeedStrength * (_h + _c * velNormR);
	// TODO: Add Turbulence Here.

	real_t left_flux[Emax], right_flux[Emax];
	ConstructFromPrim(energy_dir, left_flux, left_prim, thermal, face_vector);
	ConstructFromPrim(energy_dir, right_flux, right_prim, thermal, face_vector);
	// Solution Flux.
	ConstructSolutionFlux(flux, left_flux, right_flux, dissipation);
}