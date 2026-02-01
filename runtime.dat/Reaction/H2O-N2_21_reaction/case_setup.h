#pragma once

#define NUM_SPECIES 10
#define NUM_REA 21
#define GhostSpecies

#define ZeroDTemperature 1000.0
#define ZeroDPressure 101325.0
/**
 * @brief Ref.A detailed verification procedure for compressible reactive multicomponent Navier-Stokes solvers
 * @note: for mixing layer, One-dimensional hydrogen/oxygen laminar premixed flame.
 * *Arrhenius list: A B E, k=A*pow(T,B)*exp(-E/R/T),
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
