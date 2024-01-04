#pragma once

#define NUM_SPECIES 9
#define NUM_REA 21

/**
 * @brief Ref.A detailed verification procedure for compressible reactive multicomponent Navier-Stokes solvers
 * @note: for mixing layer
 * *Arrhenius list: A B E, k=A*pow(T,B)*exp(-E/R/T),
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
