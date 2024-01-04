#pragma once

#define NUM_SPECIES 9
#define NUM_REA 18

#define ZeroDTemperature 1150.0
#define ZeroDPressure 101325.0

/**
 * @brief
 * Ref.PREMIX:AFORTRAN Program for Modeling Steady Laminar One-Dimensional Premixed Flames Paper48
 * *Arrhenius list A B E , k=A*T^B*exp(-E/R/T)
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
