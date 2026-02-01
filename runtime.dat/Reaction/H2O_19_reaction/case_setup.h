#pragma once

#define NUM_SPECIES 9
#define NUM_REA 19

#define ZeroDTemperature 1150.0
#define ZeroDPressure 101325.0

/**
 * @brief
 * Ref.A Comprehensive Modeling Study of Hydrogen Oxidation
 * actaully there are 23 reactions listed by Table1, only 19 of them ordered
 * *Arrhenius list A B E , k=A*T^B*exp(-E/R/T)
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
