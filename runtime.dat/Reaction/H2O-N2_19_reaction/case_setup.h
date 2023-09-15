#pragma once

#define NUM_SPECIES 9
#define NUM_REA 19

#define ZeroDTemperature 1150.0
#define ZeroDPressure 101325.0

/**
 * @brief Ref.https://github.com/deepmodeling/deepflame-dev/blob/master/examples/dfHighSpeedFoam/twoD_detonationH2/H2_Ja.yaml
 * @note: for 2D-detonation
 * *Arrhenius list A B E , k=A*T^B*exp(-E/R/T), NOTE that A is modefied from Ref.yaml by author for this project
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
