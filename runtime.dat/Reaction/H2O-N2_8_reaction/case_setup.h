#pragma once

#define NUM_SPECIES 9
#define NUM_REA 8

#define BackArre true
#define ZeroDTemperature 1400.0
#define ZeroDPressure 101325.0

/**
 * @brief
 *  Ref.https://github.com/deepmodeling/deepflame-dev/blob/master/mechanisms/H2/ES80_H2-7-16.yaml
 * *Arrhenius list A B E , k=A*T^B*exp(-E/R/T), NOTE that A is modefied from Ref.yaml by author for this project
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */
