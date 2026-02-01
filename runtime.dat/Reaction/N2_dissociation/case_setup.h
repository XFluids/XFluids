#pragma once

#define NUM_SPECIES 2
#define NUM_REA 2

#define Modified_ArrheniusLaw_form
#define ZeroDTemperature 4000.0
#define ZeroDPressure 100000.0
#define ZeroDtStep 1.0E-7
#define ZeroEndTime 5.0E-4

/**
 * @brief Ref1.Table 1 of Assessment of a Two-Temperature Kinetic Model for Dissociating and Weakly Ionizing Nitrogen
 * Ref2.高超声速流场及减热减阻的数值模拟与研究.聂春生(硕士学位论文)
 * @note: only the first two kinetics without charged particles are involved based simulation of Ref2
 * *Arrhenius list A B E , k=A*T^B*exp(-E/R/T)
 * units: k: cm^3/mol/s, E: cal/mole, A: same with k
 */