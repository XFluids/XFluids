#pragma once

#define NUM_SPECIES 9
#define NUM_REA 25
#define Modified_ArrheniusLaw_form

#define BackArre true
#define ZeroDTemperature 1000.0
#define ZeroDPressure 101325.0

/**
 * @brief Ref.NEW QUASI-STEADY-STATE AND PARTIAL-EQUILIBRIUM M ETHODS FOR INTEGRATING CHEMICALLY REACTING SYSTEMS
 * *Arrhenius list: A B C, k=A*pow(T,B)*exp(-C/T),
 * @note: this Arrhenius type is not supported
 *  units: k: cm^3/(molecule*s), NA=6.02214076*10^23 molcule=1mol
 */
