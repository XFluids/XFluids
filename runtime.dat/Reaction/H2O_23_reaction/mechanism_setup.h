/*
 * @brief Define chemical mechanism
*/
#ifndef _MECHANSIM_SETUP_H_
#define _MECHANSIM_SETUP_H_
#include "dimension_setup.h"

/*
 * @brief one example:
 * Oran et al. H2-O2 detailed mechanism with 9 species & 24 reversible reactions (48 unidirectional reactions)
**/
const int num_species       = 9;
const int num_reactions     = 23;
const double universal_gas_const = 8.314510; // J/(K mol), "R0", p V = n R0 T
const double heat_release[num_species] = {25,0};//{0.5196e10, 0}; // chemical_heat_release

/**
 * @brief default error torlerence for CVODE
 */
#define RTOL  RCONST(1.0e-5)   /* scalar relative tolerance            */
#define ATOL  RCONST(1.0e-13)   /* vector absolute tolerance components */

#endif