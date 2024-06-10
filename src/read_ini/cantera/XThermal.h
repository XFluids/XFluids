#pragma once

#include "global_setup.h"

/**
 * @brief get_Gibson
 */
extern real_t Gibson(Thermal thermal, const real_t T, const int n);

/**
 *@brief calculate R for every cell
 */
extern real_t CopR(const real_t *_Wi, const real_t *yi);

/**
 * @brief calculate Hi of each species at given point    unit:J/kg/K
 */
extern real_t MoleEnthalpy(Thermal thermal, const real_t T, const int i);

/**
 * @brief Compute the Cpi of each species at given point  unit:J/kg/K
 */
extern real_t HeatCapacity(Thermal thermal, const real_t T, const int i);

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
extern real_t CopEnthalpy(Thermal thermal, const real_t *yi, const real_t T);

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
extern real_t CopHeatCapacity(Thermal thermal, const real_t *yi, const real_t T);
