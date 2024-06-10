#include "XThermal.h"
#include "Mixing_device.h"

/**
 * @brief get_Gibson
 */
real_t Gibson(Thermal thermal, const real_t T, const int n)
{
	return get_Gibson(thermal.Hia, thermal.Hib, T, thermal.Ri[n], n);
}

/**
 *@brief calculate R for every cell
 */
real_t CopR(const real_t *_Wi, const real_t *yi)
{
	return get_CopR(_Wi, yi);
}

/**
 * @brief calculate Hi of each species at given point    unit:J/kg/K
 */
real_t MoleEnthalpy(Thermal thermal, const real_t T, const int i)
{
	real_t temp = get_Enthalpy(thermal.Hia, thermal.Hib, T, universal_gas_const, i);

	return temp;
}

/**
 * @brief Compute the Cpi of each species at given point  unit:J/kg/K
 */
real_t HeatCapacity(Thermal thermal, const real_t T, const int i)
{
	return HeatCapacity(thermal.Hia, T, thermal.Ri[i], i);
}

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
real_t CopEnthalpy(Thermal thermal, const real_t *yi, const real_t T)
{
	real_t h = _DF(0.0);
	for (size_t i = 0; i < NUM_SPECIES; i++)
	{
		real_t hi = get_Enthalpy(thermal.Hia, thermal.Hib, T, thermal.Ri[i], i);
		h += hi * yi[i];
	}
	return h;
}

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
real_t CopHeatCapacity(Thermal thermal, const real_t *yi, const real_t T)
{
	real_t _CopCp = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_CopCp += yi[ii] * HeatCapacity(thermal, T, ii);

	return _CopCp;
}
