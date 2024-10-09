#include "XThermal.h"
#include "Mixing_device.h"
#include "../../solver_Reaction/ProductionRates.h"

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
 * @brief calculate Hi of each species at given point    unit:J/mol/K
 */
real_t Enthalpy(Thermal thermal, const real_t T, const int i)
{
	real_t temp = get_Enthalpy(thermal.Hia, thermal.Hib, T, thermal.Ri[i], i);

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

/**
 * @brief Compute the production rate
 */
void evalcoreWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *yidot, const real_t m_dens, const real_t m_p, const real_t m_tmp)
{
	evalcore<NUM_SPECIES, NUM_REA>(tm, rn, yi, yidot, m_dens, m_p, m_tmp);
}

/**
 * @brief Compute the production rate of constant pressure model
 */
void evalcpWrapper(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p)
{

	ydot[0] = 0; // dT/dt
	evalcp<NUM_SPECIES, NUM_REA>(tm, rn, y, ydot, m_dens, m_p);
}

/**
 * @brief Compute the production rate of constant volume model
 */
void evalcvWrapper(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p)
{
	ydot[0] = 0; // dT/dt
	evalcv<NUM_SPECIES, NUM_REA>(tm, rn, y, ydot, m_dens, m_p);
}

/**
 * @brief Compute the production rate
 */
void evalcoreWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_p, const real_t m_tmp)
{
	evalcore<NUM_SPECIES, NUM_REA>(tm, rn, yi, q, p, m_dens, m_p, m_tmp);
}

/**
 * @brief Compute the production rate of constant pressure model
 */
real_t evalcpWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p)
{

	real_t dTdt = evalcp<NUM_SPECIES, NUM_REA>(tm, rn, yi, q, p, m_dens, m_tmp, m_p);

	return dTdt;
}

/**
 * @brief Compute the production rate of constant volume model
 */
real_t evalcvWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p)
{

	real_t dTdt = evalcv<NUM_SPECIES, NUM_REA>(tm, rn, yi, q, p, m_dens, m_tmp, m_p);

	return dTdt;
}

/**
 * @brief NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
 * @brief Compute the production rate of constant volume model
 */
void Chemq2WrapperCv0(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p)
{

	Chemeq2<NUM_SPECIES, NUM_REA>(tm, rn, y + 1, dtg, y[0], rho, m_p);
}

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief Compute the production rate of constant volume model
 */
void Chemq2WrapperCv1(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p)
{

	Chemeq2<NUM_SPECIES, NUM_REA>(tm, rn, y, dtg, rho, m_p);
}

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief Compute the production rate of constant volume model
 */
void Chemq2WrapperCp1(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p)
{

	Chemeq2Cp<NUM_SPECIES, NUM_REA>(tm, rn, y, dtg, rho, m_p);
}