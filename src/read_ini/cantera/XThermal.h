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
 * @brief calculate Hi of each species at given point    unit:J/mol/K
 */
extern real_t Enthalpy(Thermal thermal, const real_t T, const int i);

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

/**
 * @brief Compute the production rate
 */
extern void evalcoreWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *yidot, const real_t m_dens, const real_t m_p, const real_t m_tmp);

/**
 * @brief Compute the production rate of constant pressure model
 */
extern void evalcpWrapper(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p);

/**
 * @brief Compute the production rate of constant volume model
 */
extern void evalcvWrapper(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p);

/**
 * @brief Compute the production rate
 */
extern void evalcoreWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_p, const real_t m_tmp);

/**
 * @brief Compute the production rate of constant pressure model
 */
extern real_t evalcpWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p);

/**
 * @brief Compute the production rate of constant volume model
 */
extern real_t evalcvWrapper(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p);

/**
 * @brief NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
 * @brief Compute the production rate of constant volume model
 */
void Chemq2WrapperCv0(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p);

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief Compute the production rate of constant volume model
 */
void Chemq2WrapperCv1(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p);
void Chemq2WrapperCp1(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p);
