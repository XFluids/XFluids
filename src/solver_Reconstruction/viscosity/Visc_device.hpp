#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"

/**
 * @brief get viscosity at temperature T(unit:K)(fit)
 * @return real_t,unit: Pa.s=kg/(m.s)
 */
real_t Viscosity(real_t fitted_coefficients_visc[order_polynominal_fitted], const real_t T0)
{
	// real_t Tref = Reference_params[3], visref = Reference_params[5];
	real_t T = T0; //* Tref; // nondimension==>dimension
	real_t viscosity = fitted_coefficients_visc[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		viscosity = viscosity * sycl::log(T) + fitted_coefficients_visc[i];
	real_t temp = sycl::exp(viscosity); // / visref;
	return temp;						// dimension==>nondimension

	// real_t Tref = Reference_params[3], visref = Reference_params[5];
	// real_t T = T0 * Tref; // nondimension==>dimension
	// real_t viscosity = fitted_coefficients_visc[order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	viscosity = viscosity * sycl::log(T) + fitted_coefficients_visc[i];
	// real_t temp = sycl::exp(viscosity) / visref;
	// return temp; // dimension==>nondimension
}

/**
 * @brief get viscosity at temperature T(unit:K)
 * @return real_t,unit: Pa.s=kg/(m.s)
 */
real_t PHI(real_t *specie_k, real_t *specie_j, real_t *fcv[NUM_SPECIES], const real_t T)
{
	real_t phi = _DF(0.0);
	phi = sycl::pow(specie_j[Wi] / specie_k[Wi], _DF(0.25)) * sycl::pow(Viscosity(fcv[int(specie_k[SID])], T) / Viscosity(fcv[int(specie_j[SID])], T), _DF(0.5));
	phi = (phi + _DF(1.0)) * (phi + _DF(1.0)) * _DF(0.5) / sycl::sqrt(_DF(2.0));
	phi = phi * sycl::pow(_DF(1.0) + specie_k[Wi] / specie_j[Wi], -_DF(0.5));
	return phi;
}

/**
 * @brief get thermal conductivity at temperature T(unit:K)
 * @return real_t,unit: W/(m.K)
 */
real_t Thermal_conductivity(real_t fitted_coefficients_therm[order_polynominal_fitted], const real_t T0)
{
	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t kref = visref * (pref / rhoref);
	real_t T = T0; //* Tref; // nondimension==>dimension
	real_t thermal_conductivity = fitted_coefficients_therm[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		thermal_conductivity = thermal_conductivity * sycl::log(T) + fitted_coefficients_therm[i];
	real_t temp = sycl::exp(thermal_conductivity); // / kref;
	return temp;								   // dimension==>nondimension

	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t kref = visref * (pref / rhoref);
	// real_t T = T0 * Tref; // nondimension==>dimension
	// real_t thermal_conductivity = fitted_coefficients_therm[order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	thermal_conductivity = thermal_conductivity * sycl::log(T) + fitted_coefficients_therm[i];
	// real_t temp = sycl::exp(thermal_conductivity) / kref;
	// return temp; // dimension==>nondimension
}

/**
 * @brief get Dkj:the binary difffusion coefficient of specie-k to specie-j via equation 5-37
 * @para TT temperature unit:K
 * @para PP pressure unit:Pa
 */
real_t GetDkj(real_t *specie_k, real_t *specie_j, real_t **Dkj_matrix, const real_t T0, const real_t P0)
{
	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t Dref = visref / rhoref;
	real_t TT = T0; // * Tref; // nondimension==>dimension
	real_t PP = P0; // * pref; // nondimension==>dimension
	real_t Dkj = Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		Dkj = Dkj * sycl::log(TT) + Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][i];
	real_t temp = (sycl::exp(Dkj) / PP); // / Dref;
	return temp;						 // unit:cm*cm/s　//dimension==>nondimension

	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t Dref = visref / rhoref;
	// real_t TT = T0 * Tref; // nondimension==>dimension
	// real_t PP = P0 * pref; // nondimension==>dimension
	// real_t Dkj = Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	Dkj = Dkj * sycl::log(TT) + Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][i];
	// real_t temp = (sycl::exp(Dkj) / PP) / Dref;
	// return temp; // unit:cm*cm/s　//dimension==>nondimension
}

/**
 * @brief get average transport coefficient
 * @param chemi is set to get species information
 */
void Get_transport_coeff_aver(const int i_id, const int j_id, const int k_id, Thermal thermal, real_t *Dkm_aver_id, real_t &viscosity_aver, real_t &thermal_conduct_aver, real_t const X[NUM_SPECIES],
							  const real_t rho, const real_t p, const real_t T, const real_t C_total, real_t *Ertemp1, real_t *Ertemp2)
{
	real_t **fcv = thermal.fitted_coefficients_visc;
	real_t **fct = thermal.fitted_coefficients_therm;
	real_t **Dkj = thermal.Dkj_matrix;
	viscosity_aver = _DF(0.0);
#if Visc_Heat
	thermal_conduct_aver = _DF(0.0);
#endif
	real_t denominator = _DF(0.0);
	real_t *specie[NUM_SPECIES];
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		specie[ii] = &(thermal.species_chara[ii * SPCH_Sz]);
	for (int k = 0; k < NUM_SPECIES; k++)
	{
		denominator = _DF(0.0);
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			// real_t temp = PHI(specie[k], specie[i], fcv, T);
			denominator = denominator + X[i] * PHI(specie[k], specie[i], fcv, T);
		}
		// calculate viscosity_aver via equattion(5-49)//
		real_t _denominator = _DF(1.0) / denominator;
		viscosity_aver = viscosity_aver + X[k] * Viscosity(fcv[int(specie[k][SID])], T) * _denominator; // Pa.s=kg/(m.s)
#if Visc_Heat
		// calculate thermal_conduct via Su Hongmin//
#ifdef ConstantFurrier
		thermal_conduct_aver = _DF(0.1);
#else
		thermal_conduct_aver = thermal_conduct_aver + X[k] * Thermal_conductivity(fct[int(specie[k][SID])], T) * _denominator;
#endif // end
#endif // end Visc_Heat
	}
#if Visc_Diffu
	// calculate diffusion coefficient specie_k to mixture via equation 5-45
#if 1 < NUM_SPECIES
	{
		real_t temp1, temp2;
		for (int k = 0; k < NUM_SPECIES; k++)
		{
			temp1 = _DF(0.0), temp2 = _DF(1.0e-20);
			for (int i = 0; i < NUM_SPECIES; i++)
			{
				if (i != k)
				{
					temp1 += (X[i] + _DF(1.0e-40)) * thermal.Wi[i]; // asmuing has mixture enven for pure gas. see SuHongMin's Dissertation P19.2-35
					temp2 += (X[i] + _DF(1.0e-40)) / (GetDkj(specie[i], specie[k], Dkj, T, p) + _DF(1.0e-40));
				}
			}											 // cause nan error while only one yi of the mixture given(temp1/temp2=0/0).
			if (sycl::step(sycl::ceil(temp1), _DF(0.0))) // =1 while temp1==0.0;temp may < 0;
				Dkm_aver_id[k] = GetDkj(specie[k], specie[k], Dkj, T, p);
			else
				Dkm_aver_id[k] = temp1 / temp2 / rho * C_total; // rho/C_total:the mole mass of mixture;
			Dkm_aver_id[k] *= _DF(1.0e-1);						// cm2/s==>m2/s

#if ESTIM_OUT
			Ertemp1[k] = temp1, Ertemp2[k] = temp2;
#endif // end ESTIM_OUT
		}
	}
#else
	{															  // NUM_SPECIES==1
		Dkm_aver_id[0] = GetDkj(specie[0], specie[0], Dkj, T, p); // trans_coeff.GetDkj(T, p, chemi.species[0], chemi.species[0], refstat);
		Dkm_aver_id[0] *= _DF(1.0e-1);							  // cm2/s==>m2/s
	}
#endif // end NUM_SPECIES>1
	   // NOTE: add limiter:
	for (int k = 0; k < NUM_SPECIES; k++)
	{
		Dkm_aver_id[k] = sycl::max(Dkm_aver_id[k], _DF(1.0e-10));
		// Dkm_aver_id[k] = sycl::min(Dkm_aver_id[k], _DF(1.0e-4));
	}
#endif // end Visc_Diffu
}
