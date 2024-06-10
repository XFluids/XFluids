#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

/**
 * @brief QSSAFun in chemq2, get ydot(production rate) in units of mol*cm^-3*s-1
 */
const real_t _refP = _DF(1.0 / 101325.0);

template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcore(Thermal *tm, Reaction *rn, real_t *yi, real_t *yidot, const real_t m_dens, const real_t m_p, const real_t m_tmp)
{
	// get Kf Kb
	real_t Kf[_NR], _Kck[_NR];
	real_t logStandConc = sycl::log(m_p / (universal_gas_const * m_tmp));
	for (size_t m = 0; m < _NR; m++)
	{
		real_t DeltaGibbs = _DF(0.0);
		real_t A = rn->Rargus[m * 6 + 0], B = rn->Rargus[m * 6 + 1], E = rn->Rargus[m * 6 + 2];
		Kf[m] = A * sycl::exp(B * sycl::log(m_tmp) - E * _DF(4.184) / Ru / m_tmp);
		int *Nu_dm_ = rn->Nu_d_ + m * _NS, m_dn = _DF(0.0);
		for (size_t n = 0; n < _NS; n++)
		{
			DeltaGibbs += Nu_dm_[n] * (sycl::log(m_p * _refP) - Gibson(*tm, m_tmp, n));
			m_dn += Nu_dm_[n];
		}
		_Kck[m] = sycl::min(sycl::exp(DeltaGibbs - m_dn * logStandConc), _DF(1.0E+40));
	}
	// get yidot(production date of species), namely ydot[2] to ydot[end]
	real_t m_cm[_NS];
	// get mole concentrations from mass fraction
	for (size_t n = 0; n < _NS; n++)
		m_cm[n] = yi[n] * tm->_Wi[n] * m_dens * _DF(1.0E-3);

	for (int react_id = 0; react_id < _NR; react_id++)
	{
		// third-body collision effect
		real_t tb = _DF(0.0);
		if (1 == rn->third_ind[react_id])
		{
			for (int it = 0; it < _NS; it++)
				tb += rn->React_ThirdCoef[react_id * _NS + it] * m_cm[it];
		}
		else
			tb = _DF(1.0);

		Kf[react_id] *= tb; // forward
		int *nu_f = rn->Nu_f_ + react_id * _NS;
		_Kck[react_id] *= Kf[react_id]; // backward
		int *nu_b = rn->Nu_b_ + react_id * _NS;
		for (int it = 0; it < _NS; it++)					  // forward
			Kf[react_id] *= sycl::pown(m_cm[it], nu_f[it]);	  // ropf
		for (int it = 0; it < _NS; it++)					  // backward
			_Kck[react_id] *= sycl::pown(m_cm[it], nu_b[it]); // ropb

		Kf[react_id] -= _Kck[react_id];
	}

	// // get omega dot (production rate in the form of mole) units: mole*cm^-s*s^-1
	for (int n = 0; n < _NS; n++)
		yidot[n] = _DF(.0);
	for (int react_id = 0; react_id < _NR; react_id++)
	{
		int *nu_d = rn->Nu_d_ + react_id * _NS;
		for (int n = 0; n < _NS; n++)
			yidot[n] += nu_d[n] * Kf[react_id]; // // get omega dot
	}
}

/**
 * @brief  get ydot(production rate) from y at constant pressure model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcp(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p)
{
	ydot[0] = _DF(.0);
	real_t *yi = y + 1, *yidot = ydot + 1, m_tmp = y[0];

	evalcore<_NS, _NR>(tm, rn, yi, yidot, m_dens, m_p, m_tmp);

	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		yidot[n] *= tm->Wi[n] / m_dens;
		ydot[0] -= yidot[n] * get_Enthalpy(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
	} // ydot[1] = mass * c_p * dT/dt while the loop ends, we need get dT/dt
	// get ydot
	ydot[0] /= get_CopCp(*tm, yi, m_tmp);
}

/**
 * @brief  get ydot(production rate) from y at constant volume model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcv(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p)
{
	ydot[0] = _DF(.0);
	real_t *yi = y + 1, *yidot = ydot + 1, m_tmp = y[0];

	evalcore<_NS, _NR>(tm, rn, yi, yidot, m_dens, m_p, m_tmp);

	for (int n = 0; n < _NS; n++)
	{
		yidot[n] *= tm->Wi[n] / m_dens; // production rate in the form of mass;
		ydot[0] -= yidot[n] * (get_Enthalpy(tm->Hia, tm->Hib, m_tmp, _DF(1.0), n) - m_tmp) * tm->Ri[n];
	} // ydot[0] = mass * c_v * dT/dt while the loop ends, we need get dT/dt

	// get ydot
	ydot[0] /= (get_CopCp(*tm, yi, m_tmp) - Ru / get_CopW(*tm, yi));
}
