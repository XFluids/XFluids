#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

const real_t _refP = _DF(1.0 / 101325.0);

/**
 * @brief Get ydot(production rate) in units of mol*cm^-3*s-1
 */
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
			DeltaGibbs += Nu_dm_[n] * (sycl::log(m_p * _refP) - get_Gibson(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n));
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
	{
		yidot[n] = _DF(.0);
		for (int react_id = 0; react_id < _NR; react_id++)
		{
			int *nu_d = rn->Nu_d_ + react_id * _NS;
			yidot[n] += nu_d[n] * Kf[react_id]; // // get omega dot
		}
	}
}

/**
 * @brief  get ydot(production rate) from y at constant pressure model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcp(Thermal *tm, Reaction *rn, real_t *y, real_t *ydot, const real_t m_dens, const real_t m_p)
{
	ydot[0] = _DF(.0);
	real_t *yi = y + 1, *yidot = ydot + 1, m_tmp = y[0], _dens = _DF(1.0) / m_dens;

	evalcore<_NS, _NR>(tm, rn, yi, yidot, m_dens, m_p, m_tmp);

	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		yidot[n] *= tm->Wi[n] * _dens;
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
	real_t *yi = y + 1, *yidot = ydot + 1, m_tmp = y[0], _dens = _DF(1.0) / m_dens;

	evalcore<_NS, _NR>(tm, rn, yi, yidot, m_dens, m_p, m_tmp);

	for (int n = 0; n < _NS; n++)
	{
		yidot[n] *= tm->Wi[n] * _dens; // production rate in the form of mass;
		ydot[0] -= yidot[n] * get_Internale(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
	} // ydot[0] = mass * c_v * dT/dt while the loop ends, we need get dT/dt

	// get ydot
	ydot[0] /= get_CopCv(*tm, yi, m_tmp);
}

/**
 * @brief Get q(production rate) and p(loss rate) in units of mol*cm^-3*s-1
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcore(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_p, const real_t m_tmp)
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
			DeltaGibbs += Nu_dm_[n] * (sycl::log(m_p * _refP) - get_Gibson(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n));
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
	}

	// // get omega dot (production rate in the form of mole) units: mole*cm^-s*s^-1
	for (int n = 0; n < _NS; n++)
	{
		q[n] = _DF(.0), p[n] = _DF(.0);
		for (int react_id = 0; react_id < _NR; react_id++)
		{
			int *nu_f = rn->Nu_f_ + react_id * _NS;
			int *nu_b = rn->Nu_b_ + react_id * _NS;
			q[n] += nu_b[n] * Kf[react_id] + nu_f[n] * _Kck[react_id]; // // get production rate
			p[n] += nu_f[n] * Kf[react_id] + nu_b[n] * _Kck[react_id]; // // get loss rate
		}
	}
}

/**
 * @brief NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
 * @brief  get q(production rate) and p(loss rate) from y at constant pressure model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE real_t evalcp(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p)
{
	evalcore<_NS, _NR>(tm, rn, yi, q, p, m_dens, m_p, m_tmp);

	real_t dTdt = _DF(0.0), _dens = _DF(1.0) / m_dens;
	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		q[n] *= tm->Wi[n] * _dens;
		p[n] *= tm->Wi[n] * _dens;
		dTdt -= (q[n] - p[n]) * get_Enthalpy(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
	} // dTdt = mass * c_p * dT/dt while the loop ends, we need get dT/dt
	// get ydot
	dTdt /= get_CopCp(*tm, yi, m_tmp);

	return dTdt;
}

/**
 * @brief NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
 * @brief  get q(production rate) and p(loss rate) from y at constant volume model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE real_t evalcv(Thermal *tm, Reaction *rn, real_t *yi, real_t *q, real_t *p, const real_t m_dens, const real_t m_tmp, const real_t m_p)
{
	evalcore<_NS, _NR>(tm, rn, yi, q, p, m_dens, m_p, m_tmp);

	real_t dTdt = _DF(0.0), _dens = _DF(1.0) / m_dens;
	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		q[n] *= tm->Wi[n] * _dens;
		p[n] *= tm->Wi[n] * _dens;
		dTdt -= (q[n] - p[n]) * get_Internale(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
	} // dTdt = mass * c_v * dT/dt while the loop ends, we need get dT/dt
	// get ydot
	dTdt /= get_CopCv(*tm, yi, m_tmp);

	return dTdt;
}

/**
 * @brief sign for one argus
 */
inline SYCL_DEVICE real_t xfrsign(const real_t a)
{
	if (a >= _DF(.0))
		return _DF(1.0);
	else
		return -_DF(1.0);
}

/**
 * @brief sign for two argus
 */
inline SYCL_DEVICE real_t xfrsign(const real_t a, const real_t b)
{
	return xfrsign(b) * sycl::fabs(a);
}

// // neomorph of Ref.eq(39) for rtaui=1/r in eq(39)
#define Alpha(rtaui) (_DF(180.0) + rtaui * (_DF(60.0) + rtaui * (_DF(11.0) + rtaui))) / (_DF(360.0) + rtaui * (_DF(60.0) + rtaui * (_DF(12.0) + rtaui)));

/**
 * @brief NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
 * @brief Chemeq2: q represents the production rate , d represents the los rate , di = pi*yi in RefP408 eq(2)
 * @brief The accuracy-based timestep calculation can be augmented with a stability-based check when at least
 * three corrector iterations are performed. For most problems, the stability check is not needed, and eliminating
 * the calculations and logic associated with the check enhances performance.
 * @ref   A Quasi-Steady-State Solver for the Stiff Ordinary Differential Equations of Reaction Kinetics
 * @param dtg: duration of time integral
 * @param itermax: iterations of correction
 * @param ymin: minimum concentration allowed for species i, too much low ymin decrease performance
 * 	*NOTE: initializing time intergation control
 * @param dto: original dt for a integration step
 * @param tfd: round-off parameter used to determine when integration is complete
 * @param epsmin: to calculate initial time step of q2 integral, intializa into _DF(1e-04).
 * @param scrtch: to calculate initial time step of q2 integral, intializa into _DF(1e-25).
 * @param sqreps: 5.0*sycl::sqrt(epsmin), parameter used to calculate initial timestep
 * @param dtmin: minimum dt for each step, automatically relax convergence restrictions while dt<=dtmin*dto for a step .
 * 	*NOTE: epsion contrl
 * @param eps: error epslion, intializa into _DF(1e-10).
 * @param epsmax: if this previous step not converged, higher for low accuracy and higher performace.
 * @param epscl: 1.0/epsmin, intermediate variable used to avoid repeated divisions, higher epscl leading to higher accuracy and lower performace.
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void Chemeq2(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, real_t &TT, const real_t rho, const real_t m_p)
{
	int itermax = 1;
	real_t ymin = _DF(1.0e-20), dtmin = _DF(1.0e-7);
	real_t eps, epsmin = _DF(1.0e-4), scrtch = _DF(1e-25);
	real_t tfd = _DF(1.0) + _DF(1.0e-10), sqreps = _DF(0.05), epsmax = _DF(1.0), epscl = _DF(1.0e4);

	real_t ys[_NS], y1[_NS];
	real_t rtau[_NS], scrarray[_NS];						 //  \delta y based input y_i value for calculate predicted value y_i^p
	real_t qs[_NS], ds[_NS], q[_NS], d[_NS]; // production and loss rate

	int gcount = 0, rcount = 0, iter;
	real_t dt = _DF(0.0);	  // timestep of this flag1 step
	real_t ts, tn = _DF(0.0); // t-t^0, current value of the independent variable relative to the start of the global timestep
	real_t TTn = TT, TT0 = TTn, TTs;

	// // // Initialize and limit y to the minimum value and save the initial yi inputs into y0
	real_t sumy = _DF(0.0);
	for (int i = 0; i < _NS; i++)
		y[i] = sycl::max(y[i], ymin), sumy += y[i];
	sumy = _DF(1.0) / sumy;
	for (int i = 0; i < _NS; i++)
		y[i] *= sumy;

	//=========================================================
	// // initial p and d before predicting
	real_t dTdt = evalcv<_NS, _NR>(tm, rn, y, q, d, rho, TTn, m_p);
	gcount++;
	// // to initilize the first 'dt'
	for (int i = 0; i < _NS; i++)
	{
		const real_t ascr = sycl::fabs(q[i]);
		const real_t scr2 = xfrsign(_DF(1.0) / y[i], _DF(0.1) * epsmin * ascr - d[i]);
		const real_t scr1 = scr2 * d[i];
		// // // If the species is already at the minimum, disregard destruction when calculating step size
		scrtch = sycl::max(-sycl::fabs(ascr - d[i]) * scr2, sycl::max(scr1, scrtch));
	}
	dt = sycl::min(sqreps / scrtch, dtg);
	dtmin *= dt;

	while (1)
	{
		int num_iter = 0;
		// // Independent variable at the start of the chemical timestep
		ts = tn, TTs = TTn;
		for (int i = 0; i < _NS; i++)
		{
			// // store the 0-subscript state using s
			ys[i] = y[i];				// y before prediction
			qs[i] = q[i], ds[i] = d[i]; // q and d before prediction
		}
		dTdt = evalcv<_NS, _NR>(tm, rn, y, q, d, rho, TTn, m_p);

		// a beginning of prediction
	apredictor:
		num_iter++;
		for (int i = 0; i < _NS; i++)
		{
			rtau[i] = dt * ds[i] / ys[i]; // 1/r in Ref.eq(39)
			real_t alpha = Alpha(rtau[i]);
			scrarray[i] = dt * (qs[i] - ds[i]) / (_DF(1.0) + alpha * rtau[i]); // \delta y
		}

		/** begin correction while loop
		 * Iteration for correction, one prediction and itermax correction
		 * if itermax > 1, need add dt recalculator based Ref.eq(48),
		 * or even more restrict requirement Ref.eq(47) for each iter
		 */
		// iter = 1;
		// while (iter <= itermax)
		{
			gcount++;

			for (int i = 0; i < _NS; i++)
			{
				y[i] = sycl::max(ys[i] + scrarray[i], ymin); // predicted y, results stored by y1
			}

			// if (1 == iter)
			{
				tn = ts + dt;
				for (int i = 0; i < _NS; i++)
					y1[i] = y[i]; // predicted y results stored by y1
			}

			// // get predicted q^p , d^p based predictd y
			evalcv<_NS, _NR>(tm, rn, y, q, d, rho, TTs, m_p);

			for (int i = 0; i < _NS; i++)
			{
				const real_t rtaub = _DF(0.5) * (rtau[i] + dt * d[i] / y[i]); // p*dt
				const real_t alpha = Alpha(rtaub);
				const real_t qt = (_DF(1.0) - alpha) * qs[i] + alpha * q[i]; // q
				scrarray[i] = (qt * dt - rtaub * ys[i]) / (_DF(1.0) + alpha * rtaub);
				// y[i] = sycl::max(ys[i] + scrarray[i], ymin); // correctied y
			}
			// iter++;
		} // // // end correction while loop

		// // Calculate new f, check for convergence, and limit decreasing functions
		// // NOTE: The order of operations in this loop is important
		eps = _DF(1e-10);
		for (int i = 0; i < _NS; i++)
		{
			const real_t scr2 = sycl::max(ys[i] + scrarray[i], _DF(0.0));
			real_t scr1 = sycl::fabs(scr2 - y1[i]);
			y[i] = sycl::max(scr2, ymin); // new y

			if ((_DF(0.25) * (ys[i] + y[i])) > ymin)
			{
				scr1 = scr1 / y[i];
				eps = sycl::max(_DF(0.5) * (scr1 + sycl::min(sycl::fabs(q[i] - d[i]) / (q[i] + d[i] + _DF(1.0e-30)), scr1)), eps);
			}
		}

		// // Check for convergence
		// // // The following section is used for the stability check
		eps = eps * epscl;

		if (eps < epsmax)
		{
			if (dtg <= (tn * tfd))
			{
				TT = TTn;
				return; // end of the reaction source solving.
			}
		}
		else
		{
			tn = ts;
		}
		// get new dt
		real_t dto = dt;
		real_t rteps = _DF(0.5) * (eps + _DF(1.0));
		rteps = _DF(0.5) * (rteps + eps / rteps);
		rteps = _DF(0.5) * (rteps + eps / rteps);
		dt = sycl::min(dt * (_DF(1.) / rteps + _DF(0.005)), tfd * (dtg - tn)); // new dt

		// // // Rebegin the step if  this previous step not converged
		if (eps > epsmax)
		{
			// add this operator to reduce dt while this flag2 step isn't convergent, avoid death loop
			dt = sycl::min(dt, _DF(0.34) * dto);
			rcount++;
			// while dt is too small, releax the convergent criterion
			if (dt <= dtmin)
				epsmax *= _DF(10.0);
			// dto = dt / dto;
			// for (int i = 0; i < _NS; i++)
			// 	rtaus[i] = rtaus[i] * dto;
			goto apredictor;
		}

		// // A valid time step has done
		epsmax = _DF(1.0);
		TTn = TTs + dTdt * dto; // new T
		gcount++;
	}
}

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief  get q(production rate) and p(loss rate) from y at constant pressure model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcp(Thermal *tm, Reaction *rn, real_t *y, real_t *q, real_t *p, const real_t m_dens, const real_t m_p)
{
	q[0] = _DF(.0), p[0] = _DF(.0);
	real_t *yi = y + 1, *qi = q + 1, *pi = p + 1, m_tmp = y[0], _dens = _DF(1.0) / m_dens;

	evalcore<_NS, _NR>(tm, rn, yi, qi, pi, m_dens, m_p, m_tmp);

	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		qi[n] *= tm->Wi[n] * _dens;
		pi[n] *= tm->Wi[n] * _dens;
		real_t hi = get_Enthalpy(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
		q[0] -= qi[n] * hi, p[0] -= pi[n] * hi;
	} // dTdt = mass * c_p * dT/dt while the loop ends, we need get dT/dt

	// get q[0], p[0]
	real_t _Cp = _DF(1.0) / get_CopCp(*tm, yi, m_tmp);
	q[0] *= _Cp, p[0] *= _Cp;
}

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief  get q(production rate) and p(loss rate) from y at constant volume model
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void evalcv(Thermal *tm, Reaction *rn, real_t *y, real_t *q, real_t *p, const real_t m_dens, const real_t m_p)
{
	q[0] = _DF(.0), p[0] = _DF(.0);
	real_t *yi = y + 1, *qi = q + 1, *pi = p + 1, m_tmp = y[0], _dens = _DF(1.0) / m_dens;

	evalcore<_NS, _NR>(tm, rn, yi, qi, pi, m_dens, m_p, m_tmp);

	// production rate in the form of mass;
	for (int n = 0; n < _NS; n++)
	{
		qi[n] *= tm->Wi[n] * _dens;
		pi[n] *= tm->Wi[n] * _dens;
		real_t ei = get_Internale(tm->Hia, tm->Hib, m_tmp, tm->Ri[n], n);
		q[0] -= qi[n] * ei, p[0] -= pi[n] * ei;
	} // dTdt = mass * c_v * dT/dt while the loop ends, we need get dT/dt

	// get q[0], p[0]
	real_t _Cv = _DF(1.0) / get_CopCv(*tm, yi, m_tmp);
	q[0] *= _Cv, p[0] *= _Cv;
}

/**
 * @brief NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
 * @brief Chemeq2: q represents the production rate , d represents the los rate , di = pi*yi in RefP408 eq(2)
 * @brief The accuracy-based timestep calculation can be augmented with a stability-based check when at least
 * three corrector iterations are performed. For most problems, the stability check is not needed, and eliminating
 * the calculations and logic associated with the check enhances performance.
 */
template <int _NS = 1, int _NR>
SYCL_DEVICE void Chemeq2(Thermal *tm, Reaction *rn, real_t *y, const real_t dtg, const real_t rho, const real_t m_p)
{
	int itermax = 1;
	const int _NL = _NS + 1;
	real_t ymin = _DF(1.0e-20), dtmin = _DF(1.0e-7);
	real_t eps, epsmin = _DF(1.0e-4), scrtch = _DF(1e-25);
	real_t tfd = _DF(1.0) + _DF(1.0e-10), sqreps = _DF(0.05), epsmax = _DF(1.0), epscl = _DF(1.0e4);

	real_t ys[_NL], y1[_NL];
	real_t rtau[_NL], scrarray[_NL];		 //  \delta y based input y_i value for calculate predicted value y_i^p
	real_t qs[_NL], ds[_NL], q[_NL], d[_NL]; // production and loss rate

	int gcount = 0, rcount = 0, iter;
	real_t dt = _DF(0.0);	  // timestep of this flag1 step
	real_t ts, tn = _DF(0.0); // t-t^0, current value of the independent variable relative to the start of the global timestep

	// // // Initialize and limit y to the minimum value and save the initial yi inputs into y0
	real_t sumy = _DF(0.0);
	for (int i = 1; i < _NL; i++)
		y[i] = sycl::max(y[i], ymin), sumy += y[i];
	sumy = _DF(1.0) / sumy;
	for (int i = 1; i < _NL; i++)
		y[i] *= sumy;

	//=========================================================
	// // initial p and d before predicting
	evalcv<_NS, _NR>(tm, rn, y, q, d, rho, m_p);
	gcount++;
	// // to initilize the first 'dt'
	for (int i = 0; i < _NL; i++)
	{
		const real_t ascr = sycl::fabs(q[i]);
		const real_t scr2 = xfrsign(_DF(1.0) / y[i], _DF(0.1) * epsmin * ascr - d[i]);
		const real_t scr1 = scr2 * d[i];
		// // // If the species is already at the minimum, disregard destruction when calculating step size
		scrtch = sycl::max(-sycl::fabs(ascr - d[i]) * scr2, sycl::max(scr1, scrtch));
	}
	dt = sycl::min(sqreps / scrtch, dtg);
	dtmin *= dt;

	while (1)
	{
		int num_iter = 0;
		// // Independent variable at the start of the chemical timestep
		ts = tn;
		for (int i = 0; i < _NL; i++)
		{
			// // store the 0-subscript state using s
			ys[i] = y[i];				// y before prediction
			qs[i] = q[i], ds[i] = d[i]; // q and d before prediction
		}

		// a beginning of prediction
	apredictor:
		num_iter++;
		for (int i = 0; i < _NL; i++)
		{
			rtau[i] = dt * ds[i] / ys[i]; // 1/r in Ref.eq(39)
			real_t alpha = Alpha(rtau[i]);
			scrarray[i] = dt * (qs[i] - ds[i]) / (_DF(1.0) + alpha * rtau[i]); // \delta y
		}

		/** begin correction while loop
		 * Iteration for correction, one prediction and itermax correction
		 * if itermax > 1, need add dt recalculator based Ref.eq(48),
		 * or even more restrict requirement Ref.eq(47) for each iter
		 */
		// iter = 1;
		// while (iter <= itermax)
		{
			gcount++;

			for (int i = 0; i < _NL; i++)
			{
				y[i] = sycl::max(ys[i] + scrarray[i], ymin); // predicted y, results stored by y1
			}

			// if (1 == iter)
			{
				tn = ts + dt;
				for (int i = 0; i < _NL; i++)
					y1[i] = y[i]; // predicted y results stored by y1
			}

			// // get predicted q^p , d^p based predictd y
			evalcv<_NS, _NR>(tm, rn, y, q, d, rho, m_p);

			for (int i = 0; i < _NL; i++)
			{
				const real_t rtaub = _DF(0.5) * (rtau[i] + dt * d[i] / y[i]); // p*dt
				const real_t alpha = Alpha(rtaub);
				const real_t qt = (_DF(1.0) - alpha) * qs[i] + alpha * q[i]; // q
				scrarray[i] = (qt * dt - rtaub * ys[i]) / (_DF(1.0) + alpha * rtaub);
				// y[i] = sycl::max(ys[i] + scrarray[i], ymin); // correctied y
			}
			// iter++;
		} // // // end correction while loop

		// // Calculate new f, check for convergence, and limit decreasing functions
		// // NOTE: The order of operations in this loop is important
		eps = _DF(1e-10);
		for (int i = 0; i < _NL; i++)
		{
			const real_t scr2 = sycl::max(ys[i] + scrarray[i], _DF(0.0));
			real_t scr1 = sycl::fabs(scr2 - y1[i]);
			y[i] = sycl::max(scr2, ymin); // new y

			if ((_DF(0.25) * (ys[i] + y[i])) > ymin)
			{
				scr1 = scr1 / y[i];
				eps = sycl::max(_DF(0.5) * (scr1 + sycl::min(sycl::fabs(q[i] - d[i]) / (q[i] + d[i] + _DF(1.0e-30)), scr1)), eps);
			}
		}

		// // Check for convergence
		// // // The following section is used for the stability check
		eps = eps * epscl;
		if (eps < epsmax)
		{
			if (dtg <= (tn * tfd))
				return; // end of the reaction source solving.
		}
		else
		{
			tn = ts;
		}
		// get new dt
		real_t dto = dt;
		real_t rteps = _DF(0.5) * (eps + _DF(1.0));
		rteps = _DF(0.5) * (rteps + eps / rteps);
		rteps = _DF(0.5) * (rteps + eps / rteps);
		dt = sycl::min(dt * (_DF(1.) / rteps + _DF(0.005)), tfd * (dtg - tn)); // new dt

		// // // Rebegin the step if  this previous step not converged
		if (eps > epsmax)
		{
			// add this operator to reduce dt while this flag2 step isn't convergent, avoid death loop
			dt = sycl::min(dt, _DF(0.34) * dto);
			rcount++;
			// while dt is too small, releax the convergent criterion
			if (dt <= dtmin)
				epsmax *= _DF(10.0);
			goto apredictor;
		}

		// // A valid time step has done
		epsmax = _DF(1.0);
		gcount++;
	}
}