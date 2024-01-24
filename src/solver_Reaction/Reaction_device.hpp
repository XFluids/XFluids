#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

#ifndef NUM_REA
#define NUM_REA 1
#endif // end NUM_REA

/**
 * @brief get_Kf
 */
SYCL_DEVICE real_t get_Kf_ArrheniusLaw(const real_t A, const real_t B, const real_t E, const real_t T)
{
#if defined(Modified_ArrheniusLaw_form)
	/**
	 * @brief: Modified ArrheniusLaw form
	 * *Arrhenius arguments:list A B C , k=A*T^B*exp(-C/T)
	 * *input units C: K, A: cm^3/molecule/s NOTE: 1 mol=NA molecule
	 * *output units: k: cm^3/mol/s
	 */
	return A * sycl::pow(T, B) * sycl::exp(-E / T);
#else
	/**
	 * @brief: Default ArrheniusLaw form
	 * *Arrhenius arguments:list A B E , k=A*T^B*exp(-E/R/T)
	 * *input units E: cal/mol, A: cm^3/mole/s NOTE: 1 cal=4.184 J*mol/K
	 * *output units: k: cm^3/mol/s
	 */
	return A * sycl::pow(T, B) * sycl::exp(-E * _DF(4.184) / Ru / T);
#endif
}

/**
 * @brief get_Kc
 */
SYCL_DEVICE real_t get_Kc(const real_t *_Wi, real_t *__restrict__ Hia, real_t *__restrict__ Hib, int *__restrict__ Nu_d_, const real_t T, const int m)
{
	real_t Kck = _DF(0.0);
	int Nu_sum = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		real_t Ri = Ru * _Wi[n];
		real_t S = get_Gibson(Hia, Hib, T, Ri, n);
		Kck += Nu_d_[m * NUM_SPECIES + n] * S;
		Nu_sum += Nu_d_[m * NUM_SPECIES + n];
	}
	Kck = sycl::exp(Kck);
	Kck *= sycl::pown(p_atm / Ru / T * _DF(1e-6), Nu_sum); // 1e-6: m^-3 -> cm^-3
	return Kck;
}

/**
 * @brief get_KbKf
 */
SYCL_DEVICE void get_KbKf(real_t *Kf, real_t *Kb, real_t *Rargus, real_t *_Wi, real_t *Hia, real_t *Hib, int *Nu_d_, const real_t T)
{
	for (size_t m = 0; m < NUM_REA; m++)
	{
		real_t A = Rargus[m * 6 + 0], B = Rargus[m * 6 + 1], E = Rargus[m * 6 + 2];
		Kf[m] = get_Kf_ArrheniusLaw(A, B, E, T);
		if (BackArre)
		{
			real_t A_ = Rargus[m * 6 + 3], B_ = Rargus[m * 6 + 4], E_ = Rargus[m * 6 + 5];
			Kb[m] = get_Kf_ArrheniusLaw(A_, B_, E_, T);
		}
		else
		{
			real_t Kck = get_Kc(_Wi, Hia, Hib, Nu_d_, T, m);
			Kb[m] = Kf[m] / Kck;
		}
	}
}

/**
 * @brief QSSAFun
 */
SYCL_DEVICE void QSSAFun(real_t *q, real_t *d, real_t *Kf, real_t *Kb, const real_t *yi, Thermal thermal, real_t *React_ThirdCoef,
						 int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
						 int *Nu_b_, int *Nu_f_, int *third_ind, const real_t rho)
{
	real_t C[MAX_SPECIES] = {_DF(0.0)}, _rho = _DF(1.0) / rho, mdcoeff = _DF(1.0), _mdcoeff = _DF(1.0);
	// #if defined(Modified_ArrheniusLaw_form)
	// 	mdcoeff = NA, _mdcoeff = _DF(1.0) / mdcoeff;
	// #endif

	for (int n = 0; n < NUM_SPECIES; n++)
		C[n] = rho * yi[n] * thermal._Wi[n] * _DF(1e-6) * mdcoeff;

	for (int n = 0; n < NUM_SPECIES; n++)
	{
		q[n] = _DF(0.0);
		d[n] = _DF(0.0);
		for (int iter = 0; iter < rns[n]; iter++)
		{
			int react_id = reaction_list[n][iter];
			// third-body collision effect
			real_t tb = _DF(0.0);
			if (1 == third_ind[react_id])
			{
				for (int it = 0; it < NUM_SPECIES; it++)
					tb += React_ThirdCoef[react_id * NUM_SPECIES + it] * C[it];
			}
			else
				tb = _DF(1.0);
			real_t RPf = Kf[react_id], RPb = Kb[react_id];
			// forward
			for (int it = 0; it < rts[react_id]; it++)
			{
				int specie_id = reactant_list[react_id][it];
				int nu_f = Nu_f_[react_id * NUM_SPECIES + specie_id];
				RPf *= sycl::pown(C[specie_id], nu_f);
			}
			// backward
			for (int it = 0; it < pls[react_id]; it++)
			{
				int specie_id = product_list[react_id][it];
				int nu_b = Nu_b_[react_id * NUM_SPECIES + specie_id];
				RPb *= sycl::pown(C[specie_id], nu_b);
			}
			q[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPf + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPb;
			d[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPb + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPf;
		}
		q[n] *= thermal.Wi[n] * _rho * _DF(1.0e6) * _mdcoeff;
		d[n] *= thermal.Wi[n] * _rho * _DF(1.0e6) * _mdcoeff;
	}
}

/**
 * @brief sign for one argus
 */
SYCL_DEVICE real_t frsign(const real_t a)
{
	// if (a > 0)
	// 	return _DF(1.0);
	// else if (0 == a)
	// 	return _DF(0.0);

	if (a >= _DF(.0))
		return _DF(1.0);
	else
		return -_DF(1.0);
}

/**
 * @brief sign for two argus
 */
SYCL_DEVICE real_t frsign(const real_t a, const real_t b)
{
	return frsign(b) * sycl::fabs(a);
}

/**
 * @brief Chemeq2: q represents the production rate , d represents the los rate , di = pi*yi in RefP408 eq(2)
 * @ref   A Quasi-Steady-State Solver for the Stiff Ordinary Differential Equations of Reaction Kinetics
 * @param dtg: duration of time integral
 */
SYCL_DEVICE void Chemeq2(const int id, Thermal thermal, real_t *Kf, real_t *Kb, real_t *React_ThirdCoef, real_t *Rargus, int *Nu_b_, int *Nu_f_, int *Nu_d_,
						 int *third_ind, int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
						 real_t *y, const real_t dtg, real_t &TT, const real_t rho, const real_t e)
{
	int itermax = 1;
	real_t ymin = _DF(1.0e-20), dtmin = _DF(1.0e-7);
	real_t eps, epsmin = _DF(1.0e-4), scrtch = _DF(1e-25);
	real_t tfd = _DF(1.0) + _DF(1.0e-10), sqreps = _DF(0.05), epsmax = _DF(1.0), epscl = _DF(1.0e4);
	/**
	 * @brief The accuracy-based timestep calculation can be augmented with a stability-based check when at least
	 * three corrector iterations are performed. For most problems, the stability check is not needed, and eliminating
	 * the calculations and logic associated with the check enhances performance.
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

	real_t ys[NUM_SPECIES], y1[NUM_SPECIES];
	real_t rtau[NUM_SPECIES], scrarray[NUM_SPECIES];						 //  \delta y based input y_i value for calculate predicted value y_i^p
	real_t qs[NUM_SPECIES], ds[NUM_SPECIES], q[NUM_SPECIES], d[NUM_SPECIES]; // production and loss rate

	int gcount = 0, rcount = 0, iter;
	real_t dt = _DF(0.0);	  // timestep of this flag1 step
	real_t ts, tn = _DF(0.0); // t-t^0, current value of the independent variable relative to the start of the global timestep
	real_t TTn = TT, TT0 = TTn, TTs;

	// // // Initialize and limit y to the minimum value and save the initial yi inputs into y0
	real_t sumy = _DF(0.0);
	for (int i = 0; i < NUM_SPECIES; i++)
		y[i] = sycl::max(y[i], ymin), sumy += y[i];
	sumy = _DF(1.0) / sumy;
	for (int i = 0; i < NUM_SPECIES; i++)
		y[i] *= sumy;

	//=========================================================
	// // initial p and d before predicting
	get_KbKf(Kf, Kb, Rargus, thermal._Wi, thermal.Hia, thermal.Hib, Nu_d_, TTn);
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;
	// // to initilize the first 'dt'
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		const real_t ascr = sycl::fabs(q[i]);
		const real_t scr2 = frsign(_DF(1.0) / y[i], _DF(0.1) * epsmin * ascr - d[i]);
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
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			// // store the 0-subscript state using s
			ys[i] = y[i];				// y before prediction
			qs[i] = q[i], ds[i] = d[i]; // q and d before prediction
		}

// // neomorph of Ref.eq(39) for rtaui=1/r in eq(39)
#define Alpha(rtaui) (_DF(180.0) + rtaui * (_DF(60.0) + rtaui * (_DF(11.0) + rtaui))) / (_DF(360.0) + rtaui * (_DF(60.0) + rtaui * (_DF(12.0) + rtaui)));

		// a beginning of prediction
	apredictor:
		num_iter++;
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			rtau[i] = dt * ds[i] / ys[i]; // 1/r in Ref.eq(39)
			real_t alpha = Alpha(rtau[i]);
			scrarray[i] = dt * (qs[i] - ds[i]) / (_DF(1.0) + alpha * rtau[i]); // \delta y
		}
		// // // begin correction while loop
		// iter = 1;
		// while (iter <= itermax)
		{
			// Iteration for correction, one prediction and itermax correction
			// if itermax > 1, need add dt recalculator based Ref.eq(48), or even more restrict requirement Ref.eq(47) for each iter
			gcount++;

			for (int i = 0; i < NUM_SPECIES; i++)
			{
				y[i] = sycl::max(ys[i] + scrarray[i], ymin); // predicted y, results stored by y1
			}

			// if (1 == iter)
			{
				tn = ts + dt;
				for (int i = 0; i < NUM_SPECIES; i++)
					y1[i] = y[i]; // predicted y results stored by y1
			}

			// // predict T, Kf, Kb, q, d based predicted y
			// TTn = get_T(thermal, y, e, TTs);
			// get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
			// // get predicted q^p , d^p based predictd y
			QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);

			for (int i = 0; i < NUM_SPECIES; i++)
			{
				const real_t rtaub = _DF(0.5) * (rtau[i] + dt * d[i] / y[i]); // p*dt
				const real_t alpha = Alpha(rtaub);
				const real_t qt = (_DF(1.0) - alpha) * qs[i] + alpha * q[i]; // q
				// real_t pb = rtaub / dt;
				scrarray[i] = (qt * dt - rtaub * ys[i]) / (_DF(1.0) + alpha * rtaub);
				// y[i] = sycl::max(ys[i] + scrarray[i], ymin); // correctied y
			}
			// iter++;
		} // // // end correction while loop

		// // Calculate new f, check for convergence, and limit decreasing functions
		// // NOTE: The order of operations in this loop is important
		eps = _DF(1e-10);
		for (int i = 0; i < NUM_SPECIES; i++)
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
				TT = get_T(thermal, y, e, TTn); // final T
				return;							// end of the reaction source solving.
			}
		}
		else
		{
			tn = ts;
		}
		// get new dt
		real_t rteps = _DF(0.5) * (eps + _DF(1.0));
		rteps = _DF(0.5) * (rteps + eps / rteps);
		rteps = _DF(0.5) * (rteps + eps / rteps);
		real_t dto = dt;
		dt = sycl::min(dt * (_DF(1.) / rteps + _DF(0.005)), tfd * (dtg - tn)); // new dt

		// // // Rebegin the step if  this previous step not converged
		if (eps > epsmax)
		{
			dt = sycl::min(dt, _DF(0.34) * dto); // add this operator to reduce dt while this flag2 step isn't convergent, avoid death loop
			rcount++;
			if (dt <= dtmin)
				epsmax *= _DF(10.0);
			// dto = dt / dto;
			// for (int i = 0; i < NUM_SPECIES; i++)
			// 	rtaus[i] = rtaus[i] * dto;
			goto apredictor;
		}

		// // A valid time step has done
		epsmax = _DF(1.0);
		TTn = get_T(thermal, y, e, TTs); // new T
		get_KbKf(Kf, Kb, Rargus, thermal._Wi, thermal.Hia, thermal.Hib, Nu_d_, TTn);
		QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
		gcount++;
	}
}

/**
 * @brief high accuracy and multi inter version
 *
 */
// SYCL_DEVICE void Chemeq2(const int id, Thermal thermal, real_t *Kf, real_t *Kb, real_t *React_ThirdCoef, real_t *Rargus, int *Nu_b_, int *Nu_f_, int *Nu_d_,
// 						 int *third_ind, int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
// 						 real_t *y, const real_t dtg, real_t &TT, const real_t rho, const real_t e)
// {
// 	int itermax = 1;
// 	bool high_level_accuracy_ = (itermax >= 3) ? true : false;
// 	real_t ymin = _DF(1.0e-20), dtmin = _DF(1.0e-7);
// 	real_t eps, epsmin = _DF(1.0e-4), scrtch = _DF(1e-25);
// 	real_t tfd = _DF(1.000008), sqreps = _DF(0.5), epsmax = _DF(1.0), epscl = _DF(1.0E2);
// 	/**
// 	 * @brief The accuracy-based timestep calculation can be augmented with a stability-based check when at least
// 	 * three corrector iterations are performed. For most problems, the stability check is not needed, and eliminating
// 	 * the calculations and logic associated with the check enhances performance.
// 	 * @param itermax: iterations of correction
// 	 * @param high_level_accuracy_: if enable accuracy through stability based check
// 	 * @param ymin: minimum concentration allowed for species i, too much low ymin decrease performance
// 	 * 	*NOTE: initializing time intergation control
// 	 * @param dto: original dt for a integration step
// 	 * @param tfd: round-off parameter used to determine when integration is complete
// 	 * @param epsmin: to calculate initial time step of q2 integral, intializa into _DF(1e-04).
// 	 * @param scrtch: to calculate initial time step of q2 integral, intializa into _DF(1e-25).
// 	 * @param sqreps: 5.0*sycl::sqrt(epsmin), parameter used to calculate initial timestep
// 	 * @param dtmin: minimum dt for each step, automatically relax convergence restrictions while dt<=dtmin*dto for a step .
// 	 * 	*NOTE: epsion contrl
// 	 * @param eps: error epslion, intializa into _DF(1e-10).
// 	 * @param epsmax: if this previous step not converged, higher for low accuracy and higher performace.
// 	 * @param epscl: 1.0/epsmin, intermediate variable used to avoid repeated divisions, higher epscl leading to higher accuracy and lower performace.
// 	 */

// 	real_t ym1_[NUM_SPECIES], ym2_[NUM_SPECIES], rtau[NUM_SPECIES];			 //, rtaus[NUM_SPECIES];
// 	real_t scrarray[NUM_SPECIES];											 //  \delta y based input y_i value for calculate predicted value y_i^p
// 	real_t ys[NUM_SPECIES], y0[NUM_SPECIES], y1[NUM_SPECIES];				 // y0: intial concentrations for the global timestep passed to Chemeq
// 	real_t qs[NUM_SPECIES], ds[NUM_SPECIES], q[NUM_SPECIES], d[NUM_SPECIES]; // production and loss rate

// 	int gcount = 0, rcount = 0, iter;
// 	real_t dt = _DF(0.0); // timestep of this flag1 step
// 	real_t tn = _DF(0.0); // t-t^0, current value of the independent variable relative to the start of the global timestep
// 	real_t ts;			  // independent variable at the start of the global timestep
// 	real_t TTn = TT, TT0 = TTn, TTs;

// 	// // // Initialize and limit y to the minimum value and save the initial yi inputs into y0
// 	for (int i = 0; i < NUM_SPECIES; i++)
// 		y0[i] = y[i], y[i] = sycl::max(y[i], ymin);

// 	real_t *species_chara = thermal.species_chara, *Hia = thermal.Hia, *Hib = thermal.Hib;
// 	//=========================================================
// 	// // initial p and d before predicting
// 	get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
// 	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
// 	gcount++;
// 	// // to initilize the first 'dt'
// 	for (int i = 0; i < NUM_SPECIES; i++)
// 	{
// 		const real_t ascr = sycl::fabs(q[i]);
// 		const real_t scr2 = frsign(_DF(1.0) / y[i], _DF(0.1) * epsmin * ascr - d[i]);
// 		const real_t scr1 = scr2 * d[i];

// 		// // // If the species is already at the minimum, disregard destruction when calculating step size
// 		scrtch = sycl::max(-sycl::fabs(ascr - d[i]) * scr2, sycl::max(scr1, scrtch));
// 	}
// 	dt = sycl::min(sqreps / scrtch, dtg);
// 	dtmin *= dt;

// 	while (1)
// 	{
// 		int num_iter = 0;
// 		// // Independent variable at the start of the chemical timestep
// 		ts = tn, TTs = TTn;
// 		for (int i = 0; i < NUM_SPECIES; i++)
// 		{
// 			// // store the 0-subscript state using s
// 			ys[i] = y[i];				// y before prediction
// 			qs[i] = q[i], ds[i] = d[i]; // q and d before prediction
// 		}

// // // neomorph of Ref.eq(39) for rtaui=1/r in eq(39)
// #define Alpha(rtaui) (_DF(180.0) + rtaui * (_DF(60.0) + rtaui * (_DF(11.0) + rtaui))) / (_DF(360.0) + rtaui * (_DF(60.0) + rtaui * (_DF(12.0) + rtaui)));

// 		// a beginning of prediction
// 	apredictor:
// 		num_iter++;
// 		for (int i = 0; i < NUM_SPECIES; i++)
// 		{
// 			rtau[i] = dt * ds[i] / ys[i]; // 1/r in Ref.eq(39)
// 			real_t alpha = Alpha(rtau[i]);
// 			scrarray[i] = dt * (qs[i] - ds[i]) / (_DF(1.0) + alpha * rtau[i]); // \delta y
// 		}
// 		// // // begin correction while loop
// 		iter = 1;
// 		while (iter <= itermax)
// 		{
// 			// Iteration for correction, one prediction and itermax correction
// 			// if itermax > 1, need add dt recalculator based Ref.eq(48), or even more restrict requirement Ref.eq(47) for each iter
// 			gcount++;

// 			for (int i = 0; i < NUM_SPECIES; i++)
// 			{
// 				if (high_level_accuracy_)
// 					ym2_[i] = ym1_[i], ym1_[i] = y[i];
// 				y[i] = sycl::max(ys[i] + scrarray[i], ymin); // predicted y, results stored by y1
// 			}

// 			if (1 == iter)
// 			{
// 				tn = ts + dt;
// 				for (int i = 0; i < NUM_SPECIES; i++)
// 					y1[i] = y[i]; // predicted y results stored by y1
// 			}

// 			// // predict T, Kf, Kb, q, d based predicted y
// 			// TTn = get_T(thermal, y, e, TTs);
// 			// get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
// 			// // get predicted q^p , d^p based predictd y
// 			QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);

// 			for (int i = 0; i < NUM_SPECIES; i++)
// 			{
// 				const real_t rtaub = _DF(0.5) * (rtau[i] + dt * d[i] / y[i]); // p*dt
// 				const real_t alpha = Alpha(rtaub);
// 				const real_t qt = (_DF(1.0) - alpha) * qs[i] + alpha * q[i]; // q
// 				// real_t pb = rtaub / dt;
// 				scrarray[i] = (qt * dt - rtaub * ys[i]) / (_DF(1.0) + alpha * rtaub);
// 				// y[i] = sycl::max(ys[i] + scrarray[i], ymin); // correctied y
// 			}
// 			iter++;
// 		} // // // end correction while loop

// 		// // Calculate new f, check for convergence, and limit decreasing functions
// 		// // NOTE: The order of operations in this loop is important
// 		eps = _DF(1e-10);
// 		for (int i = 0; i < NUM_SPECIES; i++)
// 		{
// 			const real_t scr2 = sycl::max(ys[i] + scrarray[i], _DF(0.0));
// 			real_t scr1 = sycl::fabs(scr2 - y1[i]);
// 			y[i] = sycl::max(scr2, ymin); // new y

// 			if (high_level_accuracy_)
// 				ym2_[i] = ym1_[i], ym1_[i] = y[i];

// 			if ((_DF(0.25) * (ys[i] + y[i])) > ymin)
// 			{
// 				scr1 = scr1 / y[i];
// 				eps = sycl::max(_DF(0.5) * (scr1 + sycl::min(sycl::fabs(q[i] - d[i]) / (q[i] + d[i] + _DF(1.0e-30)), scr1)), eps);
// 			}
// 		}

// 		// // Check for convergence
// 		// // // The following section is used for the stability check
// 		eps = eps * epscl;
// 		real_t stab = _DF(0.0);
// 		if (high_level_accuracy_)
// 		{
// 			stab = _DF(0.01);
// 			if (itermax >= 3)
// 				for (int i = 0; i < NUM_SPECIES; i++)
// 					stab = sycl::max(stab, sycl::fabs(y[i] - ym1_[i]) / (sycl::fabs(ym1_[i] - ym2_[i]) + _DF(1.0e-20) * y[i]));
// 		}
// 		if (eps < epsmax && stab <= _DF(1.0))
// 		{
// 			if (dtg <= (tn * tfd))
// 			{
// 				TT = get_T(thermal, y, e, TTn); // final T
// 				return;							// end of the reaction source solving.
// 			}
// 		}
// 		else
// 		{
// 			tn = ts;
// 		}
// 		// get new dt
// 		real_t rteps = _DF(0.5) * (eps + _DF(1.0));
// 		rteps = _DF(0.5) * (rteps + eps / rteps);
// 		rteps = _DF(0.5) * (rteps + eps / rteps);
// 		real_t dto = dt;
// 		if (high_level_accuracy_)
// 			dt = sycl::min(dt * (_DF(1.) / rteps + _DF(0.005)), sycl::min(tfd * (dtg - tn), dto / (stab + _DF(0.001))));
// 		else
// 			dt = sycl::min(dt * (_DF(1.) / rteps + _DF(0.005)), tfd * (dtg - tn)); // new dt

// 		// // // Rebegin the step if  this previous step not converged
// 		if (eps > epsmax || stab > _DF(1.0))
// 		{
// 			dt = sycl::min(dt, _DF(0.34) * dto); // add this operator to reduce dt while this flag2 step isn't convergent, avoid death loop
// 			rcount++;
// 			if (dt <= dtmin)
// 				epsmax *= _DF(10.0);
// 			// dto = dt / dto;
// 			// for (int i = 0; i < NUM_SPECIES; i++)
// 			// 	rtaus[i] = rtaus[i] * dto;
// 			goto apredictor;
// 		}

// 		// // A valid time step has done
// 		epsmax = _DF(1.0);
// 		TTn = get_T(thermal, y, e, TTs); // new T
// 		// get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
// 		QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
// 		gcount++;
// 	}
// }
