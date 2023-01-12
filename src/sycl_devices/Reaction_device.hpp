#pragma once

#include "schemes_device.hpp"
#include "Thermo_device.hpp"

/**
 * @brief get_Kf
 */
real_t get_Kf_ArrheniusLaw(const real_t A, const real_t B, const real_t E, const real_t T)
{
	return A * sycl::pow<real_t>(T, B) * sycl::exp(-E * 4.184 / Ru / T);
}

/**
 * @brief get_Kc
 */
real_t get_Kc(const real_t *_Wi, real_t *__restrict__ Hia, real_t *__restrict__ Hib, int *__restrict__ Nu_d_, const real_t T, const int m)
{
	real_t Kck = _DF(0.0), Nu_sum = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		real_t Ri = Ru * _Wi[n];
		real_t S = get_Gibson(Hia, Hib, T, Ri, n);
		Kck += Nu_d_[m * NUM_SPECIES + n] * S;
		Nu_sum += Nu_d_[m * NUM_SPECIES + n];
	}
	Kck = sycl::exp(Kck);
	Kck *= sycl::pow<real_t>(p_atm / Ru / T * 1e-6, Nu_sum); // 1e-6: m^-3 -> cm^-3
	return Kck;
}

/**
 * @brief get_KbKf
 */
void get_KbKf(real_t *Kf, real_t *Kb, real_t *Rargus, real_t *_Wi, real_t *Hia, real_t *Hib, int *Nu_d_, const real_t T)
{
	for (size_t m = 0; m < NUM_REA; m++)
	{
		real_t A = Rargus[m * 6 + 0], B = Rargus[m * 6 + 1], E = Rargus[m * 6 + 2];
#if CJ
		Kf[m] = sycl::min<real_t>((20 * _DF(1.0)), A * sycl::pow<real_t>(T, B) * sycl::exp(-E / T));
		Kb[m] = _DF(0.0);
#else
		Kf[m] = get_Kf_ArrheniusLaw(A, B, E, T);
		real_t Kck = get_Kc(_Wi, Hia, Hib, Nu_d_, T, m);
		Kb[m] = Kf[m] / Kck;
#endif
	}
}

/**
 * @brief QSSAFun
 */
void QSSAFun(real_t *q, real_t *d, real_t *Kf, real_t *Kb, const real_t yi[NUM_SPECIES], Thermal thermal, real_t *React_ThirdCoef,
			 int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
			 int *Nu_b_, int *Nu_f_, int *third_ind, const real_t rho)
{
	real_t C[NUM_SPECIES] = {_DF(0.0)}, _rho = _DF(1.0) / rho;
	for (int n = 0; n < NUM_SPECIES; n++)
		C[n] = rho * yi[n] * thermal._Wi[n] * _DF(1e-6);

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
				tb = 1.0;
			double RPf = Kf[react_id], RPb = Kb[react_id];
			// forward
			for (int it = 0; it < rts[react_id]; it++)
			{
				int specie_id = reactant_list[react_id][it];
				int nu_f = Nu_f_[react_id * NUM_SPECIES + specie_id];
				RPf *= sycl::pow<real_t>(C[specie_id], nu_f);
			}
			// backward
			for (int it = 0; it < pls[react_id]; it++)
			{
				int specie_id = product_list[react_id][it];
				int nu_b = Nu_b_[react_id * NUM_SPECIES + specie_id];
				RPb *= sycl::pow<real_t>(C[specie_id], nu_b);
			}
			q[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPf + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPb;
			d[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPb + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPf;
		}
		q[n] *= thermal.Wi[n] * _rho * _DF(1.0e6);
		d[n] *= thermal.Wi[n] * _rho * _DF(1.0e6);
	}
}

/**
 * @brief sign for one argus
 */
real_t sign(real_t a)
{
	if (a > 0)
		return _DF(1.0);
	else if (0 == a)
		return _DF(0.0);
	else
		return -_DF(1.0);
}

/**
 * @brief sign for two argus
 */
real_t sign(real_t a, real_t b)
{
	return sign(b) * sycl::abs(a);
}

/**
 * @brief Chemeq2
 */
void Chemeq2(const int id, Thermal thermal, real_t *Kf, real_t *Kb, real_t *React_ThirdCoef, real_t *Rargus, int *Nu_b_, int *Nu_f_, int *Nu_d_,
			 int *third_ind, int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
			 real_t y[NUM_SPECIES], const real_t dtg, real_t &TT, const real_t rho, const real_t e)
{
	/**NOTE: q represents the production rate , d represents the los rate , di = pi*yi in RefP408 eq(2)
	 * Ref.A Quasi-Steady-State Solver for the Stiff Ordinary Differential Equations of Reaction Kinetics
	 */
	//  parameter
	int itermax = 1;					  // iterations of correction, itermax > 1 haven't supported
	real_t epscl = _DF(100.0);			  // 1/epsmin, intermediate variable used to avoid repeated divisions, higher epscl leading to higher accuracy and lower performace
	real_t tfd = _DF(1.0) + _DF(1.0e-10); // round-off parameter used to determine when integration is complete
	real_t dtmin = _DF(1.0e-20);
	real_t sqreps = _DF(0.5); // 5*sqrt(\eps, parameter used to calculate initial timestep
							  // || \delta y_i^{c(Nc-1)} ||/(||\delta y_i^{c(Nc)} ||)
	real_t epsmax = _DF(1.0), epsmin = _DF(1.0e-4);
	real_t scrtch = _DF(1e-25), ymin = _DF(1.0e-20);			   // ymin: minimum concentration allowed for species i, too much low ymin decrease performance
	real_t eps, ys[NUM_SPECIES], y0[NUM_SPECIES], y1[NUM_SPECIES]; // y0: intial concentrations for the global timestep passed to Chemeq
	real_t scrarray[NUM_SPECIES], scrarraym[NUM_SPECIES];		   // y_i^p, predicted value from Eq. (35)
	real_t deltascr[NUM_SPECIES], deltascrm[NUM_SPECIES];
	real_t rtau[NUM_SPECIES], rtaus[NUM_SPECIES];							 // deprecated.
	real_t qs[NUM_SPECIES], ds[NUM_SPECIES], q[NUM_SPECIES], d[NUM_SPECIES]; // production and loss rate
	int gcount = 0, rcount = 0, iter;
	real_t dt = _DF(0.0); // timestep of this flag1 step
	real_t tn = _DF(0.0); // t-t^0, current value of the independent variable relative to the start of the global timestep
	real_t ts;			  // independent variable at the start of the global timestep
	real_t TTn = TT, TT0 = TTn, TTs;
	// // save the initial inputs
	real_t sumy = _DF(0.0);
	for (int i = 0; i < NUM_SPECIES; i++)
		y0[i] = y[i], y[i] = sycl::max(y[i], ymin), sumy += y[i];
	sumy = _DF(1.0) / sumy;
	for (int i = 0; i < NUM_SPECIES; i++)
		y[i] *= sumy;
	real_t *species_chara = thermal.species_chara, *Hia = thermal.Hia, *Hib = thermal.Hib;
	//=========================================================
	// initial p and d before predicting
	get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;
	// to initilize the first 'dt'
	real_t ascr = _DF(0.0), scr1 = _DF(0.0), scr2 = _DF(0.0); // scratch (temporary) variable
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		ascr = sycl::abs(q[i]);
		scr2 = sign(_DF(1.0) / y[i], _DF(.1) * epsmin * ascr - d[i]);
		scr1 = scr2 * d[i];
		scrtch = sycl::max(scr1, scrtch);
		scrtch = sycl::max(scrtch, -sycl::abs(ascr - d[i]) * scr2);
	}
	dt = sycl::min(sqreps / scrtch, dtg);

	//==========================================================
flag1:
	int num_iter = 0;
	ts = tn;
	TTs = TTn;
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		// store the 0-subscript state using s
		ys[i] = y[i];				// y before prediction
		qs[i] = q[i], ds[i] = d[i]; // q and d before prediction
	}

// neomorph of Ref.eq(39) for rtau=1/r in eq(39)
#define Alpha(rtau) (_DF(180.0) + rtau * (_DF(60.0) + rtau * (_DF(11.0) + rtau))) / (_DF(360.0) + rtau * (_DF(60.0) + rtau * (_DF(12.0) + rtau)));

flag2:
	num_iter++;
	// prediction
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		rtau[i] = dt * ds[i] / ys[i]; // 1/r in Ref.eq(39)
		real_t alpha = Alpha(rtau[i]);
		scrarray[i] = dt * (qs[i] - ds[i]) / (_DF(1.0) + alpha * rtau[i]); // \delta y
		y[i] = sycl::max(ys[i] + scrarray[i], ymin), y1[i] = y[i];		   // predicted y, results stored by y1
	}
	tn = ts + dt;
	// // predict T, Kf, and Kb based predicted y, the predicted assumed not accurate, only update q, d use predicted y excluded T and Kf, Kb
	// TTn = get_T(thermal, y, e, TTs);
	// get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
	// // get predicted q^p , d^p based predictd y
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	iter = 1;
	while (iter <= itermax)
	{
		// Iteration for correction, one prediction and itermax correction
		// if itermax > 1, need add dt recalculator based Ref.eq(48), or even more restrict requirement Ref.eq(47) for each iter
		gcount++;
		eps = 1.0e-10;
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			real_t rtaub = 0.5 * (rtau[i] + dt * d[i] / y[i]); // p*dt
			real_t alpha = Alpha(rtaub);
			real_t qt = (1.0 - alpha) * qs[i] + alpha * q[i]; // q
			// real_t pb = rtaub / dt;
			// scrarraym[i] = scrarray[i];
			scrarray[i] = (qt * dt - rtaub * ys[i]) / (1.0 + alpha * rtaub);
			y[i] = sycl::max(ys[i] + scrarray[i], ymin); // correctied y

			// deltascr[i] = sycl::min(scrarray[i] - scrarraym[i]); // get \delta y^c(itermax)
			// if (iter < itermax)
			// 	deltascrm[i] = deltascr[i];//get \delta y^c(itermax-1)
		}
		iter++;
		// {
		// 	// these three step is needn't for itermax==1 but must need for itermax > 1
		// 	TTn = get_T(thermal, y, e, TTn);
		// 	get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
		// 	// get the corrected q^c and d^c
		// 	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
		// }
	}
	// check convergence
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		// scr2 = sycl::max(ys[i] + scrarray[i], ymin);
		// scr1 = sycl::abs(scr2 - y1[i]);
		// y[i] = sycl::max(scr2, ymin); // new y
		scr1 = sycl::abs(y[i] - y1[i]);
		if (_DF(0.5) * _DF(0.5) * (ys[i] + y[i]) > ymin)
		{
			scr1 = scr1 / y[i];
			// eps = sycl::max(scr1, eps);
			eps = sycl::max(_DF(0.5) * (scr1 + sycl::min(sycl::abs(q[i] - d[i] + real_t(1.0e-100)) / (q[i] + d[i] + real_t(1.0e-100)), scr1)), eps);
		}
	}
	eps = eps * epscl;
	if (eps < epsmax)
	{
		if (dtg < (tn * tfd))
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
	real_t rteps = 0.5 * (eps + 1.0);
	rteps = 0.5 * (rteps + eps / rteps);
	rteps = 0.5 * (rteps + eps / rteps);
	real_t dto = dt;
	dt = sycl::min(dt * (1.0 / rteps + real_t(0.005)), (dtg - tn)); // tfd * (dtg - tn)); // new dt
	if (eps > epsmax)
	{
		dt = sycl::min(dt, 0.34 * dto); // add this operator to reduce dt while this flag2 step isn't convergent, avoid death loop
		rcount++;
		// dto = dt / dto;
		// for (int i = 0; i < NUM_SPECIES; i++)
		// 	rtaus[i] = rtaus[i] * dto;
		goto flag2;
	}

// A valid time step has done
flag3:
	TTn = get_T(thermal, y, e, TTs); // new T
	get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;
	goto flag1;
}