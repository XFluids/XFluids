#pragma once

#include "Utils_kernels.hpp"

void ZeroDimensionalFreelyFlameKernel(Setup &Ss, const int rank)
{
	real_t xi[NUM_SPECIES];										// molecular concentration, unit: mol/cm^3
	real_t yi[NUM_SPECIES] = {0.2, 0.1, 0, 0, 0, 0, 0, 0, 0.7}; // mass fraction
	get_yi(yi, Ss.h_thermal.Wi);
	real_t T0 = 1150.0, p0 = 101325.0;
	real_t R, rho, h, e, T = T0; // h: unit: J/kg // e: enternal energy

	// chemeq2 solver
	real_t t_start = 0, t_end = 5e-4, dt = 2.0e-7, run_time = t_start;
	std::string outputPrefix = INI_SAMPLE;
	std::string file_name = Ss.OutputDir + "/" + outputPrefix + "-with_0DFreelyFlameTest_Rank_" + std::to_string(rank) + ".plt";
	std::ofstream out(file_name);
	out << "variables= time, <i>T</i>[K]";
	for (size_t n = 0; n < NUM_SPECIES; n++)
		out << ", <i>Y(" << Ss.species_name[n] << ")</i>[-]";
	out << "\nzone t='" << outputPrefix << "'\n";
	/* Solver loop */
	while (run_time < t_end + dt)
	{
		R = get_CopR(Ss.h_thermal._Wi, yi), rho = p0 / R / T;
		h = get_Coph(Ss.h_thermal, yi, T); // unit: J/kg
		e = h - R * T;					   // enternal energy
		// T = get_T(Ss.h_thermal, yi, e, T); // update temperature
		get_xi(xi, yi, Ss.h_thermal._Wi, rho);
		out << run_time << " " << T;
		for (int n = 0; n < NUM_SPECIES; n++)
			out << " " << xi[n];
		out << "\n";

		real_t Kf[NUM_REA], Kb[NUM_REA];																				// yi[NUM_SPECIES],//get_yi(y, yi, id);
		get_KbKf(Kf, Kb, Ss.h_react.Rargus, Ss.h_thermal._Wi, Ss.h_thermal.Hia, Ss.h_thermal.Hib, Ss.h_react.Nu_d_, T); // get_e
		Chemeq2(0, Ss.h_thermal, Kf, Kb, Ss.h_react.React_ThirdCoef, Ss.h_react.Rargus, Ss.h_react.Nu_b_, Ss.h_react.Nu_f_, Ss.h_react.Nu_d_, Ss.h_react.third_ind,
				Ss.h_react.reaction_list, Ss.h_react.reactant_list, Ss.h_react.product_list, Ss.h_react.rns, Ss.h_react.rts, Ss.h_react.pls, yi, dt, T, rho, e);
		run_time += dt;
		// std::cout << "time = " << run_time << ", temp = " << T << "\n";
	}
	out.close();
}

#ifdef COP_CHEME
extern SYCL_EXTERNAL void ChemeODEQ2SolverKernel(int i, int j, int k, Block bl, Thermal thermal, Reaction react, real_t *UI, real_t *y, real_t *rho, real_t *T, real_t *e, const real_t dt)
{
	MARCO_DOMAIN();
#ifdef DIM_X
	if (i >= Xmax - bl.Bwidth_X)
		return;
#endif // DIM_X
#ifdef DIM_Y
	if (j >= Ymax - bl.Bwidth_Y)
		return;
#endif // DIM_Y
#ifdef DIM_Z
	if (k >= Zmax - bl.Bwidth_Z)
		return;
#endif // DIM_Z

	int id = Xmax * Ymax * k + Xmax * j + i;

	real_t Kf[NUM_REA], Kb[NUM_REA], U[Emax - NUM_COP], *yi = &(y[NUM_SPECIES * id]);		   // yi[NUM_SPECIES],//get_yi(y, yi, id);
	get_KbKf(Kf, Kb, react.Rargus, thermal._Wi, thermal.Hia, thermal.Hib, react.Nu_d_, T[id]); // get_e
	// for (size_t n = 0; n < Emax - NUM_COP; n++)
	// {
	//         U[n] = UI[Emax * id + n];
	// }
	// real_t rho1 = _DF(1.0) / U[0];
	// real_t u = U[1] * rho1;
	// real_t v = U[2] * rho1;
	// real_t w = U[3] * rho1;
	// real_t e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	Chemeq2(id, thermal, Kf, Kb, react.React_ThirdCoef, react.Rargus, react.Nu_b_, react.Nu_f_, react.Nu_d_, react.third_ind,
			react.reaction_list, react.reactant_list, react.product_list, react.rns, react.rts, react.pls, yi, dt, T[id], rho[id], e[id]);
	// update partial density according to C0
	for (int n = 0; n < NUM_COP; n++)
	{
		// if (bool(sycl::isnan(yi[n])))
		// {
		// yi[n] = _DF(1.0e-20);
		// }
		UI[Emax * id + n + 5] = yi[n] * rho[id];
	}
}
#endif // end COP_CHEME
