#include "Reaction_kernels.hpp"

#ifndef ZeroDTemperature
#define ZeroDTemperature 1150.0
#endif
#ifndef ZeroDPressure
#define ZeroDPressure 101325.0
#endif
#ifndef ZeroDtStep
#define ZeroDtStep 1.0E-7
#endif
#ifndef ZeroEndTime
#define ZeroEndTime 2.0E-4
#endif
#ifndef ZeroEqu
#define ZeroEqu 1.0E-6
#endif

real_t ZeroDimensionalFreelyFlameBlock(Setup &Ss, const int rank = 0)
{
	real_t xi[NUM_SPECIES], yi[NUM_SPECIES];		  // molecular concentration; mass fraction
	real_t p0 = ODETestRange[0], T0 = ODETestRange[1], equilibrium = ODETestRange[2]; // initial Temperature and Pressure
#ifdef ZeroMassFraction								  // initial Mass Fraction
	std::memcpy(yi, ZeroMassFraction.data(), NUM_SPECIES * sizeof(real_t));
#else
	std::memcpy(yi, Ss.h_thermal.species_ratio_in, NUM_SPECIES * sizeof(real_t));
#endif
	real_t R, rho, h, e, T = T0, Temp = _DF(0.0); // h: unit: J/kg // e: enternal energy
	R = get_CopR(Ss.h_thermal._Wi, yi), rho = p0 / R / T;
	h = get_Coph(Ss.h_thermal, yi, T); // unit: J/kg
	e = h - R * T;					   // enternal energy
	// chemeq2 solver
	std::string outputPrefix = INI_SAMPLE;
	std::string file_name = OutputDir + "/0D-Detonation-" + outputPrefix + "-" + std::to_string(int(T0)) + "K-" + std::to_string(int(p0)) + "Pa" + ".dat";
	std::ofstream out(file_name);
	out << "variables= time(s),Temperature(K)";
	for (size_t n = 0; n < NUM_SPECIES; n++)
		out << "," << Ss.species_name[n];
	// out << "variables= time[s], <i>T</i>[K]";
	for (size_t n = 0; n < NUM_SPECIES; n++)
		out << ", <i>Y(" << Ss.species_name[n] << ")</i>[-]";
	// zone name
	out << "\nzone t='0D-Detonation" << SlipOrder << "'\n";

	/* Solver loop */
	real_t run_time = _DF(0.0), t_end = ODETestRange[3], dt = ODETestRange[4], break_out = dt * ODETestRange[5];
	do
	{
		// std::cout << "";
		// std::cout << "time = " << run_time << ", temp = " << T << "\n";
		get_xi(xi, yi, Ss.h_thermal._Wi, rho);
		out << run_time << " " << T;
		for (int n = 0; n < NUM_SPECIES; n++)
			out << " " << xi[n];
		for (int n = 0; n < NUM_SPECIES; n++)
			out << " " << yi[n];
		out << "\n";

		Temp = T;
		real_t Kf[NUM_REA], Kb[NUM_REA];
		Chemeq2(0, Ss.h_thermal, Kf, Kb, Ss.h_react.React_ThirdCoef, Ss.h_react.Rargus, Ss.h_react.Nu_b_, Ss.h_react.Nu_f_, Ss.h_react.Nu_d_, Ss.h_react.third_ind,
				Ss.h_react.reaction_list, Ss.h_react.reactant_list, Ss.h_react.product_list, Ss.h_react.rns, Ss.h_react.rts, Ss.h_react.pls, yi, dt, T, rho, e);
		run_time += dt;

		// real_t yn2b_ = _DF(1.0) / (_DF(1.0) - yi[NUM_SPECIES - 1]);
		// for (int n = 0; n < NUM_SPECIES - 1; n++)
		// 	yi[n] *= (yn2_ * yn2b_);
		if (run_time > break_out)
			break;
	} while (std::fabs(Temp - T) >= equilibrium || run_time < t_end);

	out.close();
	// std::cout << "\n";
	std::cout << "0D reaction testing beginning at " << T0 << "K, " << p0 << "Pa done.\n";
	return Temp;
}

void ChemeODEQ2Solver(sycl::queue &q, Block bl, Thermal thermal, FlowData &fdata, real_t *UI, Reaction react, const real_t dt)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
				   {
								  int i = index.get_global_id(0) + bl.Bwidth_X;
								  int j = index.get_global_id(1) + bl.Bwidth_Y;
								  int k = index.get_global_id(2) + bl.Bwidth_Z;
								  ChemeODEQ2SolverKernel( i, j, k, bl, thermal, react, UI, fdata.y, rho, T, fdata.e, dt); }); })
		.wait();
}