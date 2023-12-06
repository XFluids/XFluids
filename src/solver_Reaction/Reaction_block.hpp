#include "Reaction_kernels.hpp"

void ZeroDimensionalFreelyFlameBlock(Setup &Ss, const int rank = 0)
{
	real_t xi[NUM_SPECIES], yi[NUM_SPECIES]; // molecular concentration, unit: mol/cm^3; mass fraction
	memcpy(yi, Ss.h_thermal.species_ratio_in, NUM_SPECIES * sizeof(real_t));
	// get_yi(yi, Ss.h_thermal.Wi);
	real_t T0 = _DF(1150.0), p0 = _DF(101325.0);
	real_t R, rho, h, e, T = T0; // h: unit: J/kg // e: enternal energy

	// chemeq2 solver
	real_t t_start = _DF(0.0), t_end = _DF(5e-4), dt = _DF(2.0e-7), run_time = t_start;
	std::string outputPrefix = INI_SAMPLE;
	std::string file_name = OutputDir + "/" + outputPrefix + "-with_0DFreelyFlameTest_Rank_" + std::to_string(rank) + ".dat";
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
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0) + bl.Bwidth_X;
								  int j = index.get_global_id(1) + bl.Bwidth_Y;
								  int k = index.get_global_id(2) + bl.Bwidth_Z;
								  ChemeODEQ2SolverKernel( i, j, k, bl, thermal, react, UI, fdata.y, rho, T, fdata.e, dt); }); })
		.wait();
}