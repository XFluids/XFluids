#include "global_class.h"
#include "block_sycl.hpp"

Fluid::Fluid(Setup &setup) : Fs(setup), q(setup.q), rank(0), nranks(1), SBIOutIter(0)
{
	MPI_BCs_time = 0.0;
	MPI_trans_time = 0.0;
	error_patched_times = 0;
#ifdef USE_MPI
	rank = Fs.mpiTrans->myRank;
	nranks = Fs.mpiTrans->nProcs;
#endif
	// for (size_t i = 0; i < 3; i++)
	// 	thetas[i].clear();
	// for (size_t i = 0; i < NUM_SPECIES - 3; i++)
	// 	Var_max[i].clear();
	// for (size_t j = 0; j < 6; j++)
	// 	Interface_points[j].clear();
	// pTime.clear(), Theta.clear(), Sigma.clear();

#ifdef ODESolverTest
	if (0 == rank)
		ZeroDimensionalFreelyFlame();
#endif // end ODESolverTest

		// Counts file
#ifdef SBICounts
	outputPrefix = INI_SAMPLE;
	file_name = Fs.OutputDir + "/AllSBICounts_" + outputPrefix + ".plt";
	if (Fs.myRank == 0)
	{
		std::fstream if_exist;
		if_exist.open(file_name, std::ios::in);
		if (!if_exist)
		{
			std::ofstream out(file_name, std::fstream::out);
			out.setf(std::ios::right);
			// // defining header for tecplot(plot software)
			out << "title='Time_Theta_Lambda_Sigma_NSigma";
			//" << outputPrefix << "'\n"
			// Sigma: Sum(Omega^2), NSigma(Nromalized Sigma): Sum(rho[id]*Omega^2)
#ifdef COP_CHEME
			out << "_T";
			for (size_t n = 1; n < NUM_SPECIES - 3; n++)
				out << "_Yi" << Fs.species_name[n + 1];
#endif // end COP_CHEME
			out << "_Step_teXN_teXeN2_teXe_Ymin_Ymax";
			// #if DIM_X
			// 			out << "_Xmin_Xmax_Lambdax";
			// #endif // end DIM_X
			// #if DIM_Z
			// 			out << "_Zmin_Zmax_Lambdaz";
			// #endif // end DIM_Z
			out << "'\nvariables=Time[s], <b><greek>Q</greek></b>[-], <greek>L</greek><sub>y</sub>[-], ";
			out << "<greek>e</greek><sub><greek>r</greek></sub>[m<sup>2</sup>/s<sup>2</sup>], ";
			out << "<greek>e</greek><sub><greek>r</greek>n</sub>[m<sup>2</sup>/s<sup>2</sup>], ";
			// Time[s]: Time in tecplot x-Axis variable
			//<greek>Q</greek>[-]: (theta(Theta(XN)/Theta(Xe)/Theta(N2))) in tecplot
			//<greek>L</greek><sub>y</sub>[-]: Lambday in tecplot
			//<greek>e</greek><sub><greek>r</greek></sub>: sigma in tecplot
			//<sub>max</sub>: T_max in tecplot sub{max} added
			//<i>Y(HO2)</i><sub>max</sub>[-]: Yi(HO2)_max

#ifdef COP_CHEME
			out << "<i>T</i><sub>max</sub>[K], ";
			for (size_t n = 1; n < NUM_SPECIES - 3; n++)
				out << "<i>Y(" << Fs.species_name[n + 1] << ")</i><sub>max</sub>[-], ";
#endif // end COP_CHEME
			out << "Step, Theta(XN), Theta(Xe*N2), Theta(Xe), ";
			out << "Ymin, Ymax";
			// #if DIM_X
			// 			out << ", Xmin, Xmax, <greek>L</greek><sub>x</sub>[-]";
			// #endif
			// #if DIM_Z
			// 			out << ", Zmin, Zmax, <greek>L</greek><sub>z</sub>[-]";
			// #endif
			out << "\nzone t='" << outputPrefix << "'\n";
			out.close();
		}
	}
#endif // end SBICounts
}

Fluid::~Fluid()
{
	// 设备内存
	sycl::free(d_U, q);
	sycl::free(d_U1, q);
	sycl::free(d_LU, q);
	sycl::free(Ubak, q);
	sycl::free(d_eigen_local_x, q);
	sycl::free(d_eigen_local_y, q);
	sycl::free(d_eigen_local_z, q);
	sycl::free(d_fstate.rho, q);
	sycl::free(d_fstate.p, q);
	sycl::free(d_fstate.c, q);
	sycl::free(d_fstate.H, q);
	sycl::free(d_fstate.u, q);
	sycl::free(d_fstate.v, q);
	sycl::free(d_fstate.w, q);
	sycl::free(d_fstate.T, q);
	sycl::free(d_fstate.y, q);
	sycl::free(d_fstate.e, q);
	sycl::free(d_fstate.gamma, q);
	sycl::free(d_fstate.thetaXe, q);
	sycl::free(d_fstate.thetaN2, q);
	sycl::free(d_fstate.thetaXN, q);

	// 释放主机内存
	sycl::free(h_fstate.rho, q);
	sycl::free(h_fstate.p, q);
	sycl::free(h_fstate.c, q);
	sycl::free(h_fstate.H, q);
	sycl::free(h_fstate.u, q);
	sycl::free(h_fstate.v, q);
	sycl::free(h_fstate.w, q);
	sycl::free(h_fstate.T, q);
	sycl::free(h_fstate.y, q);
	sycl::free(h_fstate.e, q);
	sycl::free(h_fstate.gamma, q);
#if 2 == EIGEN_ALLOC
	sycl::free(d_eigen_l, q);
	sycl::free(d_eigen_r, q);
#endif // end EIGEN_ALLOC

	sycl::free(d_fstate.vx, q), sycl::free(h_fstate.vx, q);
	for (size_t i = 0; i < 3; i++)
		sycl::free(d_fstate.vxs[i], q), sycl::free(h_fstate.vxs[i], q);
	for (size_t i = 0; i < 9; i++)
		sycl::free(d_fstate.Vde[i], q);

#ifdef Visc // free viscous Vars
	sycl::free(d_fstate.viscosity_aver, q);
#ifdef Visc_Heat
	sycl::free(d_fstate.thermal_conduct_aver, q);
#endif // end Visc_Heat
#ifdef Visc_Diffu
	sycl::free(d_fstate.hi, q);
	sycl::free(d_fstate.Dkm_aver, q);
#endif // end Visc_Diffu
#endif // end Visc

	sycl::free(d_FluxF, q);
	sycl::free(d_FluxG, q);
	sycl::free(d_FluxH, q);
	sycl::free(d_wallFluxF, q);
	sycl::free(d_wallFluxG, q);
	sycl::free(d_wallFluxH, q);
	sycl::free(theta, q);
	sycl::free(sigma, q);
	sycl::free(pVar_max, q);
	sycl::free(uvw_c_max, q);
	sycl::free(interface_point, q);

#ifdef ESTIM_NAN
	sycl::free(h_U, q);
	sycl::free(h_U1, q);
	sycl::free(h_LU, q);
#if DIM_X
	sycl::free(h_fstate.b1x, q);
	sycl::free(h_fstate.b3x, q);
	sycl::free(h_fstate.c2x, q);
	sycl::free(h_fstate.zix, q);
	sycl::free(d_fstate.b1x, q);
	sycl::free(d_fstate.b3x, q);
	sycl::free(d_fstate.c2x, q);
	sycl::free(d_fstate.zix, q);

	sycl::free(h_fstate.preFwx, q);
	sycl::free(h_fstate.pstFwx, q);
	sycl::free(d_fstate.preFwx, q);
#endif
#if DIM_Y
	sycl::free(h_fstate.b1y, q);
	sycl::free(h_fstate.b3y, q);
	sycl::free(h_fstate.c2y, q);
	sycl::free(h_fstate.ziy, q);
	sycl::free(d_fstate.b1y, q);
	sycl::free(d_fstate.b3y, q);
	sycl::free(d_fstate.c2y, q);
	sycl::free(d_fstate.ziy, q);

	sycl::free(h_fstate.preFwy, q);
	sycl::free(h_fstate.pstFwy, q);
	sycl::free(d_fstate.preFwy, q);
#endif
#if DIM_Z
	sycl::free(h_fstate.b1z, q);
	sycl::free(h_fstate.b3z, q);
	sycl::free(h_fstate.c2z, q);
	sycl::free(h_fstate.ziz, q);
	sycl::free(d_fstate.b1z, q);
	sycl::free(d_fstate.b3z, q);
	sycl::free(d_fstate.c2z, q);
	sycl::free(d_fstate.ziz, q);

	sycl::free(h_fstate.preFwz, q);
	sycl::free(h_fstate.pstFwz, q);
	sycl::free(d_fstate.preFwz, q);
#endif

#ifdef Visc // free viscous estimate Vars
#if DIM_X
	sycl::free(h_fstate.visFwx, q);
	sycl::free(d_fstate.visFwx, q);
#endif // end DIM_X
#if DIM_Y
	sycl::free(h_fstate.visFwy, q);
	sycl::free(d_fstate.visFwy, q);
#endif // end DIM_Y
#if DIM_Z
	sycl::free(h_fstate.visFwz, q);
	sycl::free(d_fstate.visFwz, q);
#endif // end DIM_Z
#ifdef Visc_Diffu
	sycl::free(h_fstate.Ertemp1, q);
	sycl::free(h_fstate.Ertemp2, q);
	sycl::free(h_fstate.Dkm_aver, q);
	sycl::free(d_fstate.Ertemp1, q);
	sycl::free(d_fstate.Ertemp2, q);

#if DIM_X
	sycl::free(h_fstate.Dim_wallx, q);
	sycl::free(h_fstate.hi_wallx, q);
	sycl::free(h_fstate.Yi_wallx, q);
	sycl::free(h_fstate.Yil_wallx, q);
	sycl::free(d_fstate.Dim_wallx, q);
	sycl::free(d_fstate.hi_wallx, q);
	sycl::free(d_fstate.Yi_wallx, q);
	sycl::free(d_fstate.Yil_wallx, q);
#endif
#if DIM_Y
	sycl::free(h_fstate.Dim_wally, q);
	sycl::free(h_fstate.hi_wally, q);
	sycl::free(h_fstate.Yi_wally, q);
	sycl::free(h_fstate.Yil_wally, q);
	sycl::free(d_fstate.Dim_wally, q);
	sycl::free(d_fstate.hi_wally, q);
	sycl::free(d_fstate.Yi_wally, q);
	sycl::free(d_fstate.Yil_wally, q);
#endif
#if DIM_Z
	sycl::free(h_fstate.Dim_wallz, q);
	sycl::free(h_fstate.hi_wallz, q);
	sycl::free(h_fstate.Yi_wallz, q);
	sycl::free(h_fstate.Yil_wallz, q);
	sycl::free(d_fstate.Dim_wallz, q);
	sycl::free(d_fstate.hi_wallz, q);
	sycl::free(d_fstate.Yi_wallz, q);
	sycl::free(d_fstate.Yil_wallz, q);
#endif

#endif // end Visc_Diffu
#endif // end Visc
#endif // ESTIM_NAN

	// for (size_t i = 0; i < 3; i++)
	// 	thetas[i].clear(), std::vector<real_t>().swap(thetas[i]);
	// for (size_t i = 0; i < NUM_SPECIES - 3; i++)
	// 	Var_max[i].clear(), std::vector<real_t>().swap(Var_max[i]);
	// for (size_t j = 0; j < 6; j++)
	// 	Interface_points[j].clear(), std::vector<real_t>().swap(Interface_points[j]);
	// pTime.clear(), std::vector<real_t>().swap(pTime);
	// Theta.clear(), std::vector<real_t>().swap(Theta);
	// Sigma.clear(), std::vector<real_t>().swap(Sigma);
}

void Fluid::initialize(int n)
{
	Fluid_name = Fs.fname[n]; // give a name to the fluid
	// type of material, 0: gamma gas, 1: water, 2: stiff gas
	material_property.Mtrl_ind = Fs.material_kind[n];
	// fluid indicator and EOS Parameters
	material_property.Rgn_ind = Fs.material_props[n][0];
	// gamma, A, B, rho0, mu_0, R_0, lambda_0
	material_property.Gamma = Fs.material_props[n][1];
	material_property.A = Fs.material_props[n][2];
	material_property.B = Fs.material_props[n][3];
	material_property.rho0 = Fs.material_props[n][4];
	material_property.R_0 = Fs.material_props[n][5];
	material_property.lambda_0 = Fs.material_props[n][6];
}

void Fluid::AllocateFluidMemory(sycl::queue &q)
{
	int bytes = Fs.bytes;
	int cellbytes = Fs.cellbytes;
	// 主机内存
	h_fstate.rho = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.p = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.H = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.u = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.v = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.w = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.T = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.y = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_SPECIES, q));
	h_fstate.e = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.gamma = static_cast<real_t *>(sycl::malloc_host(bytes, q));

	// 设备内存
	d_U = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_U1 = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_LU = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	Ubak = static_cast<real_t *>(sycl::malloc_shared(cellbytes, q)); // shared memory may inside device
	d_eigen_local_x = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_eigen_local_y = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_eigen_local_z = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_fstate.rho = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.p = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.H = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.u = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.v = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.w = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.T = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.y = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_SPECIES, q));
	d_fstate.e = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.gamma = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize = (7.0 * (double(cellbytes) / 1024.0) + (10.0 + NUM_SPECIES) * (double(bytes) / 1024.0)) / 1024.0;
	int Size = Fs.BlSz.X_inner * Fs.BlSz.Z_inner * sizeof(real_t);
	d_fstate.thetaXe = static_cast<real_t *>(sycl::malloc_shared(Size, q));
	d_fstate.thetaN2 = static_cast<real_t *>(sycl::malloc_shared(Size, q));
	d_fstate.thetaXN = static_cast<real_t *>(sycl::malloc_shared(Size, q));
	MemMbSize += 3.0 * real_t(Size / 1024.0) / 1024.0;

#if 2 == EIGEN_ALLOC
	d_eigen_l = static_cast<real_t *>(sycl::malloc_device(cellbytes * Emax, q));
	d_eigen_r = static_cast<real_t *>(sycl::malloc_device(cellbytes * Emax, q));
	MemMbSize += ((double(cellbytes) / 1024.0) * 2 * Emax) / 1024.0;
#endif // end EIGEN_ALLOC

	for (size_t i = 0; i < 9; i++)
		d_fstate.Vde[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0 * 10.0;
	for (size_t i = 0; i < 3; i++)
		h_fstate.vxs[i] = static_cast<real_t *>(sycl::malloc_host(bytes, q)), d_fstate.vxs[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	h_fstate.vx = static_cast<real_t *>(sycl::malloc_host(bytes, q)); // vorticity.
	d_fstate.vx = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += 4.0 * bytes / 1024.0 / 1024.0;

#ifdef Visc // allocate mem viscous Vars
	d_fstate.viscosity_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
#ifdef Visc_Heat
	d_fstate.thermal_conduct_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0;
#endif // end Visc_Heat
#ifdef Visc_Diffu
	d_fstate.hi = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	MemMbSize += NUM_SPECIES * bytes / 1024.0 / 1024.0 * 2.0;
#endif // end Visc_Diffu
#endif // end Visc

	d_FluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	MemMbSize += cellbytes / 1024.0 / 1024.0 * 6.0;
	// shared memory
	uvw_c_max = static_cast<real_t *>(sycl::malloc_shared(6 * sizeof(real_t), q));
	eigen_block_x = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	eigen_block_y = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	eigen_block_z = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	pVar_max = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 1 - 4) * sizeof(real_t), q)); // calculate Tmax, YiH2O2max, YiHO2max
	sigma = static_cast<real_t *>(sycl::malloc_shared(2 * sizeof(real_t), q));						  // Ref:https://linkinghub.elsevier.com/retrieve/pii/S0010218015003648.eq(34)
	theta = static_cast<real_t *>(sycl::malloc_shared(3 * sizeof(real_t), q));						  // Yi(Xe),Yi(N2),Yi(Xe*N2)// Ref.eq(35)
	interface_point = static_cast<real_t *>(sycl::malloc_shared(6 * sizeof(real_t), q));

#ifdef ESTIM_NAN
	h_U = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_U1 = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_LU = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
#if DIM_X
	h_fstate.b1x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.zix = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
	d_fstate.b1x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.zix = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));

	h_fstate.preFwx = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_fstate.pstFwx = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.preFwx = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif
#if DIM_Y
	h_fstate.b1y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.ziy = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
	d_fstate.b1y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.ziy = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));

	h_fstate.preFwy = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_fstate.pstFwy = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.preFwy = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif
#if DIM_Z
	h_fstate.b1z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.ziz = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
	d_fstate.b1z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.ziz = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));

	h_fstate.preFwz = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_fstate.pstFwz = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.preFwz = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif
	MemMbSize += ((double(cellbytes) / 1024.0)) / 1024.0 * double((DIM_X + DIM_Y + DIM_Z));
	// MemMbSize += ((double(bytes) / 1024.0)) / 1024.0 * double((DIM_X + DIM_Y + DIM_Z) * (NUM_COP + 3));

#ifdef Visc // allocate viscous estimating Vars
#if DIM_X
	h_fstate.visFwx = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.visFwx = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif // end DIM_X
#if DIM_Y
	h_fstate.visFwy = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.visFwy = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif // end DIM_Y
#if DIM_Z
	h_fstate.visFwz = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	d_fstate.visFwz = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
#endif // end DIM_Z
	MemMbSize += ((double(cellbytes) / 1024.0)) / 1024.0 * double(DIM_X + DIM_Y + DIM_Z);
#ifdef Visc_Diffu
	h_fstate.Ertemp1 = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Ertemp2 = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	d_fstate.Ertemp1 = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Ertemp2 = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
#if DIM_X
	h_fstate.Dim_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.hi_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yi_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yil_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	d_fstate.Dim_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.hi_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yi_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yil_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
#endif
#if DIM_Y
	h_fstate.Dim_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.hi_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yi_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yil_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	d_fstate.Dim_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.hi_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yi_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yil_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
#endif
#if DIM_Z
	h_fstate.Dim_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.hi_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yi_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	h_fstate.Yil_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
	d_fstate.Dim_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.hi_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yi_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Yil_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
#endif
	MemMbSize += ((double(bytes) / 1024.0) * (3.0 + 4.0 * double(DIM_X + DIM_Y + DIM_Z))) / 1024.0 * double(NUM_SPECIES);
#endif // end Visc_Diffu
#endif // end Visc
#endif // ESTIM_NAN

#if USE_MPI
	MPIMbSize = Fs.mpiTrans->AllocMemory(q, Fs.BlSz, Emax);
	MemMbSize += MPIMbSize;
	long double MPIMbSize = Fs.mpiTrans->AllocMemory(q, Fs.BlSz, Emax);
	MemMbSize += MPIMbSize;
	if (0 == Fs.mpiTrans->myRank)
#endif // end USE_MPI
	{
		std::cout << "Device Memory Usage: " << MemMbSize / 1024.0 << " GB\n";
#ifdef USE_MPI
		std::cout << "MPI trans Memory Size: " << MPIMbSize / 1024.0 << " GB\n";
#endif
	}
}

void Fluid::InitialU(sycl::queue &q)
{
	InitializeFluidStates(q, Fs.BlSz, Fs.ini, material_property, Fs.d_thermal, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH);
	GetCellCenterDerivative(q, Fs.BlSz, d_fstate, Fs.Boundarys);
}

void Fluid::GetTheta(sycl::queue &q)
{
	Block bl = Fs.BlSz;
	real_t *yi = d_fstate.y;
	real_t *rho = d_fstate.rho;
	real_t *vox_2 = d_fstate.vx;
	real_t *smyXe = d_fstate.thetaXe; // get (Y_Xe)^bar of each rank,Ref.https://doi.org/10.1017/S0263034600008557.P_729
	real_t *smyN2 = d_fstate.thetaN2; // get (Y_N2)^bar of each rank
	real_t *smyXN = d_fstate.thetaXN; // get (Y_Xe*Y_N2)^bar of each rank

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<2>(sycl::range<2>(bl.X_inner, bl.Z_inner), sycl::range<2>(bl.dim_block_x, bl.dim_block_z)), [=](nd_item<2> index)
				   {	
				int i = index.get_global_id(0) + bl.Bwidth_X;
				int k = index.get_global_id(1) + bl.Bwidth_Z;
				YDirThetaItegralKernel(i, k, bl, yi, smyXe, smyN2, smyXN); }); })
		.wait();

	// #ifdef USE_MPI
	// 	real_t temp = _DF(1.0) / (bl.my * bl.Y_inner);
	// 	int *root_y = new int[bl.mx * bl.mz], size = bl.X_inner *bl.Z_inner;
	// 	for (size_t pos_x = 0; pos_x < bl.mx; pos_x++)
	// 		for (size_t pos_z = 0; pos_z < bl.mz; pos_z++)
	// 		{
	// 			MPI_Group groupy;
	// 			root_y[pos_x * bl.mz + pos_z] = Fs.mpiTrans->Get_RankGroupXZ(groupy, pos_x, pos_z); // only sum ranks at YDIR
	// 			real_t *tempYXe = new real_t[size], *tempYN2 = new real_t[size], *tempYXN = new real_t[size];
	// 			for (size_t jj = 0; jj < bl.Z_inner; jj++)
	// 				for (size_t ii = 0; ii < bl.X_inner; ii++)
	// 				{
	// 					Fs.mpiTrans->GroupallReduce(&(smyXe[bl.X_inner * jj + ii]), &(tempYXe[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 					Fs.mpiTrans->GroupallReduce(&(smyN2[bl.X_inner * jj + ii]), &(tempYN2[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 					Fs.mpiTrans->GroupallReduce(&(smyXN[bl.X_inner * jj + ii]), &(tempYXN[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 				}
	// 			Fs.mpiTrans->communicator->synchronize();
	// 			if (root_y[pos_x * bl.mz + pos_z] == Fs.mpiTrans->myRank)
	// 				for (size_t jj = 0; jj < bl.Z_inner; jj++)
	// 					for (size_t ii = 0; ii < bl.X_inner; ii++)
	// 					{
	// 						smyXe[bl.X_inner * jj + ii] = temp * tempYXe[bl.X_inner * jj + ii]; //(Y_Xe)^bar=SUM(Y_Xe)/(bl.my*bl.Y_inner)
	// 						smyN2[bl.X_inner * jj + ii] = temp * tempYN2[bl.X_inner * jj + ii]; //(Y_N2)^bar=SUM(Y_N2)/(bl.my*bl.Y_inner)
	// 						smyXN[bl.X_inner * jj + ii] = temp * tempYXN[bl.X_inner * jj + ii]; //(Y_Xe*Y_N2)^bar=SUM(Y_Xe*Y_N2)/(bl.my*bl.Y_inner)
	// 					}
	// 		}
	// #endif // end USE_MPI

	for (size_t i = 0; i < 3; i++)
		theta[i] = _DF(0.0);
	auto Sum_YXN = sycl::reduction(&(theta[0]), sycl::plus<>());   // (Y_Xe*Y_N2)^bar
	auto Sum_YXeN2 = sycl::reduction(&(theta[1]), sycl::plus<>()); // (Y_Xe)^bar*(Y_N2)^bar
	auto Sum_YXe = sycl::reduction(&(theta[2]), sycl::plus<>());   // (Y_Xe)^bar*(Y_N2)^bar
	real_t _RomY = _DF(1.0) / real_t(bl.Y_inner);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<1>(sycl::range<1>(bl.X_inner * bl.Z_inner), sycl::range<1>(bl.BlockSize)), Sum_YXN, Sum_YXeN2, Sum_YXe, [=](nd_item<1> index, auto &tSum_YXN, auto &tSum_YXeN2, auto &tSum_YXe)
				   { auto id = index.get_global_id(0);
			tSum_YXN += smyXN[id];
			tSum_YXeN2 += smyXe[id]  * smyN2[id];
			tSum_YXe += smyXe[id]; }); })
		.wait();

	auto local_ndrange3d = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange3d = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	for (size_t i = 0; i < 6; i++)
		interface_point[i] = _DF(0.0);

	// #if DIM_X // XDIR
	// 	auto Rdif_Xmin = reduction(&(interface_point[0]), sycl::minimum<>());
	// 	auto Rdif_Xmax = reduction(&(interface_point[1]), sycl::maximum<>());
	// 	q.submit([&](sycl::handler &h)
	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Rdif_Xmin, Rdif_Xmax, [=](nd_item<3> index, auto &temp_Xmin, auto &temp_Xmax)
	// 							  {
	// 					int i = index.get_global_id(0) + bl.Bwidth_X;
	// 					int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 					int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 					real_t x = i * bl.dx + bl.offx;
	// 					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
	// 					if (yi[id * NUM_SPECIES - 2] > Interface_line)
	// 						temp_Xmin.combine(x), temp_Xmax.combine(x); }); });
	// #endif	  // end DIM_X

	auto Rdif_Ymin = reduction(&(interface_point[2]), sycl::minimum<>());
	auto Rdif_Ymax = reduction(&(interface_point[3]), sycl::maximum<>());
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Rdif_Ymin, Rdif_Ymax, [=](nd_item<3> index, auto &temp_Ymin, auto &temp_Ymax)
							  {	
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					real_t y = j * bl.dy + bl.offy;
					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
					if (yi[id * NUM_SPECIES - 2] > Interface_line)
						temp_Ymin.combine(y), temp_Ymax.combine(y); }); });

	// #if DIM_Z // ZDIR
	// 	auto Rdif_Zmin = reduction(&(interface_point[4]), sycl::minimum<>());
	// 	auto Rdif_Zmax = reduction(&(interface_point[5]), sycl::maximum<>());
	// 	q.submit([&](sycl::handler &h)
	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Rdif_Zmin, Rdif_Zmax, [=](nd_item<3> index, auto &temp_Zmin, auto &temp_Zmax)
	// 							  {
	// 					int i = index.get_global_id(0) + bl.Bwidth_X;
	// 					int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 					int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 					real_t z = k * bl.dz + bl.offz;
	// 					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
	// 					if (yi[id * NUM_SPECIES - 2] > Interface_line)
	// 						temp_Zmin.combine(z), temp_Zmax.combine(z); }); });
	// #endif // end DIM_Z

#ifdef COP_CHEME
	int meshSize = bl.Xmax * bl.Ymax * bl.Zmax;
	auto local_ndrange = range<1>(bl.BlockSize); // size of workgroup
	auto global_ndrange = range<1>(meshSize);
	for (size_t n = 0; n < NUM_SPECIES - 3; n++)
		pVar_max[n] = _DF(0.0);
	// Tmax
	real_t *T = d_fstate.T;
	q.submit([&](sycl::handler &h)
			 {	auto reduction_max_T = reduction(&(pVar_max[0]), sycl::maximum<>());
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_T, [=](nd_item<1> index, auto &temp_max_T){
						   auto id = index.get_global_id();
						   temp_max_T.combine(T[id]);}); });
	//	reactants
	for (size_t n = 1; n < NUM_SPECIES - 3; n++)
	{
		q.submit([&](sycl::handler &h)
				 {	auto reduction_max_Yi = reduction(&(pVar_max[n]), sycl::maximum<>());
			h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_Yi, [=](nd_item<1> index, auto &temp_max_Yi){
							   auto id = index.get_global_id();
							   temp_max_Yi.combine(yi[n + 1 + NUM_SPECIES * id]); }); });
	}
#endif // end COP_CHEME

	sigma[0] = _DF(0.0), sigma[1] = _DF(0.0);
	auto Sum_Sigma = sycl::reduction(&(sigma[0]), sycl::plus<>());
	auto Sum_Sigma1 = sycl::reduction(&(sigma[1]), sycl::plus<>());
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Sum_Sigma, Sum_Sigma1, [=](nd_item<3> index, auto &temp_Sum_Sigma, auto &temp_Sum_Sigma1)
				   {	
				int i = index.get_global_id(0) + bl.Bwidth_X;
				int j = index.get_global_id(1) + bl.Bwidth_Y;
				int k = index.get_global_id(2) + bl.Bwidth_Z;
				int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
				temp_Sum_Sigma += vox_2[id] * bl.dx * bl.dy * bl.dz;
				temp_Sum_Sigma1 += rho[id] * vox_2[id] * bl.dx * bl.dy * bl.dz; }); });
	q.wait();

#ifdef USE_MPI
	real_t Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, Sumsigma, Sumsigma1, thetaYXN, thetaYXeN2, thetaYXe;
	Fs.mpiTrans->communicator->allReduce(&(interface_point[2]), &Ymin, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
	Fs.mpiTrans->communicator->allReduce(&(interface_point[3]), &Ymax, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	// Fs.mpiTrans->communicator->allReduce(&(interface_point[0]), &Xmin, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
	// Fs.mpiTrans->communicator->allReduce(&(interface_point[1]), &Xmax, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	// Fs.mpiTrans->communicator->allReduce(&(interface_point[4]), &Zmin, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
	// Fs.mpiTrans->communicator->allReduce(&(interface_point[5]), &Zmax, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->allReduce(&(sigma[0]), &Sumsigma, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM);
	Fs.mpiTrans->communicator->allReduce(&(sigma[1]), &Sumsigma1, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM);
	// MPI_Group groupry;
	// MPI_Group_incl(Fs.mpiTrans->comm_world, bl.mx * bl.mz, root_y, &groupry);
	// Fs.mpiTrans->GroupallReduce(&(theta[0]), &thetaYXN, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupry);
	// Fs.mpiTrans->GroupallReduce(&(theta[1]), &thetaYXeN2, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupry);
	Fs.mpiTrans->communicator->allReduce(&(theta[0]), &thetaYXN, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM);
	Fs.mpiTrans->communicator->allReduce(&(theta[1]), &thetaYXeN2, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM);
	Fs.mpiTrans->communicator->allReduce(&(theta[2]), &thetaYXe, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM);

#ifdef COP_CHEME
	real_t pVar[NUM_SPECIES - 3];
	for (size_t n = 0; n < NUM_SPECIES - 3; n++)
		Fs.mpiTrans->communicator->allReduce(&(pVar_max[n]), &pVar[n], 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->synchronize();
	for (size_t n = 0; n < NUM_SPECIES - 3; n++)
		pVar_max[n] = pVar[n];
#endif // end COP_CHEME

	// synchronize
	Fs.mpiTrans->communicator->synchronize();
	// give back to each rank
	interface_point[2] = Ymin, interface_point[3] = Ymax;
	// interface_point[0] = Xmin, interface_point[1] = Xmax, interface_point[4] = Zmin, interface_point[5] = Zmax;
	sigma[0] = Sumsigma, sigma[1] = Sumsigma1, theta[0] = thetaYXN, theta[1] = thetaYXeN2, theta[2] = thetaYXe;
#endif // end USE_MPI
}

real_t Fluid::GetFluidDt(sycl::queue &q, const int Iter, const real_t physicalTime)
{
	real_t dt_ref = GetDt(q, Fs.BlSz, Fs.d_thermal, d_fstate, uvw_c_max);
#ifdef USE_MPI
	real_t lambda_x0, lambda_y0, lambda_z0, miu_max, rho_min;
	Fs.mpiTrans->communicator->allReduce(&(uvw_c_max[0]), &lambda_x0, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->allReduce(&(uvw_c_max[1]), &lambda_y0, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->allReduce(&(uvw_c_max[2]), &lambda_z0, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->allReduce(&(uvw_c_max[3]), &miu_max, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	Fs.mpiTrans->communicator->allReduce(&(uvw_c_max[4]), &rho_min, 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
	Fs.mpiTrans->communicator->synchronize();
	uvw_c_max[0] = lambda_x0, uvw_c_max[1] = lambda_y0, uvw_c_max[2] = lambda_z0, uvw_c_max[3] = miu_max, uvw_c_max[4] = rho_min;

	dt_ref = uvw_c_max[0] * Fs.BlSz._dx + uvw_c_max[1] * Fs.BlSz._dy + uvw_c_max[2] * Fs.BlSz._dz;
	real_t temp_vis = _DF(14.0 / 3.0) * miu_max / rho_min * uvw_c_max[5];
	dt_ref = sycl::max<real_t>(dt_ref, temp_vis);
	dt_ref = Fs.BlSz.CFLnumber / dt_ref;
#endif // end USE_MPI

#ifdef SBICounts
	// bool push = Iter % Fs.POutInterval == 0 ? true : false;
	if (Iter % Fs.POutInterval == 0) // append once to the counts file avoiding
	{
		GetTheta(q);

		// Output
		if (Fs.myRank == 0)
		{
			std::ofstream out;
			out.open(file_name, std::ios::out | std::ios::app);
			out.setf(std::ios::right);

			out << std::setw(11) << physicalTime << " ";											   // physical time
			/**Theta(XN/(Xe*N2))
			 * Ref: https://linkinghub.elsevier.com/retrieve/pii/S0010218015003648.eq.(35)
			 */
			out << std::setw(11) << theta[0] / theta[1] * real_t(Fs.BlSz.Y_inner * Fs.BlSz.my) << " ";
			/**Lambday diameter of the bubble
			 * Ref: https://linkinghub.elsevier.com/retrieve/pii/S0010218015003648.
			 */
			real_t offsety = (Fs.Boundarys[2] == 2 && Fs.ini.cop_center_y <= 1.0e-10) ? _DF(1.0) : _DF(0.5);
			out << std::setw(7) << (interface_point[3] - interface_point[2]) * offsety / Fs.ini.yb << " ";
			/**sigma[0]: sum(Omega^2) sigma[1]: sum(rho*Omega^2)/rho_0
			 * (no rho0 exact definition found) in Ref
			 * there three initial rho in the Domain: pre-shcok rho, post-shock rho(used, higher than the pre one), bubble rho
			 * Ref: https://linkinghub.elsevier.com/retrieve/pii/S0010218015003648.eq.(34)
			 */
			real_t rho0 = Fs.ini.blast_density_in;
			out << std::setw(11) << sigma[0] << " " << std::setw(11) << sigma[1] / rho0 << " ";
#ifdef COP_CHEME
			out << std::setw(7) << pVar_max[0] << " "; // Tmax
			for (size_t n = 1; n < NUM_SPECIES - 3; n++)
				out << std::setw(11) << pVar_max[n] << " ";
#endif														  // end COP_CHEME
			out << std::setw(7) << ++SBIOutIter << " ";		  // Step
			out << std::setw(11) << theta[0] << " ";		  // [0]XN
			out << std::setw(11) << theta[1] << " ";		  // [1]Xe*N2
			out << std::setw(11) << theta[2] << " ";		  // [2]Xe
			out << std::setw(8) << interface_point[2] << " "; // Ymin
			out << std::setw(8) << interface_point[3] << " "; // Ymax
			// #if DIM_X
			// 		out << std::setw(8) << interface_point[0] << " ";												 // Xmin
			// 		out << std::setw(8) << interface_point[1] << " ";												 // Xmax
			// 		out << std::setw(6) << (interface_point[1] - interface_point[0]) * _DF(0.5) / Fs.ini.xa << " ";	 // Lambdax
			// #endif
			// #if DIM_Z
			// 		out << std::setw(8) << interface_point[4] << " "; // Zmin
			// 		out << std::setw(8) << interface_point[5] << " "; // Zmax
			// 		real_t offsetz = (Fs.Boundarys[4] == 2 && Fs.ini.cop_center_z <= 1.0e-10) ? _DF(1.0) : _DF(0.5);
			// 		out << std::setw(7) << (interface_point[5] - interface_point[4]) * offsetz / Fs.ini.zc << " "; // Lambdaz
			// #endif
			out << "\n";
			out.close();
		}
	}
#endif // end SBICounts

	return dt_ref;
}

void Fluid::BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag)
{
	std::chrono::high_resolution_clock::time_point start_time_x = std::chrono::high_resolution_clock::now();

	if (flag == 0)
		MPI_trans_time += FluidBoundaryCondition(q, Fs, BCs, d_U);
	else
		MPI_trans_time += FluidBoundaryCondition(q, Fs, BCs, d_U1);

	std::chrono::high_resolution_clock::time_point end_time_x = std::chrono::high_resolution_clock::now();
	MPI_BCs_time += std::chrono::duration<float, std::milli>(end_time_x - start_time_x).count() * 1.0e-3f;
}

bool Fluid::UpdateFluidStates(sycl::queue &q, int flag)
{
	real_t *UI = NULL;
	if (flag == 0)
		UI = d_U;
	else
		UI = d_U1;

	if (UpdateFluidStateFlux(q, Fs.BlSz, Fs.d_thermal, UI, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma, error_patched_times))
		return true;

	return false;
}

void Fluid::UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt)
{
	UpdateURK3rd(q, Fs.BlSz, d_U, d_U1, d_LU, dt, flag);
}

void Fluid::ComputeFluidLU(sycl::queue &q, int flag)
{
	// // SYCL kernel cannot call through a function pointer
	// void (*RoeAverageLeftX[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageLeftY[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageLeftZ[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightX[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightY[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightZ[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// RoeAverageLeftX[0] = RoeAverageLeft_x_0;
	// RoeAverageLeftX[1] = RoeAverageLeft_x_1;
	// RoeAverageLeftX[2] = RoeAverageLeft_x_2;
	// RoeAverageLeftX[3] = RoeAverageLeft_x_3;
	// RoeAverageLeftX[Emax - 1] = RoeAverageLeft_x_end;

	// RoeAverageLeftY[0] = RoeAverageLeft_y_0;
	// RoeAverageLeftY[1] = RoeAverageLeft_y_1;
	// RoeAverageLeftY[2] = RoeAverageLeft_y_2;
	// RoeAverageLeftY[3] = RoeAverageLeft_y_3;
	// RoeAverageLeftY[Emax - 1] = RoeAverageLeft_y_end;

	// RoeAverageLeftZ[0] = RoeAverageLeft_z_0;
	// RoeAverageLeftZ[1] = RoeAverageLeft_z_1;
	// RoeAverageLeftZ[2] = RoeAverageLeft_z_2;
	// RoeAverageLeftZ[3] = RoeAverageLeft_z_3;
	// RoeAverageLeftZ[Emax - 1] = RoeAverageLeft_z_end;

	// RoeAverageRightX[0] = RoeAverageRight_x_0;
	// RoeAverageRightX[1] = RoeAverageRight_x_1;
	// RoeAverageRightX[2] = RoeAverageRight_x_2;
	// RoeAverageRightX[3] = RoeAverageRight_x_3;
	// RoeAverageRightX[Emax - 1] = RoeAverageRight_x_end;

	// RoeAverageRightY[0] = RoeAverageRight_y_0;
	// RoeAverageRightY[1] = RoeAverageRight_y_1;
	// RoeAverageRightY[2] = RoeAverageRight_y_2;
	// RoeAverageRightY[3] = RoeAverageRight_y_3;
	// RoeAverageRightY[Emax - 1] = RoeAverageRight_y_end;

	// RoeAverageRightZ[0] = RoeAverageRight_z_0;
	// RoeAverageRightZ[1] = RoeAverageRight_z_1;
	// RoeAverageRightZ[2] = RoeAverageRight_z_2;
	// RoeAverageRightZ[3] = RoeAverageRight_z_3;
	// RoeAverageRightZ[Emax - 1] = RoeAverageRight_z_end;

	// #ifdef COP
	// 	for (size_t i = 4; i < Emax - 1; i++)
	// 	{
	// 		RoeAverageLeftX[i] = RoeAverageLeft_x_cop;
	// 		RoeAverageLeftY[i] = RoeAverageLeft_y_cop;
	// 		RoeAverageLeftZ[i] = RoeAverageLeft_z_cop;
	// 		RoeAverageRightX[i] = RoeAverageRight_x_cop;
	// 		RoeAverageRightY[i] = RoeAverageRight_y_cop;
	// 		RoeAverageRightZ[i] = RoeAverageRight_z_cop;
	// 	}
	// #endif // end COP

	if (flag == 0)
		GetLU(q, Fs, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local_x, d_eigen_local_y, d_eigen_local_z,
			  d_eigen_l, d_eigen_r, uvw_c_max, eigen_block_x, eigen_block_y, eigen_block_z);
	else
		GetLU(q, Fs, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local_x, d_eigen_local_y, d_eigen_local_z,
			  d_eigen_l, d_eigen_r, uvw_c_max, eigen_block_x, eigen_block_y, eigen_block_z);
}

bool Fluid::EstimateFluidNAN(sycl::queue &q, int flag)
{
	Block bl = Fs.BlSz;
	real_t *UI, *LU = d_LU;
	switch (flag)
	{
	case 1:
		UI = d_U1;
		break;

	case 2:
		UI = d_U1;
		break;

	case 3:
		UI = d_U;
		break;

	case 4: // for Reaction
		UI = d_U;
		break;
	}

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	int x_offset = Fs.OutBoundary ? 0 : bl.Bwidth_X;
	int y_offset = Fs.OutBoundary ? 0 : bl.Bwidth_Y;
	int z_offset = Fs.OutBoundary ? 0 : bl.Bwidth_Z;

	bool *error;
	int *error_pos;
	error = middle::MallocShared<bool>(error, 1, q);
	error_pos = middle::MallocShared<int>(error_pos, Emax + 3, q);
	for (size_t n = 0; n < Emax + 3; n++)
		error_pos[n] = 0;
	*error = false;

	q.submit([&](sycl::handler &h) { // sycl::stream error_out(64 * 1024, 10, h);
		 h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
						{
    		int i = index.get_global_id(0) + x_offset;
			int j = index.get_global_id(1) + y_offset;
			int k = index.get_global_id(2) + z_offset;
			EstimateFluidNANKernel(i, j, k, x_offset, y_offset, z_offset, bl, error_pos, UI, LU, error); });
	 })
		.wait(); //, error_out

	if (*error)
	{
		std::cout << "Errors of LU[";
		for (size_t ii = 0; ii < Emax - 1; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[Emax - 1] << "] located at (i, j, k)= (";
		std::cout << error_pos[Emax + 1] - x_offset << ", " << error_pos[Emax + 2] - y_offset << ", " << error_pos[Emax + 3] - z_offset;
		if (flag <= 3)
			std::cout << ") inside the step " << flag << " of RungeKutta";
		else
			std::cout << ") after the Reaction solver";
		std::cout << " captured.\n";
		return true;
	}

	// free
	{
		middle::Free(error, q);
		middle::Free(error_pos, q);
	}

	return false;
}

void Fluid::ZeroDimensionalFreelyFlame()
{
#ifdef ODESolverTest
	std::cout << "0D H2-O2 freely flame testing";
	ZeroDimensionalFreelyFlameBlock(Fs);
	std::cout << " done.\n";
#endif // end ODESolverTest
}

void Fluid::ODESolver(sycl::queue &q, real_t Time)
{
#ifdef COP_CHEME
#if 0 == CHEME_SOLVER
	ChemeODEQ2Solver(q, Fs.BlSz, Fs.d_thermal, d_fstate, d_U, Fs.d_react, Time);
//#else 1 == CHEME_SOLVER // CVODE from LLNL to SYCL only support Intel GPUs
#endif // end CHEME_SOLVER
#endif // end COP_CHEME
}
