#include "global_class.h"

#include "sycl_kernels.hpp"
#include "solver_Ini/Ini_block.h"
#include "solver_BCs/BCs_block.h"
#include "solver_GetDt/GlobalDt_block.hpp"
#include "solver_Reaction/Reaction_block.hpp"
#include "solver_Reconstruction/Reconstruction_block.hpp"

extern void YDirThetaItegralKernel(int i, int k, Block bl, real_t *y, real_t *ThetaXe, real_t *ThetaN2, real_t *ThetaXN)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	int ii = i - Bwidth_X, kk = k - Bwidth_Z;
	ThetaXe[X_inner * kk + ii] = _DF(0.0), ThetaN2[X_inner * kk + ii] = _DF(0.0), ThetaXN[X_inner * kk + ii] = _DF(0.0);
	for (size_t j = bl.Bwidth_Y; j < bl.Ymax - bl.Bwidth_Y; j++)
	{
		int id = Xmax * Ymax * k + Xmax * j + i;
		real_t *yi = &(y[NUM_SPECIES * id]);
		ThetaXe[X_inner * kk + ii] += yi[bl.Xe_id];
		ThetaN2[X_inner * kk + ii] += yi[bl.N2_id];
		ThetaXN[X_inner * kk + ii] += yi[bl.Xe_id] * yi[bl.N2_id];
	}
}

// extern void XDirThetaItegralKernel(int k, Block bl, real_t *ThetaXeIn, real_t *ThetaN2In, real_t *ThetaXNIn,
//                                                  real_t *ThetaXeOut, real_t *ThetaN2Out, real_t *ThetaXNOut)
// {
//     MARCO_DOMAIN_GHOST();
//     if (k >= Z_inner + Bwidth_Z)
//         return;

//     ThetaXeOut[k] = _DF(0.0), ThetaN2Out[k] = _DF(0.0), ThetaXNOut[k] = _DF(0.0);
//     for (size_t i = bl.Bwidth_Y; i < bl.Ymax; i++)
//     {
//         int id = Xmax * Ymax * k + Xmax * j + i + 1;
//         ThetaXe[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 2];
//         ThetaN2[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 1];
//         ThetaXN[Xmax * k + i] += bl.dx * yi[NUM_SPECIES * id - 2] * yi[NUM_SPECIES * id - 1];
//     }
// }

extern void EstimateFluidNANKernel(int i, int j, int k, int x_offset, int y_offset, int z_offset, Block bl, int *error_pos, real_t *UI, real_t *LUI, bool *error) //, sycl::stream stream_ct1
{
	int Xmax = bl.Xmax;
	int Ymax = bl.Ymax;
	if (i >= Xmax - bl.Bwidth_X)
		return;
	if (j >= Ymax - bl.Bwidth_Y)
		return;
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;
	int id = (Xmax * Ymax * k + Xmax * j + i) * Emax;

	bool tempnegv = UI[0 + id] < 0 ? true : false;
	if (tempnegv)
		error_pos[0] += 10000;
	for (size_t ii = 0; ii < Emax; ii++)
	{
		bool thtemp = false;
		thtemp = sycl::isnan(UI[ii + id]);
		if (thtemp)
			error_pos[ii] += 1100;
		tempnegv = tempnegv || thtemp;

		thtemp = sycl::isinf(UI[ii + id]);
		if (thtemp)
			error_pos[ii] += 1200;
		tempnegv = tempnegv || thtemp;

		thtemp = sycl::isnan(LUI[ii + id]);
		if (thtemp)
			error_pos[ii] += 21;
		tempnegv = tempnegv || thtemp;

		thtemp = sycl::isinf(LUI[ii + id]);
		if (thtemp)
			error_pos[ii] += 22;
		tempnegv = tempnegv || thtemp;
	}
	if (tempnegv)
		*error = true, error_pos[Emax + 1] = i, error_pos[Emax + 2] = j, error_pos[Emax + 3] = k;
}

Fluid::Fluid(Setup &setup) : Fs(setup), q(setup.q), rank(0), nranks(1), SBIOutIter(0)
{
	MPI_BCs_time = 0.0;
	MPI_trans_time = 0.0;
	error_patched_times = 0;
#ifdef USE_MPI
	rank = Fs.mpiTrans->myRank;
	nranks = Fs.mpiTrans->nProcs;
#endif

	if (ReactSources && ODETest_json)
		ZeroDimensionalFreelyFlame();

	// Creat Counts file
	AllCountsHeader();
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
	sycl::free(d_fstate.hi, q);
	sycl::free(d_fstate.Cp, q);
	sycl::free(d_fstate.Ri, q);

	if (Visc)
	{ // free viscous Vars
		sycl::free(d_fstate.viscosity_aver, q);
		sycl::free(d_fstate.vx, q), sycl::free(h_fstate.vx, q);
		for (size_t i = 0; i < 3; i++)
			sycl::free(d_fstate.vxs[i], q), sycl::free(h_fstate.vxs[i], q);
		for (size_t i = 0; i < 9; i++)
			sycl::free(d_fstate.Vde[i], q);

		if (Visc_Heat)
			sycl::free(d_fstate.thermal_conduct_aver, q);
		if (Visc_Diffu)
		{
			sycl::free(yi_min, q), sycl::free(yi_max, q);
			sycl::free(Dim_min, q), sycl::free(Dim_max, q);
			sycl::free(d_fstate.Dkm_aver, q);
		}
	}

	sycl::free(uvw_c_max, q);
	sycl::free(d_FluxF, q), sycl::free(d_FluxG, q), sycl::free(d_FluxH, q);
	sycl::free(d_wallFluxF, q), sycl::free(d_wallFluxG, q), sycl::free(d_wallFluxH, q);

	if (OutOverTime)
	{
		sycl::free(d_fstate.thetaXe, q);
		sycl::free(d_fstate.thetaN2, q);
		sycl::free(d_fstate.thetaXN, q);
		sycl::free(theta, q), sycl::free(sigma, q);
		sycl::free(pVar_max, q), sycl::free(interface_point, q);
	}
#if ESTIM_OUT
	sycl::free(h_U, q), sycl::free(h_U1, q), sycl::free(h_LU, q);
	if (Fs.BlSz.DimX)
	{
		sycl::free(h_fstate.b1x, q), sycl::free(h_fstate.b3x, q);
		sycl::free(h_fstate.c2x, q), sycl::free(h_fstate.zix, q);
		sycl::free(d_fstate.b1x, q), sycl::free(d_fstate.b3x, q);
		sycl::free(d_fstate.c2x, q), sycl::free(d_fstate.zix, q);
		sycl::free(h_fstate.preFwx, q), sycl::free(h_fstate.pstFwx, q);
		sycl::free(d_fstate.preFwx, q);
	}
	if (Fs.BlSz.DimY)
	{
		sycl::free(h_fstate.b1y, q), sycl::free(h_fstate.b3y, q);
		sycl::free(h_fstate.c2y, q), sycl::free(h_fstate.ziy, q);
		sycl::free(d_fstate.b1y, q), sycl::free(d_fstate.b3y, q);
		sycl::free(d_fstate.c2y, q), sycl::free(d_fstate.ziy, q);
		sycl::free(h_fstate.preFwy, q), sycl::free(h_fstate.pstFwy, q);
		sycl::free(d_fstate.preFwy, q);
	}
	if (Fs.BlSz.DimZ)
	{
		sycl::free(h_fstate.b1z, q), sycl::free(h_fstate.b3z, q);
		sycl::free(h_fstate.c2z, q), sycl::free(h_fstate.ziz, q);
		sycl::free(d_fstate.b1z, q), sycl::free(d_fstate.b3z, q);
		sycl::free(d_fstate.c2z, q), sycl::free(d_fstate.ziz, q);
		sycl::free(h_fstate.preFwz, q), sycl::free(h_fstate.pstFwz, q);
		sycl::free(d_fstate.preFwz, q);
	}

	if (Visc)
	{ // free viscous estimate Vars
		if (Fs.BlSz.DimX)
		{
			sycl::free(h_fstate.visFwx, q);
			sycl::free(d_fstate.visFwx, q);
		}
		if (Fs.BlSz.DimY)
		{
			sycl::free(h_fstate.visFwy, q);
			sycl::free(d_fstate.visFwy, q);
		}
		if (Fs.BlSz.DimZ)
		{
			sycl::free(h_fstate.visFwz, q);
			sycl::free(d_fstate.visFwz, q);
		}

		if (Visc_Diffu)
		{
			sycl::free(h_fstate.Ertemp1, q);
			sycl::free(h_fstate.Ertemp2, q);
			sycl::free(h_fstate.Dkm_aver, q);
			sycl::free(d_fstate.Ertemp1, q);
			sycl::free(d_fstate.Ertemp2, q);

			if (Fs.BlSz.DimX)
			{
				sycl::free(h_fstate.Dim_wallx, q), sycl::free(h_fstate.hi_wallx, q);
				sycl::free(h_fstate.Yi_wallx, q), sycl::free(h_fstate.Yil_wallx, q);
				sycl::free(d_fstate.Dim_wallx, q), sycl::free(d_fstate.hi_wallx, q);
				sycl::free(d_fstate.Yi_wallx, q), sycl::free(d_fstate.Yil_wallx, q);
			}
			if (Fs.BlSz.DimY)
			{
				sycl::free(h_fstate.Dim_wally, q), sycl::free(h_fstate.hi_wally, q);
				sycl::free(h_fstate.Yi_wally, q), sycl::free(h_fstate.Yil_wally, q);
				sycl::free(d_fstate.Dim_wally, q), sycl::free(d_fstate.hi_wally, q);
				sycl::free(d_fstate.Yi_wally, q), sycl::free(d_fstate.Yil_wally, q);
			}
			if (Fs.BlSz.DimZ)
			{
				sycl::free(h_fstate.Dim_wallz, q), sycl::free(h_fstate.hi_wallz, q);
				sycl::free(h_fstate.Yi_wallz, q), sycl::free(h_fstate.Yil_wallz, q);
				sycl::free(d_fstate.Dim_wallz, q), sycl::free(d_fstate.hi_wallz, q);
				sycl::free(d_fstate.Yi_wallz, q), sycl::free(d_fstate.Yil_wallz, q);
			}
		}
	}
#endif // ESTIM_OUT
}

void Fluid::initialize(int n)
{
	Fluid_name = Fluids_name[n]; // give a name to the fluid
	// type of material, 0: gamma gas, 1: water, 2: stiff gas
	material_property.Mtrl_ind = Fs.material_props[n][0];
	// fluid indicator and EOS Parameters
	material_property.Rgn_ind = Fs.material_props[n][1];
	// gamma, A, B, rho0, R_0, lambda_0, sound_speed
	material_property.Gamma = Fs.material_props[n][2];
	material_property.A = Fs.material_props[n][3];
	material_property.B = Fs.material_props[n][4];
	material_property.rho0 = Fs.material_props[n][5];
	material_property.R_0 = Fs.material_props[n][6];
	material_property.lambda_0 = Fs.material_props[n][7];
}

void Fluid::AllocateFluidMemory(sycl::queue &q)
{
	double this_msize = _DF(0.0);
	int bytes = Fs.bytes, cellbytes = Fs.cellbytes;
	int Kbytes = (Fs.bytes >> 10), Kcellbytes = (Fs.cellbytes >> 10);
	// 主机内存
	{
		h_U = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
		h_U1 = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
		h_LU = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
		h_fstate.rho = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.p = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.c = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.H = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.u = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.v = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.w = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.T = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.e = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.gamma = static_cast<real_t *>(sycl::malloc_host(bytes, q));
		h_fstate.y = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_SPECIES, q));
		MemMbSize = double(Kbytes >> 10) * (10.0 + NUM_SPECIES + 3.0 * Emax);
		if (0 == rank)
			std::cout << "Host memory malloced(primitive variables): " << MemMbSize << " MB/" << MemMbSize / 1024.0 << " GB" << std::endl;
	}
	// 设备内存
	{
		d_U = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_U1 = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_LU = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		Ubak = static_cast<real_t *>(sycl::malloc_shared(cellbytes, q)); // shared memory may inside device
		MemMbSize = double(Kcellbytes >> 10) * 4.0;
		if (0 == rank)
			std::cout << "Device memory malloced(conservative variables, checking_point file): " << MemMbSize << " MB, "
					  << MemMbSize / 1024.0 << " GB" << std::endl;

		d_eigen_local_x = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_eigen_local_y = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_eigen_local_z = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		this_msize = double(Kcellbytes >> 10) * 3.0, MemMbSize += this_msize;
		if (0 == rank)
			std::cout << "Device memory malloced(eigen values): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
					  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;

		d_fstate.rho = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.p = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.c = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.u = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.v = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.w = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.e = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.T = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.H = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.Ri = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.Cp = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.gamma = static_cast<real_t *>(sycl::malloc_device(bytes, q));
		d_fstate.y = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_SPECIES, q));
		d_fstate.hi = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_SPECIES, q));
		this_msize = double(Kbytes >> 10) * (12.0 + NUM_SPECIES * 2.0), MemMbSize += this_msize;
		if (0 == rank)
			std::cout << "Device memory malloced(primitive variables): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
					  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;
	}

	// // allocate mem viscous Vars
	if (Visc)
	{
		{ // // vrotex
			for (size_t i = 0; i < 9; i++)
				d_fstate.Vde[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
			h_fstate.vx = static_cast<real_t *>(sycl::malloc_host(bytes, q)); // vorticity.
			d_fstate.vx = static_cast<real_t *>(sycl::malloc_device(bytes, q));
			for (size_t i = 0; i < 3; i++) // vorticity_i
			{
				h_fstate.vxs[i] = static_cast<real_t *>(sycl::malloc_host(bytes, q));
				d_fstate.vxs[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
			}
			this_msize = double(Kbytes >> 10) * 13.0, MemMbSize += this_msize;
			if (0 == rank)
				std::cout << "Device memory malloced(velocity deflection, vorticity): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
						  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;
		}

		{ // // visosity transport coefficients
			// // kinematic viscosity
			d_fstate.viscosity_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
			if (Visc_Heat) // // fourier heat transfer
				d_fstate.thermal_conduct_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
			if (Visc_Diffu) // // mass diffusion
			{
				yi_min = static_cast<real_t *>(sycl::malloc_shared(NUM_SPECIES * sizeof(real_t), q));
				yi_max = static_cast<real_t *>(sycl::malloc_shared(NUM_SPECIES * sizeof(real_t), q));
				Dim_min = static_cast<real_t *>(sycl::malloc_shared(NUM_SPECIES * sizeof(real_t), q));
				Dim_max = static_cast<real_t *>(sycl::malloc_shared(NUM_SPECIES * sizeof(real_t), q));
				d_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			}
			this_msize = double(Kbytes >> 10) * (1.0 + real_t(Visc_Heat) + real_t(Visc_Diffu) * NUM_SPECIES), MemMbSize += this_msize;
			if (0 == rank)
				std::cout << "Device memory malloced(viscosity): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
						  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;
		}
	}

	{
		d_FluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_FluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_FluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_wallFluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_wallFluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		d_wallFluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		this_msize = double(Kcellbytes >> 10) * 6.0, MemMbSize += this_msize;
		if (0 == rank)
			std::cout << "Device memory malloced(Fluxes): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
					  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;
	}

	// shared memory
	uvw_c_max = static_cast<real_t *>(sycl::malloc_shared(6 * sizeof(real_t), q));
	eigen_block_x = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	eigen_block_y = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	eigen_block_z = static_cast<real_t *>(sycl::malloc_shared(Emax * sizeof(real_t), q));
	if (OutOverTime)
	{
		// calculate Tmax, YiH2O2max, YiHO2max
		pVar_max = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 1 - 4) * sizeof(real_t), q));
		// Ref:https://linkinghub.elsevier.com/retrieve/pii/S0010218015003648.eq(34)
		sigma = static_cast<real_t *>(sycl::malloc_shared(2 * sizeof(real_t), q));
		// Yi(Xe),Yi(N2),Yi(Xe*N2)// Ref.eq(35)
		theta = static_cast<real_t *>(sycl::malloc_shared(3 * sizeof(real_t), q));
		// bubble size for transverse bubble diameter
		interface_point = static_cast<real_t *>(sycl::malloc_shared(6 * sizeof(real_t), q));
		// molecular mixing fraction
		int Size = Fs.BlSz.X_inner * Fs.BlSz.Z_inner * sizeof(real_t);
		d_fstate.thetaXe = static_cast<real_t *>(sycl::malloc_shared(Size, q));
		d_fstate.thetaN2 = static_cast<real_t *>(sycl::malloc_shared(Size, q));
		d_fstate.thetaXN = static_cast<real_t *>(sycl::malloc_shared(Size, q));
		this_msize = double(Size >> 10) * 3.0 / 1024.0, MemMbSize += this_msize;
		if (0 == rank)
			std::cout << "Device memory malloced(over time count): " << this_msize << " MB/" << this_msize / 1024.0 << " GB, "
					  << "cumulative memory: " << MemMbSize << "MB, " << MemMbSize / 1024.0 << "GB." << std::endl;
	}

#if ESTIM_OUT
	if (Fs.BlSz.DimX)
	{
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
	}
	if (Fs.BlSz.DimY)
	{
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
	}
	if (Fs.BlSz.DimZ)
	{
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
	}
	MemMbSize += ((double(cellbytes) / 1024.0)) / 1024.0 * double((Fs.BlSz.DimX + Fs.BlSz.DimY + Fs.BlSz.DimZ));
	// MemMbSize += ((double(bytes) / 1024.0)) / 1024.0 * double((Fs.BlSz.DimX + Fs.BlSz.DimY + Fs.BlSz.DimZ) * (NUM_COP + 3));

	if (Visc) // allocate viscous estimating Vars
	{
		if (Fs.BlSz.DimX)
		{
			h_fstate.visFwx = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
			d_fstate.visFwx = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		}
		if (Fs.BlSz.DimY)
		{
			h_fstate.visFwy = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
			d_fstate.visFwy = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		}
		if (Fs.BlSz.DimZ)
		{
			h_fstate.visFwz = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
			d_fstate.visFwz = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
		}
		MemMbSize += ((double(cellbytes) / 1024.0)) / 1024.0 * double(Fs.BlSz.DimX + Fs.BlSz.DimY + Fs.BlSz.DimZ);

		if (Visc_Diffu)
		{
			h_fstate.Ertemp1 = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
			h_fstate.Ertemp2 = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
			h_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
			d_fstate.Ertemp1 = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			d_fstate.Ertemp2 = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			d_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			if (Fs.BlSz.DimX)
			{
				h_fstate.Dim_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.hi_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yi_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yil_wallx = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				d_fstate.Dim_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.hi_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yi_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yil_wallx = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			}
			if (Fs.BlSz.DimY)
			{
				h_fstate.Dim_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.hi_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yi_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yil_wally = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				d_fstate.Dim_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.hi_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yi_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yil_wally = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			}
			if (Fs.BlSz.DimZ)
			{
				h_fstate.Dim_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.hi_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yi_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				h_fstate.Yil_wallz = static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * bytes, q));
				d_fstate.Dim_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.hi_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yi_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
				d_fstate.Yil_wallz = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
			}
			MemMbSize += ((double(bytes) / 1024.0 / 1024.0) * (3.0 + 4.0 * double(Fs.BlSz.DimX + Fs.BlSz.DimY + Fs.BlSz.DimZ))) * double(NUM_SPECIES);
		}
	}
#endif // ESTIM_OUT

#if USE_MPI
	MPIMbSize = (Fs.mpiTrans->AllocMemory(q, Fs.BlSz, Emax));
	MemMbSize += MPIMbSize;
	if (0 == Fs.mpiTrans->myRank)
#endif // end USE_MPI
	{
#ifdef USE_MPI
		std::cout << "MPI trans Memory Size: " << MPIMbSize << " MB/" << MPIMbSize / 1024.0 << " GB.\n";
#endif
		std::cout << "Device Memory Total Usage: " << MemMbSize << " MB/" << MemMbSize / 1024.0 << " GB.\n";
	}
}

void Fluid::InitialU(sycl::queue &q)
{
	InitializeFluidStates(q, Fs.BlSz, Fs.ini, material_property, Fs.d_thermal, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH);
}

void Fluid::AllCountsHeader()
{
	if (OutOverTime)
	{
		file_name = OutputDir + "/AllCounts_" + outputPrefix + ".dat";
		if (Fs.myRank == 0)
		{
			std::fstream if_exist;
			if_exist.open(file_name, std::ios::in);
			if (!if_exist)
			{
				std::ofstream out(file_name, std::fstream::out);
				out.setf(std::ios::right);
				// // defining header for tecplot(plot software)
				out << "title='Time_NormalizedTime_Theta_Lambda_Sigma_NSigma";
				// Sigma: Sum(Omega^2), NSigma(Nromalized Sigma): Sum(rho[id]*Omega^2)
				if (ReactSources)
				{
					out << "_T";
					for (size_t n = 1; n < NUM_SPECIES - 3; n++)
						out << "_Yi" << Fs.species_name[n + 1];
				}
				out << "_Step_teXN_teXeN2_teXe_Ymin_Ymax";
				// 			out << "_Xmin_Xmax_Lambdax";
				// 			out << "_Zmin_Zmax_Lambdaz";
				out << "'\nvariables=Time[s], t<sup>*</sup>[-], <b><greek>Q</greek></b>[-], <greek>L</greek><sub>y</sub>[-], ";
				out << "<greek>e</greek><sub><greek>r</greek></sub>[m<sup>2</sup>/s<sup>2</sup>], ";
				out << "<greek>e</greek><sub><greek>r</greek>n</sub>[m<sup>2</sup>/s<sup>2</sup>], ";
				// Time[s]: physical Time as a form of tecplot variable
				// t<sup>*</sup>[-]: normalized Time as a form of tecplot variable
				//<greek>Q</greek>[-]: (theta(Theta(XN)/Theta(Xe)/Theta(N2))) in tecplot
				//<greek>L</greek><sub>y</sub>[-]: Lambday in tecplot
				//<greek>e</greek><sub><greek>r</greek></sub>: sigma in tecplot
				//<sub>max</sub>: T_max in tecplot sub{max} added
				//<i>Y(HO2)</i><sub>max</sub>[-]: Yi(HO2)_max

				if (ReactSources)
				{
					out << "<i>T</i><sub>max</sub>[K], ";
					for (size_t n = 1; n < NUM_SPECIES - 3; n++)
						out << "<i>Y(" << Fs.species_name[n + 1] << ")</i><sub>max</sub>[-], ";
				}
				out << "Step, Theta(XN), Theta(Xe*N2), Theta(Xe), ";
				out << "Ymin, Ymax";
				// 			out << ", Xmin, Xmax, <greek>L</greek><sub>x</sub>[-]";
				// 			out << ", Zmin, Zmax, <greek>L</greek><sub>z</sub>[-]";
				out << "\nzone t='" << outputPrefix << "'\n";
				out.close();
			}
		}
	}
}

void Fluid::AllCountsPush(sycl::queue &q, const size_t Iter, const real_t Time)
{
	if ((Iter % POutInterval == 0) && (OutOverTime)) // append once to the counts file avoiding
	{
		GetTheta(q);

		// Output
		if (Fs.myRank == 0)
		{
			std::ofstream out;
			out.open(file_name, std::ios::out | std::ios::app);
			out.setf(std::ios::right);

			out << std::setw(11) << Time << " " << std::setw(11) << Time / Fs.ini.tau_H << " "; // physical time
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

			if (ReactSources)
			{
				out << std::setw(7) << pVar_max[0] << " "; // Tmax
				for (size_t n = 1; n < NUM_SPECIES - 3; n++)
					out << std::setw(11) << pVar_max[n] << " ";
			}
			out << std::setw(7) << ++SBIOutIter << " ";		  // Step
			out << std::setw(11) << theta[0] << " ";		  // [0]XN
			out << std::setw(11) << theta[1] << " ";		  // [1]Xe*N2
			out << std::setw(11) << theta[2] << " ";		  // [2]Xe
			out << std::setw(8) << interface_point[2] << " "; // Ymin
			out << std::setw(8) << interface_point[3] << " "; // Ymax
			// 		out << std::setw(8) << interface_point[0] << " ";												 // Xmin
			// 		out << std::setw(8) << interface_point[1] << " ";												 // Xmax
			// 		out << std::setw(6) << (interface_point[1] - interface_point[0]) * _DF(0.5) / Fs.ini.xa << " ";	 // Lambdax
			// 		out << std::setw(8) << interface_point[4] << " "; // Zmin
			// 		out << std::setw(8) << interface_point[5] << " "; // Zmax
			// 		real_t offsetz = (Fs.Boundarys[4] == 2 && Fs.ini.cop_center_z <= 1.0e-10) ? _DF(1.0) : _DF(0.5);
			// 		out << std::setw(7) << (interface_point[5] - interface_point[4]) * offsetz / Fs.ini.zc << " "; // Lambdaz
			out << "\n";
			out.close();
		}
	}
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

	q.submit([&](sycl::handler &h) {																													   //
		 h.parallel_for(sycl::nd_range<2>(sycl::range<2>(bl.X_inner, bl.Z_inner), sycl::range<2>(bl.dim_block_x, bl.dim_block_z)), [=](nd_item<2> index) { //
			 int i = index.get_global_id(0) + bl.Bwidth_X;
			 int k = index.get_global_id(1) + bl.Bwidth_Z;
			 YDirThetaItegralKernel(i, k, bl, yi, smyXe, smyN2, smyXN);
		 });
	 })
		.wait();

	// 	// #ifdef USE_MPI
	// 	// 	real_t temp = _DF(1.0) / (bl.my * bl.Y_inner);
	// 	// 	int *root_y = new int[bl.mx * bl.mz], size = bl.X_inner *bl.Z_inner;
	// 	// 	for (size_t pos_x = 0; pos_x < bl.mx; pos_x++)
	// 	// 		for (size_t pos_z = 0; pos_z < bl.mz; pos_z++)
	// 	// 		{
	// 	// 			MPI_Group groupy;
	// 	// 			root_y[pos_x * bl.mz + pos_z] = Fs.mpiTrans->Get_RankGroupXZ(groupy, pos_x, pos_z); // only sum ranks at YDIR
	// 	// 			real_t *tempYXe = new real_t[size], *tempYN2 = new real_t[size], *tempYXN = new real_t[size];
	// 	// 			for (size_t jj = 0; jj < bl.Z_inner; jj++)
	// 	// 				for (size_t ii = 0; ii < bl.X_inner; ii++)
	// 	// 				{
	// 	// 					Fs.mpiTrans->GroupallReduce(&(smyXe[bl.X_inner * jj + ii]), &(tempYXe[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 	// 					Fs.mpiTrans->GroupallReduce(&(smyN2[bl.X_inner * jj + ii]), &(tempYN2[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 	// 					Fs.mpiTrans->GroupallReduce(&(smyXN[bl.X_inner * jj + ii]), &(tempYXN[bl.X_inner * jj + ii]), 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::SUM, groupy);
	// 	// 				}
	// 	// 			Fs.mpiTrans->communicator->synchronize();
	// 	// 			if (root_y[pos_x * bl.mz + pos_z] == Fs.mpiTrans->myRank)
	// 	// 				for (size_t jj = 0; jj < bl.Z_inner; jj++)
	// 	// 					for (size_t ii = 0; ii < bl.X_inner; ii++)
	// 	// 					{
	// 	// 						smyXe[bl.X_inner * jj + ii] = temp * tempYXe[bl.X_inner * jj + ii]; //(Y_Xe)^bar=SUM(Y_Xe)/(bl.my*bl.Y_inner)
	// 	// 						smyN2[bl.X_inner * jj + ii] = temp * tempYN2[bl.X_inner * jj + ii]; //(Y_N2)^bar=SUM(Y_N2)/(bl.my*bl.Y_inner)
	// 	// 						smyXN[bl.X_inner * jj + ii] = temp * tempYXN[bl.X_inner * jj + ii]; //(Y_Xe*Y_N2)^bar=SUM(Y_Xe*Y_N2)/(bl.my*bl.Y_inner)
	// 	// 					}
	// 	// 		}
	// 	// #endif // end USE_MPI

	for (size_t i = 0; i < 3; i++)
		theta[i] = _DF(0.0);
	auto Sum_YXN = sycl_reduction_plus(theta[0]);	// sycl::reduction(&(theta[0]), sycl::plus<real_t>());	 // (Y_Xe*Y_N2)^bar
	auto Sum_YXeN2 = sycl_reduction_plus(theta[1]); // sycl::reduction(&(theta[1]), sycl::plus<real_t>());					 // (Y_Xe)^bar*(Y_N2)^bar
	auto Sum_YXe = sycl_reduction_plus(theta[2]);	// sycl::reduction(&(theta[2]), sycl::plus<real_t>());	 // (Y_Xe)^bar*(Y_N2)^bar
	real_t _RomY = _DF(1.0) / real_t(bl.Y_inner);
	q.submit([&](sycl::handler &h) { //
		 h.parallel_for(sycl::nd_range<1>(sycl::range<1>(bl.X_inner * bl.Z_inner), sycl::range<1>(bl.BlockSize)),
						Sum_YXN, Sum_YXeN2, Sum_YXe, [=](nd_item<1> index, auto &tSum_YXN, auto &tSum_YXeN2, auto &tSum_YXe) { //
							auto id = index.get_global_id(0);
							tSum_YXN += smyXN[id];
							tSum_YXeN2 += smyXe[id] * smyN2[id];
							tSum_YXe += smyXe[id];
						});
	 })
		.wait();

	auto local_ndrange3d = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange3d = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	for (size_t i = 0; i < 6; i++)
		interface_point[i] = _DF(0.0);

	// 	// 	auto Rdif_Xmin = reduction(&(interface_point[0]), sycl::minimum<real_t>());
	// 	// 	auto Rdif_Xmax = reduction(&(interface_point[1]), sycl::maximum<real_t>());
	// 	// 	q.submit([&](sycl::handler &h)
	// 	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Rdif_Xmin, Rdif_Xmax, [=](nd_item<3> index, auto &temp_Xmin, auto &temp_Xmax)
	// 	// 							  {
	// 	// 					int i = index.get_global_id(0) + bl.Bwidth_X;
	// 	// 					int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 	// 					int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 	// 					real_t x = i * bl.dx + bl.offx;
	// 	// 					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	// 	// 					if (yi[id * NUM_SPECIES  + bl.Xe_id] > Interface_line)
	// 	// 						temp_Xmin.combine(x), temp_Xmax.combine(x); }); });

	auto Rdif_Ymin = sycl_reduction_min(interface_point[2]); // reduction(&(interface_point[2]), sycl::minimum<real_t>());
	auto Rdif_Ymax = sycl_reduction_max(interface_point[3]); // reduction(&(interface_point[3]), sycl::maximum<real_t>());
	q.submit([&](sycl::handler &h) {						 //
		h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d),
					   Rdif_Ymin, Rdif_Ymax, [=](nd_item<3> index, auto &temp_Ymin, auto &temp_Ymax) { //
						   int i = index.get_global_id(0) + bl.Bwidth_X;
						   int j = index.get_global_id(1) + bl.Bwidth_Y;
						   int k = index.get_global_id(2) + bl.Bwidth_Z;
						   real_t y = j * bl.dy + bl.offy;
						   int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						   if (yi[id * NUM_SPECIES + bl.Xe_id] > Interface_line)
							   temp_Ymin.combine(y), temp_Ymax.combine(y);
					   });
	});

	// 	// 	auto Rdif_Zmin = reduction(&(interface_point[4]), sycl::minimum<real_t>());
	// 	// 	auto Rdif_Zmax = reduction(&(interface_point[5]), sycl::maximum<real_t>());
	// 	// 	q.submit([&](sycl::handler &h)
	// 	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Rdif_Zmin, Rdif_Zmax, [=](nd_item<3> index, auto &temp_Zmin, auto &temp_Zmax)
	// 	// 							  {
	// 	// 					int i = index.get_global_id(0) + bl.Bwidth_X;
	// 	// 					int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 	// 					int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 	// 					real_t z = k * bl.dz + bl.offz;
	// 	// 					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	// 	// 					if (yi[id * NUM_SPECIES + bl.Xe_id] > Interface_line)
	// 	// 						temp_Zmin.combine(z), temp_Zmax.combine(z); }); });

	q.wait();

	if (ReactSources)
	{
		int meshSize = bl.Xmax * bl.Ymax * bl.Zmax;
		auto local_ndrange = range<1>(bl.BlockSize); // size of workgroup
		auto global_ndrange = range<1>(meshSize);
		for (size_t n = 0; n < NUM_SPECIES - 3; n++)
			pVar_max[n] = _DF(0.0);
		// Tmax
		real_t *T = d_fstate.T;
		q.submit([&](sycl::handler &h) {																								//
			auto reduction_max_T = sycl_reduction_max(pVar_max[0]);																		// reduction(&(pVar_max[0]), sycl::maximum<real_t>());
			h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_T, [=](nd_item<1> index, auto &temp_max_T) { //
				auto id = index.get_global_id();
				temp_max_T.combine(T[id]);
			});
		});
		//	reactants
		for (size_t n = 1; n < NUM_SPECIES - 3; n++)
		{
			q.submit([&](sycl::handler &h) {																								  //
				auto reduction_max_Yi = sycl_reduction_max(pVar_max[n]);																	  // reduction(&(pVar_max[n]), sycl::maximum<real_t>());
				h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_Yi, [=](nd_item<1> index, auto &temp_max_Yi) { //
					auto id = index.get_global_id();
					temp_max_Yi.combine(yi[n + 1 + NUM_SPECIES * id]);
				});
			});
		}
		q.wait();
	}

	sigma[0] = _DF(0.0), sigma[1] = _DF(0.0);
	if (Visc)
	{
		auto Sum_Sigma = sycl_reduction_plus(sigma[0]);	 // sycl::reduction(&(sigma[0]), sycl::plus<real_t>());
		auto Sum_Sigma1 = sycl_reduction_plus(sigma[1]); // sycl::reduction(&(sigma[1]), sycl::plus<real_t>());
		q.submit([&](sycl::handler &h) {				 //
			 h.parallel_for(
				 sycl::nd_range<3>(global_ndrange3d, local_ndrange3d), Sum_Sigma, Sum_Sigma1, [=](nd_item<3> index, auto &temp_Sum_Sigma, auto &temp_Sum_Sigma1) { //
					 int i = index.get_global_id(0) + bl.Bwidth_X;
					 int j = index.get_global_id(1) + bl.Bwidth_Y;
					 int k = index.get_global_id(2) + bl.Bwidth_Z;
					 int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
					 temp_Sum_Sigma += vox_2[id] * bl.dx * bl.dy * bl.dz;
					 temp_Sum_Sigma1 += rho[id] * vox_2[id] * bl.dx * bl.dy * bl.dz;
				 });
		 })
			.wait();
	}

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

	if (ReactSources)
	{
		real_t pVar[NUM_SPECIES];
		for (size_t n = 0; n < NUM_SPECIES - 3; n++)
			Fs.mpiTrans->communicator->allReduce(&(pVar_max[n]), &pVar[n], 1, Fs.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		Fs.mpiTrans->communicator->synchronize();
		for (size_t n = 0; n < NUM_SPECIES - 3; n++)
			pVar_max[n] = pVar[n];
	}

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
	// real_t temp_vis = _DF(14.0 / 3.0) * miu_max / rho_min * uvw_c_max[5];
	// dt_ref = sycl::max(dt_ref, temp_vis);
	dt_ref = Fs.BlSz.CFLnumber / dt_ref;
#endif // end USE_MPI

	// Push a count
	AllCountsPush(q, Iter, physicalTime);

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

std::pair<bool, std::vector<float>> Fluid::UpdateFluidStates(sycl::queue &q, int flag)
{
	real_t *UI = NULL;
	if (flag == 0)
		UI = d_U;
	else
		UI = d_U1;

	return UpdateFluidStateFlux(q, Fs, Fs.d_thermal, UI, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma, error_patched_times, rank);
}

void Fluid::UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt)
{
	UpdateURK3rd(q, Fs.BlSz, d_U, d_U1, d_LU, dt, flag);
}

std::vector<float> Fluid::ComputeFluidLU(sycl::queue &q, int flag)
{
	if (flag == 0)
		return GetLU(q, Fs, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
					 material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local_x, d_eigen_local_y, d_eigen_local_z,
					 d_eigen_l, d_eigen_r, uvw_c_max, eigen_block_x, eigen_block_y, eigen_block_z, yi_min, yi_max, Dim_min, Dim_max);
	else
		return GetLU(q, Fs, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
					 material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local_x, d_eigen_local_y, d_eigen_local_z,
					 d_eigen_l, d_eigen_r, uvw_c_max, eigen_block_x, eigen_block_y, eigen_block_z, yi_min, yi_max, Dim_min, Dim_max);
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

	int x_offset = OutBoundary ? 0 : bl.Bwidth_X;
	int y_offset = OutBoundary ? 0 : bl.Bwidth_Y;
	int z_offset = OutBoundary ? 0 : bl.Bwidth_Z;

	bool *error;
	int *error_pos;
	error = middle::MallocShared<bool>(error, 1, q);
	error_pos = middle::MallocShared<int>(error_pos, Emax + 3, q);
	for (size_t n = 0; n < Emax + 3; n++)
		error_pos[n] = 0;
	*error = false;

	auto global_ndrange_max = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	Assign efk(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "EstimateFluidNANKernel");
	std::chrono::high_resolution_clock::time_point runtime_ef_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																					  // sycl::stream error_out(64 * 1024, 10, h);
		 h.parallel_for(sycl::nd_range<3>(efk.global_nd(global_ndrange_max), efk.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0) + x_offset;
			 int j = index.get_global_id(1) + y_offset;
			 int k = index.get_global_id(2) + z_offset;
			 EstimateFluidNANKernel(i, j, k, x_offset, y_offset, z_offset, bl, error_pos, UI, LU, error);
		 });
	 })
		.wait(); //, error_out
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(efk.Time(OutThisTime(runtime_ef_start)));

	if (*error)
	{
		std::cout << "\nErrors of LU[";
		for (size_t ii = 0; ii < Emax - 1; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[Emax - 1] << "] located at (i, j, k)= (";
		std::cout << error_pos[Emax + 1] - x_offset << ", " << error_pos[Emax + 2] - y_offset << ", " << error_pos[Emax + 3] - z_offset;
		if (flag <= 3)
			std::cout << ") inside the step " << flag << " of RungeKutta";
		else
			std::cout << ") after the Reaction solver";
		std::cout << " captured.\n  ERROR Code:\n";
		std::cout << "    10000: negative value of U[0].\n";
		std::cout << "    1100: nanumber value of U[ii].\n";
		std::cout << "    1200: infinite value of U[ii].\n";
		std::cout << "    21: nanumber value of LU[ii].\n";
		std::cout << "    22: infinite value of LU[ii].\n";
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
	if (0 == rank)
		ZeroDimensionalFreelyFlameBlock(Fs);
}

void Fluid::ODESolver(sycl::queue &q, real_t Time)
{
	ChemeODEQ2Solver(q, Fs, Fs.d_thermal, d_fstate, d_U, Fs.d_react, Time);
}
