#include "global_class.h"
#include "marco.h"

LAMNSS::LAMNSS(Setup &setup) : Ss(setup), dt(_DF(0.0)), Iteration(0), rank(0), nranks(1), physicalTime(0.0)
{
#if USE_MPI
	rank = Ss.mpiTrans->myRank;
	nranks = Ss.mpiTrans->nProcs;
#endif // end USE_MPI
	MPI_trans_time = 0.0, MPI_BCs_time = 0.0;
	for (int n = 0; n < NumFluid; n++)
	{
		fluids[n] = new Fluid(setup);
#if 1 != NumFluid
		fluids[n]->initialize(n);
#endif
	}

	if (Ss.OutBoundary)
	{
		VTI.nbX = Ss.BlSz.Xmax;
		VTI.minX = 0;
		VTI.maxX = Ss.BlSz.Xmax;

		VTI.nbY = Ss.BlSz.Ymax;
		VTI.minY = 0;
		VTI.maxY = Ss.BlSz.Ymax;

		VTI.nbZ = Ss.BlSz.Zmax;
		VTI.minZ = 0;
		VTI.maxZ = Ss.BlSz.Zmax;
	}
	else
	{
		VTI.nbX = Ss.BlSz.X_inner;
		VTI.minX = Ss.BlSz.Bwidth_X;
		VTI.maxX = Ss.BlSz.Xmax - Ss.BlSz.Bwidth_X;

		VTI.nbY = Ss.BlSz.Y_inner;
		VTI.minY = Ss.BlSz.Bwidth_Y;
		VTI.maxY = Ss.BlSz.Ymax - Ss.BlSz.Bwidth_Y;

		VTI.nbZ = Ss.BlSz.Z_inner;
		VTI.minZ = Ss.BlSz.Bwidth_Z;
		VTI.maxZ = Ss.BlSz.Zmax - Ss.BlSz.Bwidth_Z;
	}

	PLT = VTI;

#ifdef USE_MPI
#if DIM_X
	if (Ss.mpiTrans->neighborsBC[XMIN] == BC_COPY)
		if (Ss.OutDIRX)
			PLT.nbX += 1;
	if (Ss.mpiTrans->neighborsBC[XMAX] == BC_COPY)
		if (Ss.OutDIRX)
			PLT.nbX += 1;
#endif

#if DIM_Y
	if (Ss.mpiTrans->neighborsBC[YMIN] == BC_COPY)
		if (Ss.OutDIRY)
			PLT.nbY += 1;
	if (Ss.mpiTrans->neighborsBC[YMAX] == BC_COPY)
		if (Ss.OutDIRY)
			PLT.nbY += 1;
#endif

#if DIM_Z
	if (Ss.mpiTrans->neighborsBC[ZMIN] == BC_COPY)
		if (Ss.OutDIRZ)
			PLT.nbZ += 1;
	if (Ss.mpiTrans->neighborsBC[ZMAX] == BC_COPY)
		if (Ss.OutDIRZ)
			PLT.nbZ += 1;
#endif
#endif // use MPI

	CPT.minX = Ss.BlSz.Bwidth_X + Ss.outpos_x;
	CPT.minY = Ss.BlSz.Bwidth_Y + Ss.outpos_y;
	CPT.minZ = Ss.BlSz.Bwidth_Z + Ss.outpos_z;
	CPT.nbX = Ss.OutDIRX ? PLT.nbX : 1;
	CPT.nbY = Ss.OutDIRY ? PLT.nbY : 1;
	CPT.nbZ = Ss.OutDIRZ ? PLT.nbZ : 1;
	CPT.maxX = CPT.minX + CPT.nbX;
	CPT.maxY = CPT.minY + CPT.nbY;
	CPT.maxZ = CPT.minZ + CPT.nbZ;
}

LAMNSS::~LAMNSS()
{
	for (size_t n = 0; n < NumFluid; n++)
		fluids[n]->~Fluid();
}

float LAMNSS::OutThisTime(std::chrono::high_resolution_clock::time_point start_time)
{
#ifdef USE_MPI
	if (rank == 0)
#endif // end USE_MPI
	{
		std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration<float, std::milli>(end_time - start_time).count() / 1000.0f;
		std::cout << ", runtime: " << std::setw(10) << duration << "\n";
	}
	return duration;
}

void LAMNSS::Evolution(sycl::queue &q)
{
	bool TimeLoopOut = false, Stepstop = false;
	int OutNum = 1, TimeLoop = 0, error_out = 0, RcalOut = 0;

	duration = 0.0f;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	while (TimeLoop < Ss.nOutTimeStamps)
	{
		real_t target_t = physicalTime < Ss.OutTimeStamps[TimeLoop] ? Ss.OutTimeStamps[TimeLoop] : Ss.OutTimeStamps[TimeLoop++]; // std::max(Ss.OutTimeStamp, TimeLoop * Ss.OutTimeStamp + Ss.OutTimeStart);
		// TimeLoop++;
		while (physicalTime < target_t)
		{
			CopyToUbak(q);
			if (Iteration % Ss.OutInterval == 0 && OutNum <= Ss.nOutput || TimeLoopOut)
			{
				Output(q, rank, std::to_string(Iteration), physicalTime);
				OutNum++;
				TimeLoopOut = false;
			}
			if (RcalOut % 100 == 0)
				Output_Ubak(rank, Iteration, physicalTime);
			RcalOut++;
			// get minmum dt, if MPI used, get the minimum of all ranks
			dt = ComputeTimeStep(q); // 2.0e-6; //

			Iteration++;
			if (rank == 0) // An iteration begins at the physicalTime output on screen and ends at physicalTime + dt, which is the physicalTime of the next iteration
				std::cout << "N=" << std::setw(7) << Iteration << "  beginning physicalTime: " << std::setw(14) << std::setprecision(8) << physicalTime;
#ifdef USE_MPI
			Ss.mpiTrans->communicator->synchronize();
			real_t temp;
			Ss.mpiTrans->communicator->allReduce(&dt, &temp, 1, Ss.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
			dt = temp;
			if (rank == 0)
				std::cout << "  mpi communicated";
#endif // end USE_MPI
			if (physicalTime + dt > target_t)
				dt = target_t - physicalTime;
			if (rank == 0)
				std::cout << " dt: " << dt << " to do";
			physicalTime += dt;

#if CHEME_SPLITTING == 2
			error_out = error_out || Reaction(q, 0.5 * dt, physicalTime, Iteration);
#endif // end CHEME_SPLITTING

			// solved the fluid with 3rd order Runge-Kutta method
			error_out = error_out || SinglePhaseSolverRK3rd(q, rank, Iteration, physicalTime);

#if CHEME_SPLITTING == 2
			error_out = error_out || Reaction(q, 0.5 * dt, physicalTime, Iteration);
#elif CHEME_SPLITTING == 1
			error_out = error_out || Reaction(q, dt, physicalTime, Iteration);
#endif // end CHEME_SPLITTING

#ifdef ESTIM_NAN
#ifdef USE_MPI
			int root, maybe_root = (error_out ? rank : 0);
			Ss.mpiTrans->communicator->allReduce(&maybe_root, &root, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX);
			Ss.mpiTrans->communicator->bcast(&(error_out), 1, mpiUtils::MpiComm::INT, root);
#endif
			if (error_out)
				goto flag_ernd;
#endif

			Stepstop = Ss.nStepmax <= Iteration ? true : false;
			if (Stepstop)
				goto flag_end;
			OutThisTime(start_time);
		}
		TimeLoopOut = true;
	}

flag_end:
#ifdef COP_CHEME
	BoundaryCondition(q, 0);
	UpdateStates(q, 0, physicalTime, Iteration, "_End");
#endif // end COP_CHEME
flag_ernd:
#ifdef USE_MPI
	Ss.mpiTrans->communicator->synchronize();
#endif
	OutThisTime(start_time);
	EndProcess();
	// Output_Counts();
	// Output_Ubak(rank, Iteration - 1, physicalTime);
	Output(q, rank, std::to_string(Iteration), physicalTime); // The last step Output.
}

void LAMNSS::EndProcess()
{
#ifdef USE_MPI
	for (size_t n = 0; n < NumFluid; n++)
	{
		MPI_trans_time += fluids[n]->MPI_trans_time;
		MPI_BCs_time += fluids[n]->MPI_BCs_time;
	}
	float Ttemp, BCsTtemp, TransTtemp;
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->allReduce(&duration, &Ttemp, 1, mpiUtils::MpiComm::FLOAT, mpiUtils::MpiComm::SUM);
	Ss.mpiTrans->communicator->allReduce(&MPI_BCs_time, &BCsTtemp, 1, mpiUtils::MpiComm::FLOAT, mpiUtils::MpiComm::SUM);
	Ss.mpiTrans->communicator->allReduce(&MPI_trans_time, &TransTtemp, 1, mpiUtils::MpiComm::FLOAT, mpiUtils::MpiComm::SUM);
	Ss.mpiTrans->communicator->synchronize();
	duration = Ttemp;
	if (rank == 0)
	{
		std::cout << "MPI averaged of " << nranks << " ranks ";
///////////////////////////
#ifdef AWARE_MPI
		std::cout << "with    AWARE_MPI ";
#else
		std::cout << "without AWARE_MPI ";
#endif // end AWARE_MPI
///////////////////////////
#endif // end USE_MPI
		std::cout << SelectDv << " runtime(s):  " << std::setw(8) << std::setprecision(6) << duration / float(nranks) << std::endl;
		std::cout << "Device Memory Usage(GB)   :  " << fluids[0]->MemMbSize / 1024.0 << std::endl;
#ifdef USE_MPI
		std::cout << "MPI trans Memory Size(GB) :  " << fluids[0]->MPIMbSize / 1024.0 << std::endl;
		std::cout << "Fluids do BCs time(s)     :  " << std::setw(8) << std::setprecision(6) << BCsTtemp / float(nranks) << std::endl;
		std::cout << "MPI buffers Trans time(s) :  " << std::setw(8) << std::setprecision(6) << TransTtemp / float(nranks) << std::endl;
	}
#endif
	int error_times_patched = 0;
	for (size_t n = 0; n < NumFluid; n++)
		error_times_patched += fluids[n]->error_patched_times;
#ifdef USE_MPI
	int Etemp;
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->allReduce(&error_times_patched, &Etemp, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::SUM);
	Ss.mpiTrans->communicator->synchronize();
	error_times_patched = Etemp;
	if (rank == 0)
#endif
	{
		std::cout << "Times of error patched: " << error_times_patched << std::endl;
	}
}

bool LAMNSS::SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t Time)
{
	// estimate if rho is_nan or <0 or is_inf
	int root, maybe_root, error1 = 0, error2 = 0, error3 = 0;
	error1 = RungeKuttaSP3rd(q, rank, Step, Time, 1);
	if (error1)
		return true;

	error2 = RungeKuttaSP3rd(q, rank, Step, Time, 2);
	if (error2)
		return true;

	error3 = RungeKuttaSP3rd(q, rank, Step, Time, 3);

#ifdef ERROR_PATCH
	int error_times_patched = 0;
	for (size_t n = 0; n < NumFluid; n++)
		error_times_patched += fluids[n]->error_patched_times;
#ifdef USE_MPI
	int Etemp;
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->allReduce(&error_times_patched, &Etemp, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::SUM);
	Ss.mpiTrans->communicator->synchronize();
	error_times_patched = Etemp;
#endif
	if (error_times_patched > 10)
	{
		if (0 == rank)
			std::cout << "Too many NAN error times captured to patch, return error.\n";
		return true;
	}
#endif // end ERROR_PATCH

	return error3;
}

bool LAMNSS::EstimateNAN(sycl::queue &q, const real_t Time, const int Step, const int rank, const int flag)
{
	bool error = false, errors[NumFluid];
	int root, maybe_root, error_out = 0;
	for (int n = 0; n < NumFluid; n++)
	{
		errors[n] = fluids[n]->EstimateFluidNAN(q, flag);
		error = error || errors[n];
		// if (rank == 1 && Step == 10)
		// 	error = true;
	}
	maybe_root = (error ? rank : 0), error_out = error;
#ifdef USE_MPI
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->allReduce(&maybe_root, &root, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX);
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->bcast(&(error_out), 1, mpiUtils::MpiComm::INT, root);
	Ss.mpiTrans->communicator->synchronize();
#endif // end USE_MPI
	if (error)
		Output(q, rank, "UErs_" + std::to_string(Step) + "_RK" + std::to_string(flag), Time, true);
	if (error_out)
		Output(q, rank, "UErr_" + std::to_string(Step) + "_RK" + std::to_string(flag), Time, false);
	error = bool(error_out);

	q.wait();

	return error; // all rank == 1 or 0
}

bool LAMNSS::RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag)
{
	// estimate if rho is_nan or <0 or is_inf
	bool error = false; //, errorp1 = false, errorp2 = false, errorp3 = false;
	switch (flag)
	{
	case 1:
		// the fisrt step
		BoundaryCondition(q, 0);
		if (UpdateStates(q, 0, Time, Step, "_RK1"))
			return true;

		ComputeLU(q, 0);
#ifdef ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		UpdateU(q, 1);
		break;

	case 2:
		// the second step
		BoundaryCondition(q, 1);
		if (UpdateStates(q, 1, Time, Step, "_RK2"))
			return true;

		ComputeLU(q, 1);
#ifdef ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		UpdateU(q, 2);
		break;

	case 3:
		// the third step
		BoundaryCondition(q, 1);
		if (UpdateStates(q, 1, Time, Step, "_RK3"))
			return true;

		ComputeLU(q, 1);
#ifdef ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		UpdateU(q, 3);
		break;
	}

	return false;
}

real_t LAMNSS::ComputeTimeStep(sycl::queue &q)
{
	real_t dt_ref = _DF(1.0e-10);
#if NumFluid == 1
	dt_ref = fluids[0]->GetFluidDt(q, Iteration, physicalTime);
#elif NumFluid == 2
	dt_ref = fluids[0]->GetFluidDt(levelset, Iteration, physicalTime);
	for (size_t n = 1; n < NumFluid; n++)
		dt_ref = min(dt_ref, fluids[1]->GetFluidDt(levelset, Iteration, physicalTime));
#endif

	return dt_ref;
}

void LAMNSS::ComputeLU(sycl::queue &q, int flag)
{
	fluids[0]->ComputeFluidLU(q, flag);
}

void LAMNSS::UpdateU(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->UpdateFluidURK3(q, flag, dt);
}

void LAMNSS::BoundaryCondition(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->BoundaryCondition(q, Ss.Boundarys, flag);
}

bool LAMNSS::UpdateStates(sycl::queue &q, int flag, const real_t Time, const int Step, std::string RkStep)
{

	bool error[NumFluid] = {false}, error_t = false;
	for (int n = 0; n < NumFluid; n++)
	{
		error[n] = fluids[n]->UpdateFluidStates(q, flag);
		// if (Time > 0.0000003 && rank == 1)
		// 	error[n] = true;
		error_t = error_t || error[n]; // rank error
	}
	// if (Step)
	{
#ifdef ESTIM_NAN
		std::string Stepstr = std::to_string(Step);
		if (error_t)
		{
			Output(q, rank, "PErs_" + Stepstr + RkStep, Time, true);
			std::cout << "Output DIR(X, Y, Z = " << Ss.OutDIRX << ", " << Ss.OutDIRY << ", " << Ss.OutDIRZ << ") has been done at Step = PErs_" << Stepstr << RkStep << std::endl;
		}
#ifdef USE_MPI
		int root, maybe_root = (error_t ? rank : 0), error_out = error_t; // error_out==1 in all rank for all rank out after bcast
		Ss.mpiTrans->communicator->synchronize();
		Ss.mpiTrans->communicator->allReduce(&maybe_root, &root, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX);
		Ss.mpiTrans->communicator->synchronize();
		Ss.mpiTrans->communicator->bcast(&(error_out), 1, mpiUtils::MpiComm::INT, root);
		Ss.mpiTrans->communicator->synchronize();
		error_t = bool(error_out);
#endif // end USE_MPI
		if (error_t)
			Output(q, rank, "PErr_" + Stepstr + RkStep, Time, false);
#else
		error_t = false;
#endif // end ESTIM_NAN
	}

	return error_t; // all rank == 1 or 0
}

void LAMNSS::AllocateMemory(sycl::queue &q)
{
	d_BCs = static_cast<BConditions *>(malloc_device(6 * sizeof(BConditions), q));

	q.memcpy(d_BCs, Ss.Boundarys, 6 * sizeof(BConditions));

	// host arrays for each fluid
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->AllocateFluidMemory(q);

	// levelset->AllocateLSMemory();
}

void LAMNSS::InitialCondition(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->InitialU(q);

	Read_Ubak(q, rank, &(Iteration), &(physicalTime));
}

bool LAMNSS::Reaction(sycl::queue &q, real_t dt, real_t Time, const int Step)
{
#ifdef COP_CHEME
	BoundaryCondition(q, 0);
	bool error = UpdateStates(q, 0, Time, Step, "_React");
	if (error)
		return true;
	fluids[0]->ODESolver(q, dt);

	return EstimateNAN(q, Time, Step, rank, 4);
#endif // end COP_CHEME
	return false;
}

void LAMNSS::CopyToUbak(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->Ubak, fluids[n]->d_U, Ss.cellbytes);
	q.wait();
}

void LAMNSS::CopyToU(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->d_U, fluids[n]->h_U, Ss.cellbytes);
	q.wait();
}

void LAMNSS::Output_Ubak(const int rank, const int Step, const real_t Time)
{
	std::string file_name, outputPrefix = INI_SAMPLE;
	file_name = Ss.OutputDir + "/" + outputPrefix + "_ReCal";
#ifdef USE_MPI
	file_name += "_rank_" + std::to_string(rank);
#endif

	std::ofstream fout(file_name, std::ios::out | std::ios::binary);
	fout.write((char *)&(Step), sizeof(Step));
	fout.write((char *)&(Time), sizeof(Time));
	for (size_t n = 0; n < NumFluid; n++)
		fout.write((char *)(fluids[n]->Ubak), Ss.cellbytes);
	fout.close();

	if (rank == 0)
		std::cout << "ReCal-file of Step = " << Step << " has been output." << std::endl;
}

bool LAMNSS::Read_Ubak(sycl::queue &q, const int rank, int *Step, real_t *Time)
{
	int size = Ss.cellbytes, all_read = 1;
	std::string file_name, outputPrefix = INI_SAMPLE;
	file_name = Ss.OutputDir + "/" + outputPrefix + "_ReCal";
#ifdef USE_MPI
	file_name += "_rank_" + std::to_string(rank);
#endif

	std::ifstream fin(file_name, std::ios::in | std::ios::binary);
	if (!fin.is_open())
		all_read = 0;
#ifdef USE_MPI
	int root, maybe_root = (all_read ? 0 : rank);
	Ss.mpiTrans->communicator->allReduce(&maybe_root, &root, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX);
	Ss.mpiTrans->communicator->bcast(&(all_read), 1, mpiUtils::MpiComm::INT, root);
#endif
	if (!all_read)
	{
		if (rank == 0)
			std::cout << "ReCal-file not exist or open failed, ReCal closed." << std::endl;
		return false;
	}
	fin.read((char *)Step, sizeof(int));
	fin.read((char *)Time, sizeof(real_t));
	for (size_t n = 0; n < NumFluid; n++)
		fin.read((char *)(fluids[n]->Ubak), size);
	fin.close();

	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->d_U, fluids[n]->Ubak, Ss.cellbytes);
	q.wait();

	return true; // ReIni U for additonal continued caculate
}

void LAMNSS::CopyDataFromDevice(sycl::queue &q, bool error)
{
	// copy mem from device to host
	int bytes = Ss.bytes, cellbytes = Ss.cellbytes;
	for (int n = 0; n < NumFluid; n++)
	{
		q.memcpy(fluids[n]->h_fstate.rho, fluids[n]->d_fstate.rho, bytes);
		q.memcpy(fluids[n]->h_fstate.p, fluids[n]->d_fstate.p, bytes);
		q.memcpy(fluids[n]->h_fstate.c, fluids[n]->d_fstate.c, bytes);
		q.memcpy(fluids[n]->h_fstate.H, fluids[n]->d_fstate.H, bytes);
		q.memcpy(fluids[n]->h_fstate.u, fluids[n]->d_fstate.u, bytes);
		q.memcpy(fluids[n]->h_fstate.v, fluids[n]->d_fstate.v, bytes);
		q.memcpy(fluids[n]->h_fstate.w, fluids[n]->d_fstate.w, bytes);
		q.memcpy(fluids[n]->h_fstate.T, fluids[n]->d_fstate.T, bytes);
		q.memcpy(fluids[n]->h_fstate.e, fluids[n]->d_fstate.e, bytes);
		q.memcpy(fluids[n]->h_fstate.gamma, fluids[n]->d_fstate.gamma, bytes);

		for (size_t i = 0; i < 3; i++)
			q.memcpy(fluids[n]->h_fstate.vxs[i], fluids[n]->d_fstate.vxs[i], bytes).wait();
		q.memcpy(fluids[n]->h_fstate.vx, fluids[n]->d_fstate.vx, bytes);
#ifdef COP
		q.memcpy(fluids[n]->h_fstate.y, fluids[n]->d_fstate.y, bytes * NUM_SPECIES);
#endif // COP
#ifdef ESTIM_NAN
		if (error)
		{
#ifdef Visc // copy vosicous estimating Vars
#if DIM_X
			q.memcpy(fluids[n]->h_fstate.visFwx, fluids[n]->d_fstate.visFwx, NUM_SPECIES * bytes);
#endif
#if DIM_Y
			q.memcpy(fluids[n]->h_fstate.visFwy, fluids[n]->d_fstate.visFwy, NUM_SPECIES * bytes);
#endif
#if DIM_Z
			q.memcpy(fluids[n]->h_fstate.visFwz, fluids[n]->d_fstate.visFwz, NUM_SPECIES * bytes);
#endif
#ifdef Visc_Diffu
			q.memcpy(fluids[n]->h_fstate.Ertemp1, fluids[n]->d_fstate.Ertemp1, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Ertemp2, fluids[n]->d_fstate.Ertemp2, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Dkm_aver, fluids[n]->d_fstate.Dkm_aver, NUM_SPECIES * bytes);
#if DIM_X
			q.memcpy(fluids[n]->h_fstate.Dim_wallx, fluids[n]->d_fstate.Dim_wallx, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.hi_wallx, fluids[n]->d_fstate.hi_wallx, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yi_wallx, fluids[n]->d_fstate.Yi_wallx, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yil_wallx, fluids[n]->d_fstate.Yil_wallx, NUM_SPECIES * bytes);
#endif
#if DIM_Y
			q.memcpy(fluids[n]->h_fstate.Dim_wally, fluids[n]->d_fstate.Dim_wally, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.hi_wally, fluids[n]->d_fstate.hi_wally, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yi_wally, fluids[n]->d_fstate.Yi_wally, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yil_wally, fluids[n]->d_fstate.Yil_wally, NUM_SPECIES * bytes);
#endif
#if DIM_Z
			q.memcpy(fluids[n]->h_fstate.Dim_wallz, fluids[n]->d_fstate.Dim_wallz, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.hi_wallz, fluids[n]->d_fstate.hi_wallz, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yi_wallz, fluids[n]->d_fstate.Yi_wallz, NUM_SPECIES * bytes);
			q.memcpy(fluids[n]->h_fstate.Yil_wallz, fluids[n]->d_fstate.Yil_wallz, NUM_SPECIES * bytes);
#endif
#endif // end Visc_Diffu
#endif // end Visc

			q.memcpy(fluids[n]->h_U, fluids[n]->d_U, cellbytes);
			q.memcpy(fluids[n]->h_U1, fluids[n]->d_U1, cellbytes);
			q.memcpy(fluids[n]->h_LU, fluids[n]->d_LU, cellbytes);
#if DIM_X
			q.memcpy(fluids[n]->h_fstate.b1x, fluids[n]->d_fstate.b1x, bytes);
			q.memcpy(fluids[n]->h_fstate.b3x, fluids[n]->d_fstate.b3x, bytes);
			q.memcpy(fluids[n]->h_fstate.c2x, fluids[n]->d_fstate.c2x, bytes);
			q.memcpy(fluids[n]->h_fstate.zix, fluids[n]->d_fstate.zix, bytes * NUM_COP);
			q.memcpy(fluids[n]->h_fstate.preFwx, fluids[n]->d_fstate.preFwx, cellbytes);
			q.memcpy(fluids[n]->h_fstate.pstFwx, fluids[n]->d_wallFluxF, cellbytes);
#endif
#if DIM_Y
			q.memcpy(fluids[n]->h_fstate.b1y, fluids[n]->d_fstate.b1y, bytes);
			q.memcpy(fluids[n]->h_fstate.b3y, fluids[n]->d_fstate.b3y, bytes);
			q.memcpy(fluids[n]->h_fstate.c2y, fluids[n]->d_fstate.c2y, bytes);
			q.memcpy(fluids[n]->h_fstate.ziy, fluids[n]->d_fstate.ziy, bytes * NUM_COP);
			q.memcpy(fluids[n]->h_fstate.preFwy, fluids[n]->d_fstate.preFwy, cellbytes);
			q.memcpy(fluids[n]->h_fstate.pstFwy, fluids[n]->d_wallFluxG, cellbytes);
#endif
#if DIM_Z
			q.memcpy(fluids[n]->h_fstate.b1z, fluids[n]->d_fstate.b1z, bytes);
			q.memcpy(fluids[n]->h_fstate.b3z, fluids[n]->d_fstate.b3z, bytes);
			q.memcpy(fluids[n]->h_fstate.c2z, fluids[n]->d_fstate.c2z, bytes);
			q.memcpy(fluids[n]->h_fstate.ziz, fluids[n]->d_fstate.ziz, bytes * NUM_COP);
			q.memcpy(fluids[n]->h_fstate.preFwz, fluids[n]->d_fstate.preFwz, cellbytes);
			q.memcpy(fluids[n]->h_fstate.pstFwz, fluids[n]->d_wallFluxH, cellbytes);
#endif
		}
#endif // end ESTIM_NAN
	}
	q.wait();
}

// void LAMNSS::Output_Counts()
// {
// 	if (rank == 0)
// 	{
// 		real_t rho0 = Ss.ini.blast_density_in;
// 		std::string outputPrefix = INI_SAMPLE;
// 		std::string file_name = Ss.OutputDir + "/AllCounts_" + outputPrefix + "_rho0_" + std::to_string(Ss.ini.blast_density_in) + "_" + std::to_string(Ss.ini.cop_density_in) + "_" + std::to_string(Ss.ini.blast_density_out) + ".plt";
// 		std::ofstream out(file_name);
// 		// // defining header for tecplot(plot software)
// 		out.setf(std::ios::right);
// 		out << "title='" << outputPrefix << "'\n"
// 			<< "variables=Time[s], <b><greek>Q</greek></b>[-], <greek>e</greek><sub><greek>r</greek></sub>[m<sup>2</sup>/s<sup>2</sup>], <i>T</i><sub>max</sub>[K], ";
// 		// Time[s]: Time in tecplot x-Axis variable
// 		//<greek>Q</greek>[-]: (theta(Theta(XN)/Theta(Xe)/Theta(N2))) in tecplot
// 		//<greek>e</greek><sub><greek>r</greek></sub>: sigma in tecplot
// 		//<sub>max</sub>: T_max in tecplot sub{max} added
// 		//<i>Y(HO2)</i><sub>max</sub>[-]: Yi(HO2)_max
// 		//<i>Y(H2O2)</i><sub>max</sub>[-]: Yi(H2O2)_max
// 		//<greek>L</greek><sub>x</sub>[-]: Gamx in tecplot
// 		//<greek>L</greek><sub>y</sub>[-]: Gamy in tecplot
// 		//<greek>L</greek><sub>z</sub>[-]: Gamz in tecplot

// #ifdef COP_CHEME
// 		for (size_t n = 1; n < NUM_SPECIES - 3; n++)
// 			out << "<i>Y(" << Ss.species_name[n + 1] << ")</i><sub>max</sub>[-], ";
// 			// out << "<i>Y(HO2)</i><sub>max</sub>[-], <i>Y(H2O2)</i><sub>max</sub>[-], ";
// #endif // end COP_CHEME
// #if DIM_Y
// 		out << "<greek>L</greek><sub>y</sub>[-], ";
// #endif
// 		out << "Theta(Xe), Theta(N2), Theta(XN), ";
// #if DIM_Y
// 		out << "Ymin, Ymax, ";
// #endif
// #if DIM_X
// 		out << "Xmin, Xmax, <greek>L</greek><sub>x</sub>[-]";
// #endif
// #if DIM_Z
// 		out << "Zmin, Zmax, <greek>L</greek><sub>z</sub>[-], ";
// #endif
// 		out << "\nzone t='Time_Theta_Sigma_T";
// #ifdef COP_CHEME
// 		for (size_t n = 1; n < NUM_SPECIES - 3; n++)
// 			out << "_Y(" << Ss.species_name[n + 1] << "), ";
// #endif // end COP_CHEME
// 		out << "_Gamy_teXN_teXeN2_TeXN_Ymin_Ymax";
// #if DIM_X
// 		out << "_Xmin_Xmax_Gamx";
// #endif // end DIM_X
// #if DIM_Z
// 		out << "_Zmin_Zmax_Gamz";
// #endif // end DIM_Z
// 		out << "', i= " << fluids[0]->pTime.size() << ", j= 1, k= 1 \n";

// 		for (int i = 0; i < fluids[0]->pTime.size(); i++)
// 		{
// 			out << std::setw(11) << fluids[0]->pTime[i] << " ";		// physical time
// 			out << std::setw(11) << fluids[0]->Theta[i] << " ";		// Theta(XN/(Xe*N2))
// 			out << std::setw(11) << fluids[0]->Sigma[i] / rho0 << " "; // sigma: sigma_rho*rho_0(with no rho0 definition found)
// 			out << std::setw(7) << fluids[0]->Var_max[0][i] << " "; // Tmax
// #ifdef COP_CHEME
// 			for (size_t n = 1; n < NUM_SPECIES - 3; n++)
// 				out << std::setw(11) << fluids[0]->Var_max[n][i] << " ";
// 				// out << std::setw(11) << fluids[0]->Var_max[1][i] << " ";				  // Yi(HO2)_max
// 				// out << std::setw(11) << fluids[0]->Var_max[2][i] << " ";				  // Yi(H2O2)_max
// #endif																				  // end COP_CHEME
// #if DIM_Y
// 			real_t offsety = (Ss.Boundarys[2] == 2 && Ss.ini.cop_center_y <= 1.0e-10) ? _DF(1.0) : _DF(0.5);
// 			out << std::setw(7) << (fluids[0]->Interface_points[3][i] - fluids[0]->Interface_points[2][i]) * offsety / Ss.ini.yb << " "; // Gamy
// #endif
// 			// out << std::setw(7) << i + 1 << " ";				   // Step
// 			out << std::setw(11) << fluids[0]->thetas[0][i] << " "; // [0]XN
// 			out << std::setw(11) << fluids[0]->thetas[1][i] << " "; // [1]Xe*N2
// 			out << std::setw(3) << fluids[0]->thetas[2][i] << " "; // Theta(XN)
// #if DIM_Y
// 			out << std::setw(8) << fluids[0]->Interface_points[2][i] << " "; // Ymin
// 			out << std::setw(8) << fluids[0]->Interface_points[3][i] << " "; // Ymax
// #endif
// #if DIM_X
// 			out << std::setw(8) << fluids[0]->Interface_points[0][i] << " ";															  // Xmin
// 			out << std::setw(8) << fluids[0]->Interface_points[1][i] << " ";															  // Xmax
// 			out << std::setw(6) << (fluids[0]->Interface_points[1][i] - fluids[0]->Interface_points[0][i]) * _DF(0.5) / Ss.ini.xa << " "; // Gamx
// #endif
// #if DIM_Z
// 			out << std::setw(8) << fluids[0]->Interface_points[4][i] << " "; // Zmin
// 			out << std::setw(8) << fluids[0]->Interface_points[5][i] << " "; // Zmax
// 			real_t offsetz = (Ss.Boundarys[4] == 2 && Ss.ini.cop_center_z <= 1.0e-10) ? _DF(1.0) : _DF(0.5);
// 			out << std::setw(7) << (fluids[0]->Interface_points[5][i] - fluids[0]->Interface_points[4][i]) * offsetz / Ss.ini.zc << " "; // Gamz
// #endif
// 			out << "\n";
// 		}
// 		out.close();
// 	}
// }

void LAMNSS::Output(sycl::queue &q, int rank, std::string interation, real_t Time, bool error)
{
	// Write time in string timeFormat
	std::ostringstream timeFormat;
	timeFormat.width(11);
	timeFormat.fill('0');
	timeFormat << Time;
	// Write istep in string stepFormat
	std::ostringstream stepFormat;
	stepFormat.width(7);
	stepFormat.fill('0');
	stepFormat << interation;
	// Write Mpi Rank in string rankFormat
	std::ostringstream rankFormat;
	rankFormat.width(5);
	rankFormat.fill('0');
	rankFormat << rank;

	CopyDataFromDevice(q, error); // only copy when output

	if (error)
	{ // only out pvti and vti error files
		Output_vti(rank, timeFormat, stepFormat, rankFormat, true);
	}
	else if (!(Ss.OutDIRX && Ss.OutDIRY && Ss.OutDIRZ))
	{
#ifdef OUT_PLT
		Output_cplt(rank, timeFormat, stepFormat, rankFormat);
#endif
#ifdef OUT_VTI
		Output_cvti(rank, timeFormat, stepFormat, rankFormat);
#endif // end OUT_PLT
	}
	else
	{
#ifdef OUT_PLT
		Output_plt(rank, timeFormat, stepFormat, rankFormat, error);
#endif
#ifdef OUT_VTI
		Output_vti(rank, timeFormat, stepFormat, rankFormat, error);
#endif // end OUT_PLT
	}

	if (rank == 0)
		std::cout << "Output DIR(X, Y, Z = " << (Ss.OutDIRX && DIM_X) << ", " << (Ss.OutDIRY && DIM_Y) << ", " << (Ss.OutDIRZ && DIM_Z) << ") has been done at Step = " << interation << std::endl;
}

void LAMNSS::Output_vti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error)
{
	// Init var names
	int Onbvar = 6 + (DIM_X + DIM_Y + DIM_Z) * 2; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES;
#endif // end COP
	std::map<int, std::string> variables_names;
	int index = 0;
#if DIM_X
	variables_names[index] = "DIR-X";
	index++;
	variables_names[index] = "OV-u";
	index++;
#endif // end DIM_X
#if DIM_Y
	variables_names[index] = "DIR-Y";
	index++;
	variables_names[index] = "OV-v";
	index++;
#endif // end DIM_Y
#if DIM_Z
	variables_names[index] = "DIR-Z";
	index++;
	variables_names[index] = "OV-w";
	index++;
#endif // end DIM_Z
#ifdef ESTIM_NAN
	if (error)
	{
#ifdef Visc // Out name of viscous out estimating Vars
		Onbvar += Emax * (DIM_X + DIM_Y + DIM_Z);
		for (size_t mm = 0; mm < Emax; mm++)
		{
#if DIM_X
			variables_names[index] = "E-Fw-vis-x[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_X
#if DIM_Y
			variables_names[index] = "E-Fw-vis-y[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_Y
#if DIM_Z
			variables_names[index] = "E-Fw-vis-z[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_Z
		}
#ifdef Visc_Diffu
		Onbvar += (NUM_SPECIES * 3);
		Onbvar += (NUM_SPECIES * 3) * (DIM_X + DIM_Y + DIM_Z);
		Onbvar += (NUM_SPECIES) * (DIM_X + DIM_Y + DIM_Z);
		// Onbvar += (NUM_SPECIES * 3) * (DIM_X + DIM_Y + DIM_Z);

		for (size_t mm = 0; mm < NUM_SPECIES; mm++)
		{
			variables_names[index] = "E-vis_Dim[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Dimtemp1[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Dimtemp2[" + std::to_string(mm) + "]";
			index++;

#if DIM_X
			variables_names[index] = "E-vis_Dimwallx[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_hi_wallx[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yi_wallx[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yil_wallx[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_X
#if DIM_Y
			variables_names[index] = "E-vis_Dimwally[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_hi_wally[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yi_wally[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yil_wally[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_Y
#if DIM_Z
			variables_names[index] = "E-vis_Dimwallz[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_hi_wallz[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yi_wallz[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-vis_Yil_wallz[" + std::to_string(mm) + "]";
			index++;
#endif // DIM_Z
		}
#endif
#endif
#if DIM_X
		Onbvar += (3 + NUM_COP);
		variables_names[index] = "E-xb1";
		index++;
		variables_names[index] = "E-xb3";
		index++;
		variables_names[index] = "E-xc2";
		index++;
		for (size_t nn = 0; nn < NUM_COP; nn++)
		{
			variables_names[index] = "E-xzi[" + std::to_string(nn) + "]";
			index++;
		}
		Onbvar += 2 * Emax;
		for (size_t mm = 0; mm < Emax; mm++)
		{
			variables_names[index] = "E-Fw-prev-x[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-Fw-pstv-x[" + std::to_string(mm) + "]";
			index++;
		}
#endif // end DIM_X
#if DIM_Y
		Onbvar += (3 + NUM_COP);
		variables_names[index] = "E-yb1";
		index++;
		variables_names[index] = "E-yb3";
		index++;
		variables_names[index] = "E-yc2";
		index++;
		for (size_t nn = 0; nn < NUM_COP; nn++)
		{
			variables_names[index] = "E-yzi[" + std::to_string(nn) + "]";
			index++;
		}
		Onbvar += 2 * Emax;
		for (size_t mm = 0; mm < Emax; mm++)
		{
			variables_names[index] = "E-Fw-prev-y[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-Fw-pstv-y[" + std::to_string(mm) + "]";
			index++;
		}
#endif // end DIM_Y
#if DIM_Z
		Onbvar += (3 + NUM_COP);
		variables_names[index] = "E-zb1";
		index++;
		variables_names[index] = "E-zb3";
		index++;
		variables_names[index] = "E-zc2";
		index++;
		for (size_t nn = 0; nn < NUM_COP; nn++)
		{
			variables_names[index] = "E-zzi[" + std::to_string(nn) + "]";
			index++;
		}
		Onbvar += 2 * Emax;
		for (size_t mm = 0; mm < Emax; mm++)
		{
			variables_names[index] = "E-Fw-prev-z[" + std::to_string(mm) + "]";
			index++;
			variables_names[index] = "E-Fw-pstv-z[" + std::to_string(mm) + "]";
			index++;
		}
#endif // end DIM_Z
		Onbvar += 3 * Emax;
		for (size_t u = 0; u < Emax; u++)
		{
			variables_names[index] = "E-U[" + std::to_string(u) + "]";
			index++;
			variables_names[index] = "E-U1[" + std::to_string(u) + "]";
			index++;
			variables_names[index] = "E-LU[" + std::to_string(u) + "]";
			index++;
		}
	}
#endif // ESTIM_NAN
	variables_names[index] = "OV-c";
	index++;
	variables_names[index] = "O-rho";
	index++;
	variables_names[index] = "O-p";
	index++;
	variables_names[index] = "O-gamma";
	index++;
	variables_names[index] = "O-T";
	index++;
	variables_names[index] = "O-e";
	index++;
	Onbvar += 4;
	variables_names[index] = "O-vorticity";
	index++;
	variables_names[index] = "O-vorticity_x";
	index++;
	variables_names[index] = "O-vorticity_y";
	index++;
	variables_names[index] = "O-vorticity_z";
	index++;
#ifdef COP
	for (size_t ii = Onbvar - NUM_SPECIES; ii < Onbvar; ii++)
		variables_names[ii] = "Y" + std::to_string(ii - Onbvar + NUM_SPECIES) + "(" + Ss.species_name[ii - Onbvar + NUM_SPECIES] + ")";
#endif // COP

	int xmin = 0, ymin = 0, xmax = 0, ymax = 0, zmin = 0, zmax = 0, mx = 0, my = 0, mz = 0;
	real_t dx = 0.0, dy = 0.0, dz = 0.0;
#if DIM_X
	xmin = Ss.BlSz.myMpiPos_x * VTI.nbX;
	xmax = Ss.BlSz.myMpiPos_x * VTI.nbX + VTI.nbX;
	dx = Ss.BlSz.dx;
#endif // DIM_X
#if DIM_Y
	ymin = Ss.BlSz.myMpiPos_y * VTI.nbY;
	ymax = Ss.BlSz.myMpiPos_y * VTI.nbY + VTI.nbY;
	dy = Ss.BlSz.dy;
#endif // DIM_Y
#if DIM_Z
	zmin = (Ss.BlSz.myMpiPos_z * VTI.nbZ);
	zmax = (Ss.BlSz.myMpiPos_z * VTI.nbZ + VTI.nbZ);
	dz = Ss.BlSz.dz;
#endif // DIM_Z

	std::string file_name, outputPrefix = INI_SAMPLE;
	std::string temp_name = "./VTI_" + outputPrefix + "_Step_Time_" + stepFormat.str() + "." + timeFormat.str();
	file_name = Ss.OutputDir + "/" + temp_name;
#ifdef USE_MPI
	file_name = file_name + "_rank_" + rankFormat.str();
	std::string headerfile_name = Ss.OutputDir + "/VTI_" + outputPrefix + "_Step_" + stepFormat.str() + ".pvti";
#endif
	file_name += ".vti";

#ifdef USE_MPI
	if (!error)
	{
		mx = Ss.BlSz.mx;
		my = Ss.BlSz.my;
		mz = (3 == DIM_X + DIM_Y + DIM_Z) ? Ss.BlSz.mz : 0;
		if (0 == rank) // write header
		{
			std::fstream outHeader;
			// dummy string here, when using the full VTK API, data can be compressed
			// here, no compression used
			std::string compressor("");
			// open pvti header file
			outHeader.open(headerfile_name.c_str(), std::ios_base::out);
			outHeader << "<?xml version=\"1.0\"?>" << std::endl;
			if (isBigEndian())
				outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"BigEndian\"" << compressor << ">" << std::endl;
			else
				outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
			outHeader << "  <PImageData WholeExtent=\"";
			outHeader << 0 << " " << mx * VTI.nbX << " ";
			outHeader << 0 << " " << my * VTI.nbY << " ";
			outHeader << 0 << " " << mz * VTI.nbZ << "\" GhostLevel=\"0\" "
					  << "Origin=\""
					  << Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
					  << "Spacing=\""
					  << dx << " " << dy << " " << dz << "\">"
					  << std::endl;
			outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;
			for (int iVar = 0; iVar < Onbvar; iVar++)
			{
#if USE_DOUBLE
			outHeader << "      <PDataArray type=\"Float64\" Name=\"" << variables_names.at(iVar) << "\"/>" << std::endl;
#else
			outHeader << "      <PDataArray type=\"Float32\" Name=\"" << variables_names.at(iVar) << "\"/>" << std::endl;
#endif // end USE_DOUBLE
			}
		outHeader << "    </PCellData>" << std::endl;
		// Out put for 2D && 3D;
		for (int iPiece = 0; iPiece < Ss.mpiTrans->nProcs; ++iPiece)
		{
			std::ostringstream pieceFormat;
			pieceFormat.width(5);
			pieceFormat.fill('0');
			pieceFormat << iPiece;
			std::string pieceFilename = temp_name + "_rank_" + pieceFormat.str() + ".vti";
// get MPI coords corresponding to MPI rank iPiece
#if 3 == DIM_X + DIM_Y + DIM_Z
			int coords[3];
#else
			int coords[2];
#endif
			Ss.mpiTrans->communicator->getCoords(iPiece, DIM_X + DIM_Y + DIM_Z, coords);
			outHeader << " <Piece Extent=\"";
			// pieces in first line of column are different (due to the special
			// pvti file format with overlapping by 1 cell)
			if (coords[0] == 0)
				outHeader << 0 << " " << VTI.nbX << " ";
			else
				outHeader << coords[0] * VTI.nbX << " " << coords[0] * VTI.nbX + VTI.nbX << " ";

			if (coords[1] == 0)
				outHeader << 0 << " " << VTI.nbY << " ";
			else
				outHeader << coords[1] * VTI.nbY << " " << coords[1] * VTI.nbY + VTI.nbY << " ";
#if 3 == DIM_X + DIM_Y + DIM_Z
			if (coords[2] == 0)
				outHeader << 0 << " " << VTI.nbZ << " ";
			else
				outHeader << coords[2] * VTI.nbZ << " " << coords[2] * VTI.nbZ + VTI.nbZ << " ";
#else
			outHeader << 0 << " " << 0;
#endif
			outHeader << "\" Source=\"";
			outHeader << pieceFilename << "\"/>" << std::endl;
		}
		outHeader << "</PImageData>" << std::endl;
		outHeader << "</VTKFile>" << std::endl;
		// close header file
		outHeader.close();
		} // end writing pvti header
	}
#endif

	std::fstream outFile;
	outFile.open(file_name.c_str(), std::ios_base::out);
	// write xml data header
	if (isBigEndian())
		outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
	else
		outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

	outFile << "  <ImageData WholeExtent=\""
			<< xmin << " " << xmax << " "
			<< ymin << " " << ymax << " "
			<< zmin << " " << zmax << "\" "
			<< "Origin=\""
			<< Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
			<< "Spacing=\""
			<< dx << " " << dy << " " << dz << "\">" << std::endl;
	outFile << "  <Piece Extent=\""
			<< xmin << " " << xmax << " "
			<< ymin << " " << ymax << " "
			<< zmin << " " << zmax << ""
			<< "\">" << std::endl;
	outFile << "    <PointData>\n";
	outFile << "    </PointData>\n";
	// write data in binary format
	outFile << "    <CellData>" << std::endl;
	for (int iVar = 0; iVar < Onbvar; iVar++)
	{
#if USE_DOUBLE
		outFile << "     <DataArray type=\"Float64\" Name=\"";
#else
		outFile << "     <DataArray type=\"Float32\" Name=\"";
#endif // end USE_DOUBLE
		outFile << variables_names.at(iVar)
				<< "\" format=\"appended\" offset=\""
				<< iVar * VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(real_t) + iVar * sizeof(unsigned int)
				<< "\" />" << std::endl;
	}
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
	// write the leading undescore
	outFile << "_";
	// then write heavy data (column major format)
	unsigned int nbOfWords = VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(real_t);
	{
		//[0]x
#if DIM_X
		MARCO_OUTLOOP
		{
		real_t tmp = (DIM_X) ? (i - Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner) + _DF(0.5)) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin : 0.0;
		outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		  //[1]u
		MARCO_OUTLOOP
		{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.u[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.u[id] : fluids[1]->h_fstate.u[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		}		  // for i	  // for j		  // for k
#endif			  // end DIM_X
#if DIM_Y
		//[2]y
		MARCO_OUTLOOP
		{
					real_t tmp = (DIM_Y) ? (j - Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner) + _DF(0.5)) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		  //[3]v
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.v[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.v[id] : fluids[1]->h_fstate.v[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		}		  // for i
#endif			  // end DIM_Y
#if DIM_Z
		//[4]z
		MARCO_OUTLOOP
		{
					real_t tmp = (DIM_Z) ? (i - Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner) + _DF(0.5)) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		  //[5]w
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.w[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.w[id] : fluids[1]->h_fstate.w[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		}		  // for i
#endif			  // end DIM_Z

#ifdef ESTIM_NAN
		if (error)
		{
#ifdef Visc // Out content of viscous out estimating Vars
					for (size_t mm = 0; mm < Emax; mm++)
					{
#if DIM_X
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.visFwx[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
#if DIM_Y
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.visFwy[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
#if DIM_Z
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.visFwz[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
					}

#ifdef Visc_Diffu
					for (size_t mm = 0; mm < NUM_SPECIES; mm++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Dkm_aver[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Ertemp1[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Ertemp2[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}

#if DIM_X
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Dim_wallx[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.hi_wallx[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yi_wallx[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yil_wallx[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
#if DIM_Y
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Dim_wally[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.hi_wally[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yi_wally[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yil_wally[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
#if DIM_Z
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Dim_wallz[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.hi_wallz[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yi_wallz[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.Yil_wallz[mm + NUM_SPECIES * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
#endif
					}
#endif
#endif
#if DIM_X
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b1x[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}

					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b3x[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}

					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.c2x[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}

					for (size_t nn = 0; nn < NUM_COP; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.zix[nn + NUM_COP * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}

					for (size_t nn = 0; nn < Emax; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.preFwx[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.pstFwx[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}
#endif // end DIM_X

#if DIM_Y
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b1y[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b3y[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.c2y[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					for (size_t nn = 0; nn < NUM_COP; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.ziy[nn + NUM_COP * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}

					for (size_t nn = 0; nn < Emax; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.preFwy[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.pstFwy[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}
#endif // end DIM_Y
#if DIM_Z
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b1z[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.b3z[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.c2z[id];
			outFile.write((char *)&tmp, sizeof(real_t));
					}
					for (size_t nn = 0; nn < NUM_COP; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.ziz[nn + NUM_COP * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}

					for (size_t nn = 0; nn < Emax; nn++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.preFwz[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_fstate.pstFwz[nn + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}
#endif // end DIM_Z

					for (size_t u = 0; u < Emax; u++)
					{
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_U[u + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_U1[u + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
			MARCO_OUTLOOP
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				real_t tmp = fluids[0]->h_LU[u + Emax * id];
				outFile.write((char *)&tmp, sizeof(real_t));
			}
					}
		}
#endif // end ESTIM_NAN

		//[6]V-c
		MARCO_OUTLOOP
		{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
			real_t tmp = fluids[0]->h_fstate.c[id];
#elif 2 == NumFluid
			real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.c[id] : fluids[1]->h_fstate.c[id];
#endif
			outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		//[6]rho
		MARCO_OUTLOOP
		{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.rho[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.rho[id] : fluids[1]->h_fstate.rho[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		//[7]P
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.p[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.p[id] : fluids[1]->h_fstate.p[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		  //[8]Gamma
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.gamma[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.gamma[id] : fluids[1]->h_fstate.gamma[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		//[9]T
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.T[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.T[id] : fluids[1]->h_fstate.T[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		//[10]e
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.e[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.e[id] : fluids[1]->h_fstate.e[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i

		  //[11]vorticity
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					real_t tmp = sqrt(fluids[0]->h_fstate.vx[id]);
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					real_t tmp = fluids[0]->h_fstate.vxs[0][id];
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for i
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					real_t tmp = fluids[0]->h_fstate.vxs[1][id];
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for j
		MARCO_OUTLOOP
		{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					real_t tmp = fluids[0]->h_fstate.vxs[2][id];
					outFile.write((char *)&tmp, sizeof(real_t));
		} // for k

#ifdef COP
		//[COP]yii
		for (int ii = 0; ii < NUM_SPECIES; ii++)
		{
					MARCO_OUTLOOP
					{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.y[ii + NUM_SPECIES * id]; // h_fstate.y[ii][id];
			outFile.write((char *)&tmp, sizeof(real_t));
					} // for i
		}			  // for yii
#endif				  // end  COP
	}				  // End Var Output
	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	outFile.close();
}

void LAMNSS::Output_plt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error)
{
	std::string outputPrefix = INI_SAMPLE;
	std::string file_name = Ss.OutputDir + "/PLT_" + outputPrefix + "_Step_Time_" + stepFormat.str() + "." + timeFormat.str();
#ifdef USE_MPI
	file_name += "_rank_" + rankFormat.str();
#endif
	file_name += ".plt";

	// Init var names
	int Onbvar = 5 + (DIM_X + DIM_Y + DIM_Z) * 2; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES;
#endif // end COP

	std::map<int, std::string> variables_names;
	int index = 0;
#if DIM_X
	variables_names[index] = "x[m]";
	index++;
#endif // end DIM_X
#if DIM_Y
	variables_names[index] = "y[m]";
	index++;
#endif // end DIM_Y
#if DIM_Z
	variables_names[index] = "z[m]";
	index++;
#endif // end DIM_Z
#if DIM_X
	variables_names[index] = "<i>u</i>[m/s]";
	index++;
#endif // end DIM_X
#if DIM_Y
	variables_names[index] = "<i>v</i>[m/s]";
	index++;
#endif // end DIM_Y
#if DIM_Z
	variables_names[index] = "<i>w</i>[m/s]";
	index++;
#endif // end DIM_Z
	variables_names[index] = "<i><greek>r</greek></i>[kg/m<sup>3</sup>]"; // rho
	index++;
	variables_names[index] = "<i>p</i>[Pa]"; // pressure
	index++;
	variables_names[index] = "<i>T</i>[K]"; // temperature
	index++;
	variables_names[index] = "<i>c</i>[m/s]"; // sound speed
	index++;
	variables_names[index] = "<i><greek>g</greek></i>[-]"; // gamma
	index++;
#ifdef COP
	for (size_t ii = Onbvar - NUM_SPECIES; ii < Onbvar; ii++)
		variables_names[ii] = "<i>Y" + std::to_string(ii - Onbvar + NUM_SPECIES) + "(" + Ss.species_name[ii - Onbvar + NUM_SPECIES] + ")</i>[-]";
#endif // COP

	real_t posx = -Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner);
	real_t posy = -Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner);
	real_t posz = -Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner);

	std::ofstream out(file_name);
	// defining header for tecplot(plot software)
	out << "title='" << outputPrefix << "'\nvariables=";
	for (int iVar = 0; iVar < Onbvar - 1; iVar++)
		out << " " << variables_names.at(iVar) << ", ";
	out << variables_names.at(Onbvar - 1) << "\n";
	out << "zone t='" << outputPrefix << "_" << timeFormat.str();
#ifdef USE_MPI
	out << "_rank_" << std::to_string(rank);
#endif // end USE_MPI
	out << "', i= " << VTI.nbX + DIM_X << ", j= " << VTI.nbY + DIM_Y << ", k= " << VTI.nbZ + DIM_Z << "  DATAPACKING=BLOCK, VARLOCATION=(["
		<< DIM_X + DIM_Y + DIM_Z + 1 << "-" << Onbvar << "]=CELLCENTERED) SOLUTIONTIME= " << timeFormat.str() << "\n";

	real_t dimx = DIM_X, dimy = DIM_Y, dimz = DIM_Z;
#if DIM_X
	for (int k = VTI.minZ; k < VTI.maxZ + DIM_Z; k++)
		for (int j = VTI.minY; j < VTI.maxY + DIM_Y; j++)
		{
					for (int i = VTI.minX; i < VTI.maxX + DIM_X; i++)
			out << dimx * ((i + posx) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin) << " ";
					out << "\n";
		}
#endif
#if DIM_Y
	for (int k = VTI.minZ; k < VTI.maxZ + DIM_Z; k++)
		for (int j = VTI.minY; j < VTI.maxY + DIM_Y; j++)
		{
					for (int i = VTI.minX; i < VTI.maxX + DIM_X; i++)
			out << dimy * ((j + posy) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin) << " ";
					out << "\n";
		}
#endif
#if DIM_Z
	for (int k = VTI.minZ; k < VTI.maxZ + DIM_Z; k++)
		for (int j = VTI.minY; j < VTI.maxY + DIM_Y; j++)
		{
					for (int i = VTI.minX; i < VTI.maxX + DIM_X; i++)
			out << dimz * ((k + posz) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin) << " ";
					out << "\n";
		}
#endif
#if DIM_X
	MARCO_POUTLOOP(fluids[0]->h_fstate.u[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
#endif
#if DIM_Y
	MARCO_POUTLOOP(fluids[0]->h_fstate.v[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
#endif
#if DIM_Z
	MARCO_POUTLOOP(fluids[0]->h_fstate.w[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
#endif
	MARCO_POUTLOOP(fluids[0]->h_fstate.rho[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.p[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.T[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.c[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.gamma[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);

#ifdef COP
	for (size_t n = 0; n < NUM_SPECIES; n++)
		MARCO_POUTLOOP(fluids[0]->h_fstate.y[n + NUM_SPECIES * (Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i)]);
#endif
	out.close();
}

void LAMNSS::GetCPT_OutRanks(int *OutRanks, int rank, int nranks)
{ // compressible out: output dirs less than caculate dirs

	bool Out1, Out2, Out3;
	int if_outrank = -1;
	real_t temx = _DF(0.5) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin;
	real_t temy = _DF(0.5) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin;
	real_t temz = _DF(0.5) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin;
	real_t posx = -Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner);
	real_t posy = -Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner);
	real_t posz = -Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner);

	////method 1/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (int k = VTI.minZ; k < VTI.maxZ; k++)
		for (int j = VTI.minY; j < VTI.maxY; j++)
					for (int i = VTI.minX; i < VTI.maxX; i++)
					{ //&& Ss.OutDIRX//&& Ss.OutDIRY//&& Ss.OutDIRZ
			int pos_x = i + posx, pos_y = j + posy, pos_z = k + posz;
			Out1 = ((!Ss.OutDIRX) && (pos_x == Ss.outpos_x)); // fabs(OutPoint[0] - Ss.outpos_x) < temx
			Out2 = ((!Ss.OutDIRY) && (pos_y == Ss.outpos_y)); // fabs(OutPoint[1] - Ss.outpos_y) < temy
			Out3 = ((!Ss.OutDIRZ) && (pos_z == Ss.outpos_z)); // fabs(OutPoint[2] - Ss.outpos_z) < temz
			if (Out1 || Out2 || Out3)
			{
				if_outrank = rank;
			}
					}

#ifdef USE_MPI
	Ss.mpiTrans->communicator->allGather(&(if_outrank), 1, mpiUtils::MpiComm::INT, OutRanks, 1, mpiUtils::MpiComm::INT);
	// std::cout << "OutRanks[rank], recvcounts[rank], displs[rank]= " << OutRanks[rank] << " of rank: " << rank << std::endl; //<< ", " << recvcounts[rank] << ", " << displs[rank]
#endif
}

// Need DIM_X+DIM_Y+DIM_Z > 1
void LAMNSS::Output_cvti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat)
{
#if DIM_X + DIM_Y + DIM_Z > 1
	// Init var names
	int Onbvar = 6 + (DIM_X + DIM_Y + DIM_Z) * 2; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES;
#endif // end COP
	std::map<int, std::string> variables_names;
	int index = 0;
#if DIM_X
	variables_names[index] = "DIR-X";
	index++;
	variables_names[index] = "OV-u";
	index++;
#endif // end DIM_X
#if DIM_Y
	variables_names[index] = "DIR-Y";
	index++;
	variables_names[index] = "OV-v";
	index++;
#endif // end DIM_Y
#if DIM_Z
	variables_names[index] = "DIR-Z";
	index++;
	variables_names[index] = "OV-w";
	index++;
#endif // end DIM_Z
	variables_names[index] = "OV-c";
	index++;
	variables_names[index] = "O-rho";
	index++;
	variables_names[index] = "O-p";
	index++;
	variables_names[index] = "O-gamma";
	index++;
	variables_names[index] = "O-T";
	index++;
	variables_names[index] = "O-e";
	index++;
	Onbvar += 4;
	variables_names[index] = "O-vorticity";
	index++;
	variables_names[index] = "O-vorticity_x";
	index++;
	variables_names[index] = "O-vorticity_y";
	index++;
	variables_names[index] = "O-vorticity_z";
	index++;
#ifdef COP
	for (size_t ii = Onbvar - NUM_SPECIES; ii < Onbvar; ii++)
		variables_names[ii] = "Y" + std::to_string(ii - Onbvar + NUM_SPECIES) + "(" + Ss.species_name[ii - Onbvar + NUM_SPECIES] + ")";
#endif // COP

	int *OutRanks = new int[nranks]{0};
#ifdef USE_MPI
	GetCPT_OutRanks(OutRanks, rank, nranks);
#endif // end USE_MPI

	int xmin = 0, ymin = 0, xmax = 0, ymax = 0, zmin = 0, zmax = 0, mx = 0, my = 0, mz = 0;
	real_t dx = 0.0, dy = 0.0, dz = 0.0;
#if DIM_X
	if (Ss.OutDIRX)
	{
		xmin = Ss.BlSz.myMpiPos_x * VTI.nbX;
		xmax = Ss.BlSz.myMpiPos_x * VTI.nbX + VTI.nbX;
		dx = Ss.BlSz.dx;
	}
#endif // DIM_X
#if DIM_Y
	if (Ss.OutDIRY)
	{
		ymin = Ss.BlSz.myMpiPos_y * VTI.nbY;
		ymax = Ss.BlSz.myMpiPos_y * VTI.nbY + VTI.nbY;
		dy = Ss.BlSz.dy;
	}
#endif // DIM_Y
#if DIM_Z
	if (Ss.OutDIRZ)
	{
		zmin = (Ss.BlSz.myMpiPos_z * VTI.nbZ);
		zmax = (Ss.BlSz.myMpiPos_z * VTI.nbZ + VTI.nbZ);
		dz = Ss.BlSz.dz;
	}
#endif // DIM_Z

	std::string file_name, outputPrefix = INI_SAMPLE;
	std::string temp_name = "./CVTI_" + outputPrefix + "_Step_Time_" + stepFormat.str() + "." + timeFormat.str();
	file_name = Ss.OutputDir + "/" + temp_name;
#ifdef USE_MPI
	file_name = file_name + "_rank_" + rankFormat.str();
	std::string headerfile_name = Ss.OutputDir + "/CVTI_" + outputPrefix + "_Step_" + stepFormat.str() + ".pvti";
#endif
	file_name += ".vti";

#ifdef USE_MPI
	mx = (Ss.OutDIRX) ? Ss.BlSz.mx : 0;
	my = (Ss.OutDIRY) ? Ss.BlSz.my : 0;
	mz = (Ss.OutDIRZ) ? Ss.BlSz.mz : 0;
	if (0 == rank) // write header
	{
		std::fstream outHeader;
		std::string compressor("");
		// open pvti header file
		outHeader.open(headerfile_name.c_str(), std::ios_base::out);
		outHeader << "<?xml version=\"1.0\"?>" << std::endl;
		if (isBigEndian())
			outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"BigEndian\"" << compressor << ">" << std::endl;
		else
			outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
		outHeader << "  <PImageData WholeExtent=\"";
		outHeader << 0 << " " << mx * VTI.nbX << " ";
		outHeader << 0 << " " << my * VTI.nbY << " ";
		outHeader << 0 << " " << mz * VTI.nbZ << "\" GhostLevel=\"0\" "
				  << "Origin=\""
				  << Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
				  << "Spacing=\""
				  << dx << " " << dy << " " << dz << "\">"
				  << std::endl;
		outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;
		for (int iVar = 0; iVar < Onbvar; iVar++)
		{
#if USE_DOUBLE
			outHeader << "      <PDataArray type=\"Float64\" Name=\"" << variables_names.at(iVar) << "\"/>" << std::endl;
#else
			outHeader << "      <PDataArray type=\"Float32\" Name=\"" << variables_names.at(iVar) << "\"/>" << std::endl;
#endif // end USE_DOUBLE
		}
		outHeader << "    </PCellData>" << std::endl;

		// Out put for 2D && 3D;
		for (int iPiece = 0; iPiece < Ss.mpiTrans->nProcs; ++iPiece)
			if (OutRanks[iPiece] >= 0)
			{
				std::ostringstream pieceFormat;
				pieceFormat.width(5);
				pieceFormat.fill('0');
				pieceFormat << iPiece;
				std::string pieceFilename = temp_name + "_rank_" + pieceFormat.str() + ".vti";
				// get MPI coords corresponding to MPI rank iPiece
				// #if 3 == DIM_X + DIM_Y + DIM_Z
				// 			int coords[3];
				// #else
				// 			int coords[2];
				// #endif
				int coords[3], OnbX = (Ss.OutDIRX) ? VTI.nbX : 0, OnbY = (Ss.OutDIRY) ? VTI.nbY : 0, OnbZ = (Ss.OutDIRZ) ? VTI.nbZ : 0;
				Ss.mpiTrans->communicator->getCoords(iPiece, DIM_X + DIM_Y + DIM_Z, coords);
				outHeader << " <Piece Extent=\"";
				// pieces in first line of column are different (due to the special
				// pvti file format with overlapping by 1 cell)
#if DIM_X
				if (coords[0] == 0)
					outHeader << 0 << " " << OnbX << " ";
				else
					outHeader << coords[0] * OnbX << " " << coords[0] * OnbX + OnbX << " ";
#else
				outHeader << 0 << " " << 0;
#endif // end DIM_X
#if DIM_Y
				if (coords[1] == 0)
					outHeader << 0 << " " << OnbY << " ";
				else
					outHeader << coords[1] * OnbY << " " << coords[1] * OnbY + OnbY << " ";
#else
				outHeader << 0 << " " << 0;
#endif // end DIM_Y
#if DIM_Z
				if (coords[2] == 0)
					outHeader << 0 << " " << OnbZ << " ";
			else
					outHeader << coords[2] * OnbZ << " " << coords[2] * OnbZ + OnbZ << " ";
#else
			outHeader << 0 << " " << 0;
#endif // end DIM_Z
			outHeader << "\" Source=\"";
			outHeader << pieceFilename << "\"/>" << std::endl;
		}
		outHeader << "</PImageData>" << std::endl;
		outHeader << "</VTKFile>" << std::endl;
		// close header file
		outHeader.close();
	} // end writing pvti header
#endif
	int minX = CPT.minX, minY = CPT.minY, minZ = CPT.minZ;
	int nbX = (Ss.OutDIRX) ? VTI.nbX : 1, nbY = (Ss.OutDIRY) ? VTI.nbY : 1, nbZ = (Ss.OutDIRZ) ? VTI.nbZ : 1;
	int maxX = minX + nbX, maxY = minY + nbY, maxZ = minZ + nbZ;

	if (OutRanks[rank] >= 0)
	{
		std::fstream outFile;
		outFile.open(file_name.c_str(), std::ios_base::out);
		// write xml data header
		if (isBigEndian())
			outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
		else
			outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

		outFile << "  <ImageData WholeExtent=\""
				<< xmin << " " << xmax << " "
				<< ymin << " " << ymax << " "
				<< zmin << " " << zmax << "\" "
				<< "Origin=\""
				<< Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
				<< "Spacing=\""
				<< dx << " " << dy << " " << dz << "\">" << std::endl;
		outFile << "  <Piece Extent=\""
				<< xmin << " " << xmax << " "
				<< ymin << " " << ymax << " "
				<< zmin << " " << zmax << ""
				<< "\">" << std::endl;
		outFile << "    <PointData>\n";
		outFile << "    </PointData>\n";
		// write data in binary format
		outFile << "    <CellData>" << std::endl;
		for (int iVar = 0; iVar < Onbvar; iVar++)
		{
#if USE_DOUBLE
		outFile << "     <DataArray type=\"Float64\" Name=\"";
#else
		outFile << "     <DataArray type=\"Float32\" Name=\"";
#endif // end USE_DOUBLE
		outFile << variables_names.at(iVar)
				<< "\" format=\"appended\" offset=\""
				<< iVar * nbX * nbY * nbZ * sizeof(real_t) + iVar * sizeof(unsigned int)
				<< "\" />" << std::endl;
		}
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
	// write the leading undescore
	outFile << "_";
	// then write heavy data (column major format)
	unsigned int nbOfWords = nbX * nbY * nbZ * sizeof(real_t);
#if DIM_X
	//[0]x
	MARCO_COUTLOOP
	{
		real_t tmp = (DIM_X) ? (i - Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner) + _DF(0.5)) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin : 0.0;
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[1]u
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.u[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
#endif
#if DIM_Y
	//[2]y
	MARCO_COUTLOOP
	{
		real_t tmp = (DIM_Y) ? (j - Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner) + _DF(0.5)) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin : 0.0;
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[3]v
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.v[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
#endif
#if DIM_Z
	//[4]z
	MARCO_COUTLOOP
	{
		real_t tmp = (DIM_Z) ? (i - Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner) + _DF(0.5)) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin : 0.0;
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[5]w
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.w[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
#endif
	//[6]V-c
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.c[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[7]rho
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.rho[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[8]P
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.p[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[9]Gamma
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.gamma[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[10]T
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.T[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[11]e
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.e[id];
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	//[12]vorticity
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = sqrt(fluids[0]->h_fstate.vx[id]);
		outFile.write((char *)&tmp, sizeof(real_t));
	}
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.vxs[0][id];
		outFile.write((char *)&tmp, sizeof(real_t));
	} // for i
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.vxs[1][id];
		outFile.write((char *)&tmp, sizeof(real_t));
	} // for j
	MARCO_COUTLOOP
	{
		int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
		real_t tmp = fluids[0]->h_fstate.vxs[2][id];
		outFile.write((char *)&tmp, sizeof(real_t));
	} // for k
#ifdef COP
	//[COP]yii
	for (int ii = 0; ii < NUM_SPECIES; ii++)
	{
		MARCO_COUTLOOP
		{
			int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
			real_t tmp = fluids[0]->h_fstate.y[ii + NUM_SPECIES * id]; // h_fstate.y[ii][id];
			outFile.write((char *)&tmp, sizeof(real_t));
			}
	}
#endif // end  COP
	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	outFile.close();
	}
#endif // end DIM_X+DIM_Y+DIM_Z > 1
}

void LAMNSS::Output_cplt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat)
{ // compressible out: output dirs less than caculate dirs
	int Cnbvar = 16;
#ifdef COP
	Cnbvar += NUM_SPECIES;
#endif

	int *OutRanks = new int[nranks];
	real_t temx = _DF(0.5) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin;
	real_t temy = _DF(0.5) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin;
	real_t temz = _DF(0.5) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin;
	real_t posx = -Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner);
	real_t posy = -Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner);
	real_t posz = -Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner);

	////method 1/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_MPI
	GetCPT_OutRanks(OutRanks, rank, nranks);
	if (OutRanks[rank] >= 0)
#endif // end USE_MPI
	{
		real_t *OutPoint = new real_t[Cnbvar]; // OutPoint: each point;
		std::string outputPrefix = INI_SAMPLE;
		std::string file_name = Ss.OutputDir + "/CPLT_" + outputPrefix + "_Step_Time_" + stepFormat.str() + "." + timeFormat.str() + "_" + rankFormat.str() + ".plt";
		std::ofstream out(file_name);
		// // defining header for tecplot(plot software)
		out << "title='" << outputPrefix << "'\nvariables=";
#if DIM_X
		out << "x[m], ";
#endif // end DIM_X
#if DIM_Y
		out << "y[m], ";
#endif // end DIM_Y
#if DIM_Z
		out << "z[m], ";
#endif // end DIM_Z
		out << "<i><greek>r</greek></i>[kg/m<sup>3</sup>], <i>p</i>[Pa], <i>c</i>[m/s]";
#if DIM_X
		out << ", <i>u</i>[m/s]";
#endif // end DIM_X
#if DIM_Y
		out << ", <i>v</i>[m/s]";
#endif // end DIM_Y
#if DIM_Z
		out << ", <i>w</i>[m/s]";
#endif // end DIM_Z
		out << ", <i><greek>g</greek></i>[-], <i>T</i>[K], <i>e</i>[J], |<i><greek>w</greek></i>|[s<sup>-1</sup>]";
		out << ", <i><greek>w</greek></i><sub>x</sub>[s<sup>-1</sup>], <i><greek>w</greek></i><sub>y</sub>[s<sup>-1</sup>], <i><greek>w</greek></i><sub>z</sub>[s<sup>-1</sup>]";
#ifdef COP
		for (size_t n = 0; n < NUM_SPECIES; n++)
			out << ", <i>Y(" << Ss.species_name[n] << ")</i>[-]";
#endif
		out << "\n";
		out << "zone t='" << outputPrefix << "_" << timeFormat.str();
#ifdef USE_MPI
		out << "_rank_" << std::to_string(rank);
#endif // end USE_MPI
		out << "', i= " << CPT.nbX << ", j= " << CPT.nbY << ", k= " << CPT.nbZ << ", SOLUTIONTIME= " << timeFormat.str() << "\n";

		for (int k = CPT.minZ; k < CPT.maxZ; k++)
			for (int j = CPT.minY; j < CPT.maxY; j++)
				for (int i = CPT.minX; i < CPT.maxX; i++)
				{ //&& Ss.OutDIRX//&& Ss.OutDIRY//&& Ss.OutDIRZ
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					int pos_x = i + posx, pos_y = j + posy, pos_z = k + posz;
					OutPoint[0] = (DIM_X) ? (pos_x)*Ss.BlSz.dx + temx : 0.0;
					OutPoint[1] = (DIM_Y) ? (pos_y)*Ss.BlSz.dy + temy : 0.0;
					OutPoint[2] = (DIM_Z) ? (pos_z)*Ss.BlSz.dz + temz : 0.0;
					OutPoint[3] = fluids[0]->h_fstate.rho[id];
					OutPoint[4] = fluids[0]->h_fstate.p[id];
					OutPoint[5] = fluids[0]->h_fstate.c[id];
					OutPoint[6] = fluids[0]->h_fstate.u[id];
					OutPoint[7] = fluids[0]->h_fstate.v[id];
					OutPoint[8] = fluids[0]->h_fstate.w[id];
					OutPoint[9] = fluids[0]->h_fstate.gamma[id];
					OutPoint[10] = fluids[0]->h_fstate.T[id];
					OutPoint[11] = fluids[0]->h_fstate.e[id];
					OutPoint[12] = sqrt(fluids[0]->h_fstate.vx[id]);
					OutPoint[13] = fluids[0]->h_fstate.vxs[0][id];
					OutPoint[14] = fluids[0]->h_fstate.vxs[1][id];
					OutPoint[15] = fluids[0]->h_fstate.vxs[2][id];
#if COP
					for (int n = 0; n < NUM_SPECIES; n++)
						OutPoint[Cnbvar - NUM_SPECIES + n] = fluids[0]->h_fstate.y[n + NUM_SPECIES * id];
#endif
#if DIM_X
					out << OutPoint[0] << " "; // x
#endif										   // end DIM_X
#if DIM_Y
					out << OutPoint[1] << " "; // y
#endif										   // end DIM_Y
#if DIM_Z
					out << OutPoint[2] << " "; // z
#endif										   // end DIM_Z
											   // rho, p, c
					out << OutPoint[3] << " " << OutPoint[4] << " " << OutPoint[5] << " ";
#if DIM_X
					out << OutPoint[6] << " ";
#endif // end DIM_X
#if DIM_Y
					out << OutPoint[7] << " ";
#endif // end DIM_Y
#if DIM_Z
					out << OutPoint[8] << " "; // u, v, w
#endif										   // end DIM_Z

					out << OutPoint[9] << " " << OutPoint[10] << " " << OutPoint[11] << " ";						 // gamma, T, e
					out << OutPoint[12] << " " << OutPoint[13] << " " << OutPoint[14] << " " << OutPoint[15] << " "; // Vorticity
#if COP
					for (int n = 0; n < NUM_SPECIES; n++)
						out << OutPoint[Cnbvar - NUM_SPECIES + n] << " "; // Yi
#endif
					out << "\n";
				}
		out.close();
		free(OutPoint);
	}
}
