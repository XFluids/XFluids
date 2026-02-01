//  C++ headers
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "timer/timer.h"
#include "global_class.h"

XFLUIDS::XFLUIDS(Setup &setup) : Ss(setup), dt(_DF(0.0)), Iteration(0), rank(0), nranks(1), physicalTime(0.0)
{
#if USE_MPI
	rank = Ss.mpiTrans->myRank;
	nranks = Ss.mpiTrans->nProcs;
#endif // end USE_MPI

	outputPrefix = "-" + std::string(SelectDv) + "-" + std::string(INI_SAMPLE);
	if (Ss.BlSz.DimZ)
		outputPrefix = "Z" + outputPrefix;
	if (Ss.BlSz.DimY)
		outputPrefix = "Y" + outputPrefix;
	if (Ss.BlSz.DimX)
		outputPrefix = "X" + outputPrefix;

	MPI_trans_time = 0.0, MPI_BCs_time = 0.0, duration = 0.0f, duration_backup = 0.0f;
	for (int n = 0; n < NumFluid; n++)
	{
		fluids[n] = new Fluid(setup);
		// if (1 < NumFluid)
		// 	fluids[n]->initialize(n);
	}

	if (OutBoundary)
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

	// #ifdef USE_MPI
	// 	if (Ss.BlSz.DimX)
	// 	{
	// 		if (Ss.mpiTrans->neighborsBC[XMIN] == BC_COPY)
	// 			if (Ss.OutDirX)
	// 				PLT.nbX += 1;
	// 		if (Ss.mpiTrans->neighborsBC[XMAX] == BC_COPY)
	// 			if (Ss.OutDirX)
	// 				PLT.nbX += 1;
	// 	}

	// 	if (Ss.BlSz.DimY)
	// 	{
	// 		if (Ss.mpiTrans->neighborsBC[YMIN] == BC_COPY)
	// 			if (Ss.OutDirY)
	// 				PLT.nbY += 1;
	// 		if (Ss.mpiTrans->neighborsBC[YMAX] == BC_COPY)
	// 			if (Ss.OutDirY)
	// 				PLT.nbY += 1;
	// 	}

	// 	if (Ss.BlSz.DimZ)
	// 	{
	// 		if (Ss.mpiTrans->neighborsBC[ZMIN] == BC_COPY)
	// 			if (Ss.OutDirZ)
	// 				PLT.nbZ += 1;
	// 		if (Ss.mpiTrans->neighborsBC[ZMAX] == BC_COPY)
	// 			if (Ss.OutDirZ)
	// 				PLT.nbZ += 1;
	// }
	// #endif // use MPI
}

XFLUIDS::~XFLUIDS()
{
	for (size_t n = 0; n < NumFluid; n++)
		fluids[n]->~Fluid();
}

void XFLUIDS::Evolution(sycl::queue &q)
{
	int OutNum = 1, TimeLoop = 0, error_out = 0, RcalOut = 0;
	bool TimeLoopOut = false, Stepstop = false, timer_create = true;
	// timer beginning point definition
	std::chrono::high_resolution_clock::time_point start_time;

	bool reAdv = false;
	std::vector<size_t> adv_size{0};
	Setup::adv_nd[0].resize(1), Setup::sbm_id = 0;
	// Setup::adv_nd.resize(2);
	std::string adv_name = OutputDir + "/" + outputPrefix;
	if (PositivityPreserving)
		adv_name += "-pp(on)";
	else
		adv_name += "-pp(off)";
	if (Visc)
		adv_name += "-vis(on)";
	else
		adv_name += "-vis(off)";
	if (ReactSources)
	{
		if (SlipOrder == std::string("Strang"))
			adv_name += "-rode(strang)";
		else
			adv_name += "-rode(lie)";
	}
	else
		adv_name += "-rode(off)";

	adv_name += "_AdaptiveRange_(";
	adv_name += std::to_string(Ss.BlSz.X_inner * int(Ss.BlSz.DimX)) + "+" + std::to_string(Ss.BlSz.Bwidth_X * int(Ss.BlSz.DimX)) + ")x(";
	adv_name += std::to_string(Ss.BlSz.Y_inner * int(Ss.BlSz.DimY)) + "+" + std::to_string(Ss.BlSz.Bwidth_Y * int(Ss.BlSz.DimY)) + ")x(";
	adv_name += std::to_string(Ss.BlSz.Z_inner * int(Ss.BlSz.DimZ)) + "+" + std::to_string(Ss.BlSz.Bwidth_Y * int(Ss.BlSz.DimZ)) + ")";
	if (0 == rank)
	{
		std::ifstream advIn(adv_name, std::ios_base::in | std::ios_base::binary);
		if (advIn.is_open())
		{
			Setup::adv_push = false, reAdv = true;
			for (size_t ii = 0; ii < OutAdvRange_json; ii++)
			{
				size_t szie = 0;
				advIn.read(reinterpret_cast<char *>(&szie), sizeof(size_t));
				Setup::adv_nd[ii].resize(szie / sizeof(Assign)), adv_size.push_back(szie);
				advIn.read(reinterpret_cast<char *>(Setup::adv_nd[ii].data()), szie);
			}
		}
		else
			std::cout << "AdaptiveRange-file not exist or open failed, AdaptiveRange recaluated." << std::endl;
		advIn.close();
	}
#ifdef USE_MPI
	reAdv = Ss.mpiTrans->BocastTrue(reAdv), Setup::adv_push = !reAdv;
	for (size_t ii = 0; ii < OutAdvRange_json; ii++)
	{
		size_t temp_size = Setup::adv_nd[ii].size();
		Ss.mpiTrans->communicator->bcast((char *)&temp_size, 4, mpiUtils::MpiComm::CHAR, 0), Setup::adv_nd[ii].resize(temp_size);
		Ss.mpiTrans->communicator->bcast(reinterpret_cast<char *>(Setup::adv_nd[ii].data()), Setup::adv_nd[ii].size() * sizeof(Assign), mpiUtils::MpiComm::CHAR, 0);
	}
#endif

	while (TimeLoop < Ss.OutTimeStamps.size())
	{
		real_t tbak = _DF(0.0);
		real_t target_t = (physicalTime < Ss.OutTimeStamps[TimeLoop].time) ? Ss.OutTimeStamps[TimeLoop].time : Ss.OutTimeStamps[TimeLoop++].time;
		OutAtThis = Ss.OutTimeStamps[std::max(0, TimeLoop - 1)];
		while (physicalTime < target_t)
		{
			CopyToUbak(q);
			if ((((Iteration % OutInterval == 0) || TimeLoopOut) && OutNum <= nOutput))
			{
				OutNum++;
				TimeLoopOut = false;
				// if (Iteration > 0)	// solution checkingpoint file output
				Output(q, OutAtThis.Reinitialize(physicalTime, std::to_string(Iteration)));
			}
			if ((RcalOut % RcalInterval == 0) && Iteration > Setup::adv_nd.size())
				Output_Ubak(rank, Iteration, physicalTime, duration);
			RcalOut++;
			Iteration++;

			if (timer_create)
			{ // // creat timer beginning point, execute only once.
				timer_create = false;
				if (0 == rank)
					std::cout << "Timer beginning at this point.\n";
				start_time = std::chrono::high_resolution_clock::now();
			}

			{ // // get minmum dt, if MPI used, get the minimum of all ranks
				dt = ComputeTimeStep(q);
				if (physicalTime + dt > target_t)
					dt = target_t - physicalTime;
				tbak = physicalTime;
				physicalTime += dt;
			}

			// // screen log print
			// // an iteration begins at physicalTime and ends at physicalTime + dt;
			if (rank == 0)
				std::cout << "N=" << std::setw(7) << Iteration << "  beginning physicalTime: " << std::setw(14) << std::setprecision(8) << tbak
						  << " dt: " << std::setw(14) << dt << "End physicalTime: " << std::setw(14) << std::setprecision(8) << physicalTime << "\n";

			{ // a advance time step
				// // strang slipping
				if ((ReactSources) && (SlipOrder == std::string("Strang")))
				{
					error_out = error_out || Reaction(q, dt, physicalTime, Iteration);
					if (0 == rank)
						std::cout << "<<<<<<<<<<<<<< " << SlipOrder << "Split First step has been done" << std::endl;
				}
				// // solved the fluid with 3rd order Runge-Kutta method
				error_out = error_out || SinglePhaseSolverRK3rd(q, rank, Iteration, physicalTime);
				// // reaction sources
				if (ReactSources)
				{
					error_out = error_out || Reaction(q, dt, physicalTime, Iteration);
					if (0 == rank)
						std::cout << "<<<<<<<<<<<<<< Reaction step has been done" << std::endl;
				}
			}

			{ // adaptive range assignmet step
				for (size_t ii = 0; ii < Setup::adv_nd[0].size(); ii++)
				{
					if (!Setup::adv_id && Setup::adv_push)
					{
						if (0 == rank)
						{
							std::ofstream advcsv(adv_name + ".csv", std::ios::out | std::ios::app);
							advcsv << ", ";
							for (size_t ii = 0; ii < Setup::adv_nd[Ss.adv_id].size() - 1; ii++)
								advcsv << std::string(Setup::adv_nd[Ss.adv_id][ii].tag) << ", ";
							advcsv << std::string(Setup::adv_nd[Ss.adv_id].back().tag) << std::endl;
						}
						break;
					}
					if (Setup::adv_nd[0][ii].time > Setup::adv_nd[Ss.adv_id][ii].time)
						Setup::adv_nd[0][ii] = Setup::adv_nd[Ss.adv_id][ii];
				}
				if (0 == rank)
					if ((!reAdv && Setup::adv_nd.size() == Iteration - 1) || Setup::adv_push || (1 == Iteration))
					{
						std::cout << "<<<<<<<<<<<<<< Adaptive Range Assignment(time: s): " << std::endl;
						for (size_t ii = 0; ii < Setup::adv_nd[Ss.adv_id].size(); ii++)
							std::cout << "    " << std::setw(30) << std::string(Setup::adv_nd[Ss.adv_id][ii].tag) << " submission, "
									  << Setup::adv_nd[Ss.adv_id][ii].local_nd[0] << ", "
									  << Setup::adv_nd[Ss.adv_id][ii].local_nd[1] << ", "
									  << Setup::adv_nd[Ss.adv_id][ii].local_nd[2] << ", "
									  << std::fixed << std::setprecision(10) << Setup::adv_nd[Ss.adv_id][ii].time
									  << std::defaultfloat << std::endl;
						std::ofstream advcsv(adv_name + ".csv", std::ios::out | std::ios::app);
						if (Setup::adv_push && advcsv.is_open())
						{
							advcsv << "(" << Setup::adv_nd[Ss.adv_id][0].local_nd[0] << " " << Setup::adv_nd[Ss.adv_id][0].local_nd[1] << " "
								   << Setup::adv_nd[Ss.adv_id][0].local_nd[2] << "), " << std::fixed << std::setprecision(10);
							for (size_t ii = 0; ii < Setup::adv_nd[Ss.adv_id].size() - 1; ii++)
								advcsv << Setup::adv_nd[Ss.adv_id][ii].time << ", ";
							advcsv << Setup::adv_nd[Ss.adv_id].back().time << std::defaultfloat << std::endl;
						}
						advcsv.close();
					}
				Setup::sbm_id = 0;
				Setup::adv_id = (Iteration < Setup::adv_nd.size() && Setup::adv_push) ? Iteration : 0;
				if (0 == rank && Setup::adv_nd.size() == Iteration && Setup::adv_push)
				{
					std::ofstream osData(adv_name, std::ios_base::out | std::ios_base::binary);
					for (size_t ii = 0; ii < OutAdvRange_json; ii++)
					{
						size_t ts_size = (Setup::adv_nd[ii].size() - 1) * sizeof(Assign);
						osData.write(reinterpret_cast<char *>(&ts_size), sizeof(size_t));
						osData.write(reinterpret_cast<char *>(&Setup::adv_nd[ii][1]), ts_size);
					}
					osData.close();
					Setup::adv_nd[0].erase(Setup::adv_nd[0].begin());
				}
				Setup::adv_push = Setup::adv_id;
			}

			// // if stop based error captured
			if (error_out)
				goto flag_ernd;

            { // // if stop based nStepmax
				Stepstop = Ss.nStepmax <= Iteration ? true : false;
				if (Stepstop)
					goto flag_end;
			}
		}
		TimeLoopOut = true;
	}

flag_end:
	if (ReactSources)
	{
		BoundaryCondition(q, 0);
		UpdateStates(q, 0, physicalTime, Iteration, "_End");
	}
flag_ernd:
#ifdef USE_MPI
	Ss.mpiTrans->communicator->synchronize();
#endif
	// // The last step Output,
	Output(q, Ss.OutTimeStamps.back().Reinitialize(physicalTime, std::to_string(Iteration)));
	EndProcess();
}

void XFLUIDS::EndProcess()
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
		std::cout << "<--------------------------------------------------->\n";
		std::cout << "MPI averaged of " << nranks << " ranks ";
///////////////////////////
#ifdef AWARE_MPI
		std::cout << "with    AWARE_MPI ";
#else
		std::cout << "without AWARE_MPI ";
#endif // end AWARE_MPI
///////////////////////////
#else
	std::cout << "<--------------------------------------------------->\n";
#endif // end USE_MPI
#ifdef USE_MPI
		std::cout << "Device Memory Usage(GB)   :  " << fluids[0]->MemMbSize / 1024.0 << std::endl;
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
		// std::cout << "Times of error patched: " << error_times_patched << std::endl;
        float runtime_total = 0.f;
		for (size_t i = 0; i < LU_rt.size(); i++)
		{
#if __SYNC_TIMER_
			if (!(3 == i || 7 == i || 11 == i || 15 == i || 21 == i))
				runtime_total += LU_rt[i];
#else
			if (3 == i || 7 == i || 11 == i || 15 == i || 16 == i || 17 == i || 21 == i || 22 == i)
				runtime_total += LU_rt[i];
#endif
		}
        std::cout << "runtime: " << runtime_total << "\n";
	}
}

bool XFLUIDS::SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t Time)
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

bool XFLUIDS::EstimateNAN(sycl::queue &q, const real_t Time, const int Step, const int rank, const int flag)
{
	std::chrono::high_resolution_clock::time_point runtime_nan_start_time = std::chrono::high_resolution_clock::now();

	bool error = false, errors[NumFluid];
	for (int n = 0; n < NumFluid; n++)
	{
		errors[n] = fluids[n]->EstimateFluidNAN(q, flag);
		error = error || errors[n];
		// if (rank == 1 && Step == 10)
		// 	error = true;
	}

	std::string StepEr = std::to_string(Step) + "_RK" + std::to_string(flag);
	if (error)
		Output(q, OutAtThis.Reinitialize(Time, "UErs_" + StepEr), 1);
#ifdef USE_MPI
	error = Ss.mpiTrans->BocastTrue(error);
#endif // end USE_MPI
	if (error)
		Output(q, OutAtThis.Reinitialize(Time, "UErr_" + StepEr), 2);
	q.wait();

	runtime_estimatenan += OutThisTime(runtime_nan_start_time);

	return error; // all rank == 1 or 0
}

bool XFLUIDS::RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag)
{
	// estimate if rho is_nan or <0 or is_inf
	bool error = false;
	std::chrono::high_resolution_clock::time_point runtime_start_time;
	switch (flag)
	{
	case 1:
		// the fisrt step
		runtime_start_time = std::chrono::high_resolution_clock::now();
		BoundaryCondition(q, 0);
		runtime_boundary += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		if (UpdateStates(q, 0, Time, Step, "_RK1"))
			return true;
		runtime_updatestates += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		ComputeLU(q, 0);
		runtime_computelu += OutThisTime(runtime_start_time);
#if ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		runtime_start_time = std::chrono::high_resolution_clock::now();
		UpdateU(q, 1);
		runtime_updateu += OutThisTime(runtime_start_time);
		break;

	case 2:
		// the second step
		runtime_start_time = std::chrono::high_resolution_clock::now();
		BoundaryCondition(q, 1);
		runtime_boundary += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		if (UpdateStates(q, 1, Time, Step, "_RK2"))
			return true;
		runtime_updatestates += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		ComputeLU(q, 1);
		runtime_computelu += OutThisTime(runtime_start_time);
#if ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		runtime_start_time = std::chrono::high_resolution_clock::now();
		UpdateU(q, 2);
		runtime_updateu += OutThisTime(runtime_start_time);
		break;

	case 3:
		// the third step
		runtime_start_time = std::chrono::high_resolution_clock::now();
		BoundaryCondition(q, 1);
		runtime_boundary += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		if (UpdateStates(q, 1, Time, Step, "_RK3"))
			return true;
		runtime_updatestates += OutThisTime(runtime_start_time);

		runtime_start_time = std::chrono::high_resolution_clock::now();
		ComputeLU(q, 1);
		runtime_computelu += OutThisTime(runtime_start_time);
#if ESTIM_NAN
		if (EstimateNAN(q, Time, Step, rank, flag))
			return true;
#endif // end ESTIM_NAN

		runtime_start_time = std::chrono::high_resolution_clock::now();
		UpdateU(q, 3);
		runtime_updateu += OutThisTime(runtime_start_time);
		break;
	}

	if (0 == rank)
		std::cout << "<<<<<<<<<<<<<< NS time advancing RungeKutta Step " << flag << " has been done" << std::endl;

	return false;
}

real_t XFLUIDS::ComputeTimeStep(sycl::queue &q)
{
	real_t dt_ref = _DF(1.0e-10);
	std::chrono::high_resolution_clock::time_point runtime_dt_start_time = std::chrono::high_resolution_clock::now();
	dt_ref = fluids[0]->GetFluidDt(q, Iteration, physicalTime);

// // if MPI used, get the minimum of all ranks
#ifdef USE_MPI
	real_t temp;
	Ss.mpiTrans->communicator->synchronize();
	Ss.mpiTrans->communicator->allReduce(&dt_ref, &temp, 1, Ss.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
	dt_ref = temp;
#endif

	runtime_getdt += OutThisTime(runtime_dt_start_time);

	return dt_ref;
}

void XFLUIDS::ComputeLU(sycl::queue &q, int flag)
{
	std::vector<float> LU_rt_temp = fluids[0]->ComputeFluidLU(q, flag);
	// vector.resize() has to do only once:
	static bool dummy = (LU_rt.resize(LU_rt_temp.size()), LU_rt.assign(LU_rt.size(), 0), true);
	for (size_t i = 0; i < LU_rt.size(); i++)
		LU_rt[i] += LU_rt_temp[i];
}

void XFLUIDS::UpdateU(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->UpdateFluidURK3(q, flag, dt);
}

void XFLUIDS::BoundaryCondition(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->BoundaryCondition(q, Ss.Boundarys, flag);
}

bool XFLUIDS::UpdateStates(sycl::queue &q, int flag, const real_t Time, const int Step, std::string RkStep)
{
	bool error_t = false;

	// for (int n = 0; n < NumFluid; n++)
	{
		std::pair temp_pair = fluids[0]->UpdateFluidStates(q, flag);
		// if (Time > 1.0E-6 && rank == 1)
		// 	error[n] = true;
		error_t = error_t || temp_pair.first; // rank error
		static bool dummy = (UD_rt.resize(temp_pair.second.size()), UD_rt.assign(UD_rt.size(), 0), true);
		if (std::string::npos != RkStep.find("_RK")) // Only Timer UpdateStates inside GetLU
			for (size_t i = 0; i < UD_rt.size(); i++)
				UD_rt[i] += temp_pair.second[i];
	}

	{
		std::string Stepstr = std::to_string(Step);
		if (error_t)
			Output(q, OutAtThis.Reinitialize(Time, "PErs_" + Stepstr + RkStep), 1);
#ifdef USE_MPI
		error_t = Ss.mpiTrans->BocastTrue(error_t);
#endif // end USE_MPI
		if (error_t)
			Output(q, OutAtThis.Reinitialize(Time, "PErr_" + Stepstr + RkStep), 2);
	}

	return error_t; // all rank == 1 or 0
}

void XFLUIDS::AllocateMemory(sycl::queue &q)
{
	d_BCs = static_cast<BConditions *>(malloc_device(6 * sizeof(BConditions), q));
	q.memcpy(d_BCs, Ss.Boundarys, 6 * sizeof(BConditions));

	// host arrays for each fluid
	if (0 == rank)
		std::cout << "<---------------------------------------------------> \n";
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->AllocateFluidMemory(q);
	// levelset->AllocateLSMemory();
	if (0 == rank)
		std::cout << "<---------------------------------------------------> \n";

	for (size_t nn = 0; nn < Ss.OutTimeStamps.size(); nn++)
		Ss.OutTimeStamps[nn].Initialize(Ss.BlSz, Ss.species_name, fluids[0]->h_fstate);
	OutAtThis = Ss.OutTimeStamps[0];
}

void XFLUIDS::InitialCondition(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->InitialU(q);

	ReadCheckingPoint = Read_Ubak(q, rank, &(Iteration), &(physicalTime), &(duration_backup));
	duration = duration_backup;
}

bool XFLUIDS::Reaction(sycl::queue &q, const real_t dt, const real_t Time, const int Step)
{
	std::chrono::high_resolution_clock::time_point runtime_reation = std::chrono::high_resolution_clock::now();

	if (UpdateStates(q, 0, Time, Step, "_React"))
		return true;

	real_t tem_dt = dt;
	if (SlipOrder == std::string("Strang"))
		tem_dt *= _DF(0.5);
	fluids[0]->ODESolver(q, tem_dt);

	runtime_rea += OutThisTime(runtime_reation);

	return EstimateNAN(q, Time, Step, rank, 4);

	return false;
}

void XFLUIDS::CopyToUbak(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->Ubak, fluids[n]->d_U, Ss.cellbytes);
	q.wait();
}

void XFLUIDS::CopyToU(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->d_U, fluids[n]->h_U, Ss.cellbytes);
	q.wait();
}

void XFLUIDS::Output_Ubak(const int rank, const int Step, const real_t Time, const float Time_consumption, bool solution)
{
	std::string file_name;
	if (solution)
		file_name = OutputDir + "/cal/" + outputPrefix + "_CheckingPoint";
	else
		file_name = OutputDir + "/" + outputPrefix + "_CheckingPoint";
#ifdef USE_MPI
	file_name += "_rank_" + std::to_string(rank);
#endif

	unsigned long long need = (sizeof(Step) + sizeof(Time) + sizeof(duration) + Ss.cellbytes);
	if (disk_avail<disk::memType::B>(Ss.WorkDir, need, "CheckingPoint output of rank: " + std::to_string(rank)))
	{
		std::ofstream fout(file_name, std::ios::out | std::ios::binary);
		fout.write((char *)&(Step), sizeof(Step));
		fout.write((char *)&(Time), sizeof(Time));
		fout.write((char *)&(Time_consumption), sizeof(duration));
		for (size_t n = 0; n < NumFluid; n++)
			fout.write((char *)(fluids[n]->Ubak), Ss.cellbytes);
		fout.close();
	}

	if (rank == 0)
	{
		if (solution)
			std::cout << "Solution ";
		std::cout << "CheckingPoint file of Step = " << Step << " has been output." << std::endl;
	}
}

bool XFLUIDS::Read_Ubak(sycl::queue &q, const int rank, int *Step, real_t *Time, float *Time_consumption)
{
	int size = Ss.cellbytes, all_read = 1;
	std::string file_name;
	file_name = OutputDir + "/" + outputPrefix + "_CheckingPoint";
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
			std::cout << "CheckingPoint-file not exist or open failed, CheckingPoint closed." << std::endl;
		return false;
	}
	fin.read((char *)Step, sizeof(int));
	fin.read((char *)Time, sizeof(real_t));
	fin.read((char *)Time_consumption, sizeof(float));
	for (size_t n = 0; n < NumFluid; n++)
		fin.read((char *)(fluids[n]->Ubak), size);
	fin.close();

	for (int n = 0; n < NumFluid; n++)
		q.memcpy(fluids[n]->d_U, fluids[n]->Ubak, Ss.cellbytes);
	q.wait();

	return true; // ReIni U for additonal continued caculate
}

void XFLUIDS::CopyDataFromDevice(sycl::queue &q, bool error)
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
#ifdef COP
		q.memcpy(fluids[n]->h_fstate.y, fluids[n]->d_fstate.y, bytes * NUM_SPECIES);
#endif // COP
		if (Visc)
		{
			q.memcpy(fluids[n]->h_fstate.vx, fluids[n]->d_fstate.vx, bytes);
			for (size_t i = 0; i < 3; i++)
				q.memcpy(fluids[n]->h_fstate.vxs[i], fluids[n]->d_fstate.vxs[i], bytes).wait();
		}

		if (error)
		{
			q.memcpy(fluids[n]->h_U, fluids[n]->d_U, cellbytes);
			q.memcpy(fluids[n]->h_U1, fluids[n]->d_U1, cellbytes);
			q.memcpy(fluids[n]->h_LU, fluids[n]->d_LU, cellbytes);

#if ESTIM_OUT
			if (Visc) // copy vosicous estimating Vars
			{
				if (Ss.BlSz.DimX)
					q.memcpy(fluids[n]->h_fstate.visFwx, fluids[n]->d_fstate.visFwx, NUM_SPECIES * bytes);
				if (Ss.BlSz.DimY)
					q.memcpy(fluids[n]->h_fstate.visFwy, fluids[n]->d_fstate.visFwy, NUM_SPECIES * bytes);
				if (Ss.BlSz.DimZ)
					q.memcpy(fluids[n]->h_fstate.visFwz, fluids[n]->d_fstate.visFwz, NUM_SPECIES * bytes);

				if (Visc_Diffu)
				{
					q.memcpy(fluids[n]->h_fstate.Ertemp1, fluids[n]->d_fstate.Ertemp1, NUM_SPECIES * bytes);
					q.memcpy(fluids[n]->h_fstate.Ertemp2, fluids[n]->d_fstate.Ertemp2, NUM_SPECIES * bytes);
					q.memcpy(fluids[n]->h_fstate.Dkm_aver, fluids[n]->d_fstate.Dkm_aver, NUM_SPECIES * bytes);
					if (Ss.BlSz.DimX)
					{
						q.memcpy(fluids[n]->h_fstate.Dim_wallx, fluids[n]->d_fstate.Dim_wallx, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.hi_wallx, fluids[n]->d_fstate.hi_wallx, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yi_wallx, fluids[n]->d_fstate.Yi_wallx, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yil_wallx, fluids[n]->d_fstate.Yil_wallx, NUM_SPECIES * bytes);
					}
					if (Ss.BlSz.DimY)
					{
						q.memcpy(fluids[n]->h_fstate.Dim_wally, fluids[n]->d_fstate.Dim_wally, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.hi_wally, fluids[n]->d_fstate.hi_wally, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yi_wally, fluids[n]->d_fstate.Yi_wally, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yil_wally, fluids[n]->d_fstate.Yil_wally, NUM_SPECIES * bytes);
					}
					if (Ss.BlSz.DimZ)
					{
						q.memcpy(fluids[n]->h_fstate.Dim_wallz, fluids[n]->d_fstate.Dim_wallz, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.hi_wallz, fluids[n]->d_fstate.hi_wallz, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yi_wallz, fluids[n]->d_fstate.Yi_wallz, NUM_SPECIES * bytes);
						q.memcpy(fluids[n]->h_fstate.Yil_wallz, fluids[n]->d_fstate.Yil_wallz, NUM_SPECIES * bytes);
					}
				}
			}

			if (Ss.BlSz.DimX)
			{
				q.memcpy(fluids[n]->h_fstate.b1x, fluids[n]->d_fstate.b1x, bytes);
				q.memcpy(fluids[n]->h_fstate.b3x, fluids[n]->d_fstate.b3x, bytes);
				q.memcpy(fluids[n]->h_fstate.c2x, fluids[n]->d_fstate.c2x, bytes);
				q.memcpy(fluids[n]->h_fstate.zix, fluids[n]->d_fstate.zix, bytes * NUM_COP);
				q.memcpy(fluids[n]->h_fstate.preFwx, fluids[n]->d_fstate.preFwx, cellbytes);
				q.memcpy(fluids[n]->h_fstate.pstFwx, fluids[n]->d_wallFluxF, cellbytes);
			}
			if (Ss.BlSz.DimY)
			{
				q.memcpy(fluids[n]->h_fstate.b1y, fluids[n]->d_fstate.b1y, bytes);
				q.memcpy(fluids[n]->h_fstate.b3y, fluids[n]->d_fstate.b3y, bytes);
				q.memcpy(fluids[n]->h_fstate.c2y, fluids[n]->d_fstate.c2y, bytes);
				q.memcpy(fluids[n]->h_fstate.ziy, fluids[n]->d_fstate.ziy, bytes * NUM_COP);
				q.memcpy(fluids[n]->h_fstate.preFwy, fluids[n]->d_fstate.preFwy, cellbytes);
				q.memcpy(fluids[n]->h_fstate.pstFwy, fluids[n]->d_wallFluxG, cellbytes);
			}
			if (Ss.BlSz.DimZ)
			{
				q.memcpy(fluids[n]->h_fstate.b1z, fluids[n]->d_fstate.b1z, bytes);
				q.memcpy(fluids[n]->h_fstate.b3z, fluids[n]->d_fstate.b3z, bytes);
				q.memcpy(fluids[n]->h_fstate.c2z, fluids[n]->d_fstate.c2z, bytes);
				q.memcpy(fluids[n]->h_fstate.ziz, fluids[n]->d_fstate.ziz, bytes * NUM_COP);
				q.memcpy(fluids[n]->h_fstate.preFwz, fluids[n]->d_fstate.preFwz, cellbytes);
				q.memcpy(fluids[n]->h_fstate.pstFwz, fluids[n]->d_wallFluxH, cellbytes);
			}
#endif // end ESTIM_OUT
		}
	}
	q.wait();
}

void XFLUIDS::GetCPT_OutRanks(int *OutRanks, OutSize &CVTI, OutSlice pos)
{
	// compressible out: output dirs less than caculate dirs
	Block bl = Ss.BlSz;
	bool Out1, Out2, Out3;
	int if_outrank = nranks > 1 ? -1 : rank;
	real_t temx = _DF(0.5) * bl.dx + bl.Domain_xmin;
	real_t temy = _DF(0.5) * bl.dy + bl.Domain_ymin;
	real_t temz = _DF(0.5) * bl.dz + bl.Domain_zmin;
	real_t posx = -bl.Bwidth_X + bl.myMpiPos_x * (bl.X_inner);
	real_t posy = -bl.Bwidth_Y + bl.myMpiPos_y * (bl.Y_inner);
	real_t posz = -bl.Bwidth_Z + bl.myMpiPos_z * (bl.Z_inner);

	for (int k = VTI.minZ; k < VTI.maxZ; k++)
		for (int j = VTI.minY; j < VTI.maxY; j++)
			for (int i = VTI.minX; i < VTI.maxX; i++)
			{
				int pos_x = i + posx, pos_y = j + posy, pos_z = k + posz;
				real_t x = bl.DimX_t * ((pos_x + _DF(0.5)) * bl.dx + bl.Domain_xmin);
				real_t y = bl.DimY_t * ((pos_y + _DF(0.5)) * bl.dy + bl.Domain_ymin);
				real_t z = bl.DimZ_t * ((pos_z + _DF(0.5)) * bl.dz + bl.Domain_zmin);
				Out1 = (!pos.OutDirX), Out2 = (!pos.OutDirY), Out3 = (!pos.OutDirZ);
				if (Out1 || Out2 || Out3)
				{
					if_outrank = rank;
				}
				if (Out1 && (x - bl.dx <= pos.outpos_x) && (x > pos.outpos_x))
				{
					CVTI.minX = i;
					break;
				}
				if (Out2 && (y - bl.dy <= pos.outpos_y) && (y > pos.outpos_y))
				{
					CVTI.minX = j;
					break;
				}
				if (Out3 && (z - bl.dz <= pos.outpos_z) && (z > pos.outpos_z))
				{
					CVTI.minZ = k;
					break;
				}
			}

#ifdef USE_MPI
	MPI_Allgather(&(if_outrank), 1, MPI_INT, OutRanks, 1, MPI_INT, MPI_COMM_WORLD);
	// Ss.mpiTrans->communicator->allGather(&(if_outrank), 1, mpiUtils::MpiComm::INT, OutRanks, 1, mpiUtils::MpiComm::INT);
#endif

	CVTI.nbX = pos.OutDirX ? VTI.nbX : 1;
	CVTI.nbY = pos.OutDirY ? VTI.nbY : 1;
	CVTI.nbZ = pos.OutDirZ ? VTI.nbZ : 1;
	CVTI.maxX = CVTI.minX + CVTI.nbX, CVTI.maxY = CVTI.minY + CVTI.nbY, CVTI.maxZ = CVTI.minZ + CVTI.nbZ;
}

void XFLUIDS::GetSPT_OutRanks(int *OutRanks, std::vector<Criterion> &var)
{
	// partial out: output size less than caculate size
	int if_outrank = !std::empty(var) ? -1 : rank;
	OutSize rank_size = VTI;

	for (int k = VTI.minZ; k < VTI.maxZ; k++)
		for (int j = VTI.minY; j < VTI.maxY; j++)
			for (int i = VTI.minX; i < VTI.maxX; i++)
			{
				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
				for (size_t n = 0; n < var.size(); n++)
				{
					if (var[n].in_range(id))
					{
						if_outrank = rank;
						goto flag_out;
						// rank_size.minX = std::max(rank_size.minX, i);
						// rank_size.maxX = std::min(rank_size.maxX, i);
						// rank_size.minY = std::max(rank_size.minY, j);
						// rank_size.maxY = std::min(rank_size.maxY, j);
						// rank_size.minZ = std::max(rank_size.minZ, k);
						// rank_size.maxZ = std::min(rank_size.maxZ, k);
					}
				}
			}

flag_out:
	OutRanks[rank] = if_outrank;

#ifdef USE_MPI
	MPI_Allgather(&(if_outrank), 1, MPI_INT, OutRanks, 1, MPI_INT, MPI_COMM_WORLD);
	// Ss.mpiTrans->communicator->synchronize();
	// Ss.mpiTrans->communicator->allGather(&(if_outrank), 1, mpiUtils::MpiComm::INT, OutRanks, 1, mpiUtils::MpiComm::INT);
#endif
}

std::vector<OutVar> XFLUIDS::Output_variables(FlowData &data, std::vector<std::string> &sp, size_t error)
{
	Block bl = Ss.BlSz;
	// Init var names
	std::vector<OutVar> vars;
	vars.clear();

	vars.push_back(OutVar("rho", data.rho));
	vars.push_back(OutVar("p", data.p));
	vars.push_back(OutVar("T", data.T));
#ifdef COP
	for (size_t nn = 0; nn < sp.size(); nn++)
		vars.push_back(OutVar("y" + std::to_string(nn) + "[" + sp[nn] + "]", data.y, sp.size(), nn));
#endif // COP

	if (1 == error)
	{
		for (size_t u = 0; u < Emax; u++)
		{
			vars.push_back(OutVar("E-U[" + std::to_string(u) + "]", fluids[0]->h_U, Emax, u));
			vars.push_back(OutVar("E-U1[" + std::to_string(u) + "]", fluids[0]->h_U1, Emax, u));
			vars.push_back(OutVar("E-LU[" + std::to_string(u) + "]", fluids[0]->h_LU, Emax, u));
		}
#if ESTIM_OUT
		if (Ss.BlSz.DimX)
		{
			// vars.push_back(OutVar( "E-xb1", fluids[0]->h_fstate.b1x));
			// vars.push_back(OutVar( "E-xb3", fluids[0]->h_fstate.b3x));
			// vars.push_back(OutVar( "E-xc3", fluids[0]->h_fstate.c2x));
			for (size_t mm = 0; mm < Emax; mm++)
			{
				vars.push_back(OutVar("E-Fw-prev-x[" + std::to_string(mm) + "]", fluids[0]->h_fstate.preFwx, Emax, mm));
				vars.push_back(OutVar("E-Fw-pstv-x[" + std::to_string(mm) + "]", fluids[0]->h_fstate.pstFwx, Emax, mm));
			}
		}
		if (Ss.BlSz.DimY)
		{
			// vars.push_back(OutVar( "E-yb1", fluids[0]->h_fstate.b1y));
			// vars.push_back(OutVar( "E-yb3", fluids[0]->h_fstate.b3y));
			// vars.push_back(OutVar( "E-yc3", fluids[0]->h_fstate.c2y));
			for (size_t mm = 0; mm < Emax; mm++)
			{
				vars.push_back(OutVar("E-Fw-prev-y[" + std::to_string(mm) + "]", fluids[0]->h_fstate.preFwy, Emax, mm));
				vars.push_back(OutVar("E-Fw-pstv-y[" + std::to_string(mm) + "]", fluids[0]->h_fstate.pstFwy, Emax, mm));
			}
		}
		if (Ss.BlSz.DimZ)
		{
			// vars.push_back(OutVar( "E-zb1", fluids[0]->h_fstate.b1z));
			// vars.push_back(OutVar( "E-zb3", fluids[0]->h_fstate.b3z));
			// vars.push_back(OutVar( "E-zc3", fluids[0]->h_fstate.c2z));
			for (size_t mm = 0; mm < Emax; mm++)
			{
				vars.push_back(OutVar("E-Fw-prev-z[" + std::to_string(mm) + "]", fluids[0]->h_fstate.preFwz, Emax, mm));
				vars.push_back(OutVar("E-Fw-pstv-z[" + std::to_string(mm) + "]", fluids[0]->h_fstate.pstFwz, Emax, mm));
			}
		}

		if (Visc) // Out name of viscous out estimating Vars
			for (size_t mm = 0; mm < Emax; mm++)
			{
				if (Ss.BlSz.DimX)
					vars.push_back(OutVar("E-Fw-vis-x[" + std::to_string(mm) + "]", fluids[0]->h_fstate.visFwx, Emax, mm));
				if (Ss.BlSz.DimY)
					vars.push_back(OutVar("E-Fw-vis-y[" + std::to_string(mm) + "]", fluids[0]->h_fstate.visFwy, Emax, mm));
				if (Ss.BlSz.DimZ)
					vars.push_back(OutVar("E-Fw-vis-z[" + std::to_string(mm) + "]", fluids[0]->h_fstate.visFwz, Emax, mm));

				if (Visc_Diffu)
					for (size_t mm = 0; mm < sp.size(); mm++)
					{
						vars.push_back(OutVar("E-vis_Dim[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Dkm_aver, sp.size(), mm));
						vars.push_back(OutVar("E-vis_Dimtemp1[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Ertemp1, sp.size(), mm));
						vars.push_back(OutVar("E-vis_Dimtemp2[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Ertemp2, sp.size(), mm));
						if (Ss.BlSz.DimX)
						{
							vars.push_back(OutVar("E-vis_Dimwallx[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Dim_wallx, sp.size(), mm));
							vars.push_back(OutVar("E-vis_hi_wallx[" + std::to_string(mm) + "]", fluids[0]->h_fstate.hi_wallx, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yi_wallx[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yi_wallx, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yil_wallx[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yil_wallx, sp.size(), mm));
						}
						if (Ss.BlSz.DimY)
						{
							vars.push_back(OutVar("E-vis_Dimwally[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Dim_wally, sp.size(), mm));
							vars.push_back(OutVar("E-vis_hi_wally[" + std::to_string(mm) + "]", fluids[0]->h_fstate.hi_wally, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yi_wally[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yi_wally, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yil_wally[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yil_wally, sp.size(), mm));
						}
						if (Ss.BlSz.DimZ)
						{
							vars.push_back(OutVar("E-vis_Dimwallz[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Dim_wallz, sp.size(), mm));
							vars.push_back(OutVar("E-vis_hi_wallz[" + std::to_string(mm) + "]", fluids[0]->h_fstate.hi_wallz, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yi_wallz[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yi_wallz, sp.size(), mm));
							vars.push_back(OutVar("E-vis_Yil_wallz[" + std::to_string(mm) + "]", fluids[0]->h_fstate.Yil_wallz, sp.size(), mm));
						}
					}
			}
#endif // ESTIM_OUT
	}

	return vars;
}

void XFLUIDS::Output(sycl::queue &q, OutFmt ctrl, size_t error)
{
	// Write time in string timeFormat
	OutString osr(ctrl.time, rank, ctrl.inter);
	CopyDataFromDevice(q, error); // only copy when output

	if (error)
	{
		// Init error var names
		std::vector<OutVar> error_vars = Output_variables(fluids[0]->h_fstate, Ss.species_name, error);
		if (OutVTI)
			Output_vti(error_vars, osr, error);
		if (OutDAT)
			Output_cplt(ctrl.out_vars, ctrl.pos, osr);

		if (rank == 0)
			std::cout << "Errors Captured solution";
	}
	else if (ctrl.CPOut)
	{
		if (OutDAT)
			Output_cplt(ctrl.out_vars, ctrl.pos, osr);
		if (OutVTI)
			Output_cvti(ctrl.out_vars, ctrl.pos, osr);

		if (rank == 0)
			std::cout << "Compress Dimensions solution";
	}
	else if (ctrl.SPOut)
	{
		if (OutVTI)
			Output_svti(ctrl.out_vars, ctrl.cri_list, osr);

		if (rank == 0)
			std::cout << "Partial Domain solution";
	}
	else
	{
		if (OutDAT)
			Output_cplt(ctrl.out_vars, ctrl.pos, osr);
		if (OutVTI)
			Output_svti(ctrl.out_vars, ctrl.cri_list, osr);

		if (rank == 0)
			std::cout << "Common Domain solution";
	}
	if (rank == 0)
		std::cout << " has been done at Step = " << ctrl.inter << ", Time = " << ctrl.time << std::endl;
}

template <typename T>
void XFLUIDS::Output_vti(std::vector<OutVar> error_vars, OutString &osr, size_t error)
{
	real_t dx = 0.0, dy = 0.0, dz = 0.0;
	int xmin = 0, ymin = 0, xmax = 0, ymax = 0, zmin = 0, zmax = 0;
	if (Ss.BlSz.DimX)
		xmin = Ss.BlSz.myMpiPos_x * VTI.nbX, xmax = Ss.BlSz.myMpiPos_x * VTI.nbX + VTI.nbX, dx = Ss.BlSz.dx;
	if (Ss.BlSz.DimY)
		ymin = Ss.BlSz.myMpiPos_y * VTI.nbY, ymax = Ss.BlSz.myMpiPos_y * VTI.nbY + VTI.nbY, dy = Ss.BlSz.dy;
	if (Ss.BlSz.DimZ)
		zmin = Ss.BlSz.myMpiPos_z * VTI.nbZ, zmax = Ss.BlSz.myMpiPos_z * VTI.nbZ + VTI.nbZ, dz = Ss.BlSz.dz;

	std::string file_name;
	std::string temp_name = "./VTI_" + outputPrefix + "_Step_Time_" + osr.stepFormat.str() + "." + osr.timeFormat.str();

	std::string headerfile_name = OutputDir + "/VTI_" + outputPrefix + "_Step_" + osr.stepFormat.str() + ".pvti";
	{ // out pvti header
		if (0 == rank && 2 == error)
		{
			// // write header
			std::fstream outHeader;
			// // dummy string here, when using the full VTK API, data can be compressed
			// // here, no compression used
			std::string compressor("");
			// // open pvti header file
			int mx = (Ss.BlSz.DimX) ? Ss.BlSz.mx : 0;
			int my = (Ss.BlSz.DimY) ? Ss.BlSz.my : 0;
			int mz = (Ss.BlSz.DimZ) ? Ss.BlSz.mz : 0;
			outHeader.open(headerfile_name.c_str(), std::ios_base::out);
			outHeader << "<?xml version=\"1.0\"?>" << std::endl;
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
			for (int iVar = 0; iVar < error_vars.size(); iVar++)
				outHeader << "      <PDataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"" << error_vars[iVar].name << "\"/>" << std::endl;
			outHeader << "    </PCellData>" << std::endl;
			// // Out put for 2D && 3D;
			for (int iPiece = 0; iPiece < Ss.nRanks; ++iPiece)
			{
				std::ostringstream pieceFormat;
				pieceFormat.width(5);
				pieceFormat.fill('0');
				pieceFormat << iPiece;
				std::string pieceFilename = temp_name + "_rank_" + pieceFormat.str() + ".vti";
				// get MPI coords corresponding to MPI rank iPiece
				int coords[3] = {0, 0, 0};
#ifdef USE_MPI
				Ss.mpiTrans->communicator->getCoords(iPiece, Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ, coords);
#endif
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

				if (3 == Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ)
				{
					if (coords[2] == 0)
						outHeader << 0 << " " << VTI.nbZ << " ";
					else
						outHeader << coords[2] * VTI.nbZ << " " << coords[2] * VTI.nbZ + VTI.nbZ << " ";
				}
				else
					outHeader << 0 << " " << 0;

				outHeader << "\" Source=\"";
				outHeader << pieceFilename << "\"/>" << std::endl;
			}
			outHeader << "</PImageData>" << std::endl;
			outHeader << "</VTKFile>" << std::endl;
			// // close header file
			outHeader.close();
		} // end writing pvti header
	}

	file_name = OutputDir + "/" + temp_name + "_rank_" + osr.rankFormat.str() + ".vti";
	unsigned long long need = error_vars.size() * (VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(T) + 4); // Bytes
	if (disk_avail<disk::B>(Ss.WorkDir, need, "Error output of rank: " + std::to_string(rank)))
	{
		std::fstream outFile;
		outFile.open(file_name.c_str(), std::ios_base::out);
		// write xml data header
		outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
		outFile << "  <ImageData WholeExtent=\""
				<< xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << "\" "
				<< "Origin=\""
				<< Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
				<< "Spacing=\"" << dx << " " << dy << " " << dz << "\">" << std::endl;
		outFile << "  <Piece Extent=\""
				<< xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << "\">" << std::endl;
		outFile << "    <PointData>\n";
		outFile << "    </PointData>\n";
		// write data in binary format
		outFile << "    <CellData>" << std::endl;
		for (int iVar = 0; iVar < error_vars.size(); iVar++)
		{
			outFile << "     <DataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"";
			outFile << error_vars[iVar].name << "\" format=\"appended\" offset=\""
					<< iVar * VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(T) + iVar * sizeof(unsigned int)
					<< "\" />" << std::endl;
		}
		outFile << "    </CellData>" << std::endl;
		outFile << "  </Piece>" << std::endl;
		outFile << "  </ImageData>" << std::endl;
		outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
		// write the leading undescore
		outFile << "_";

		// then write heavy data (column major format)
		{
			for (size_t iv = 0; iv < error_vars.size(); iv++)
				error_vars[iv].vti_binary<T>(Ss.BlSz, outFile, VTI);
		} // End Var Output

		outFile << "  </AppendedData>" << std::endl;
		outFile << "</VTKFile>" << std::endl;
		outFile.close();
	}
}

template <typename T>
void XFLUIDS::Output_svti(std::vector<OutVar> &varout, std::vector<Criterion> &cri, OutString &osr)
{
	int *OutRanks = new int[nranks]{0};
	real_t dx = 0.0, dy = 0.0, dz = 0.0;
	int minMpiPos_x = 0, maxMpiPos_x = Ss.BlSz.mx - 1;
	int minMpiPos_y = 0, maxMpiPos_y = Ss.BlSz.my - 1;
	int minMpiPos_z = 0, maxMpiPos_z = Ss.BlSz.mz - 1;
	int xmin = 0, ymin = 0, xmax = 0, ymax = 0, zmin = 0, zmax = 0, mx = 0, my = 0, mz = 0;
	if (Ss.BlSz.DimX)
		xmin = Ss.BlSz.myMpiPos_x * VTI.nbX, xmax = Ss.BlSz.myMpiPos_x * VTI.nbX + VTI.nbX, dx = Ss.BlSz.dx;
	if (Ss.BlSz.DimY)
		ymin = Ss.BlSz.myMpiPos_y * VTI.nbY, ymax = Ss.BlSz.myMpiPos_y * VTI.nbY + VTI.nbY, dy = Ss.BlSz.dy;
	if (Ss.BlSz.DimZ)
		zmin = Ss.BlSz.myMpiPos_z * VTI.nbZ, zmax = Ss.BlSz.myMpiPos_z * VTI.nbZ + VTI.nbZ, dz = Ss.BlSz.dz;

	std::string file_name;
	std::string temp_name = "./VTI_" + outputPrefix + "_Step_Time_" + osr.stepFormat.str() + "." + osr.timeFormat.str();
	file_name = OutputDir + "/" + temp_name;
#ifdef USE_MPI
	file_name = file_name + "_rank_" + osr.rankFormat.str();

	GetSPT_OutRanks(OutRanks, cri);
	if (nranks > 1)
	{
		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_x), &(minMpiPos_x), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MIN, OutRanks, true);
		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_y), &(minMpiPos_y), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MIN, OutRanks, true);
		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_z), &(minMpiPos_z), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MIN, OutRanks, true);

		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_x), &(maxMpiPos_x), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX, OutRanks, true);
		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_y), &(maxMpiPos_y), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX, OutRanks, true);
		Ss.mpiTrans->GroupallReduce(&(Ss.BlSz.myMpiPos_z), &(maxMpiPos_z), 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX, OutRanks, true);

		if (((minMpiPos_x - 1 == Ss.BlSz.myMpiPos_x) || (maxMpiPos_x + 1 == Ss.BlSz.myMpiPos_x)) && (Ss.BlSz.mx > 4))
			OutRanks[rank] = rank;
	}
#endif
	file_name += ".vti";

	std::string headerfile_name = OutputDir + "/VTI_" + outputPrefix + "_Time_" + osr.timeFormat.str() + ".pvti";
	{ // out pvti header
		mx = Ss.BlSz.mx;
		my = Ss.BlSz.my;
		mz = (3 == Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ) ? Ss.BlSz.mz : 0;

		if ((OutRanks[rank] >= 0) && (minMpiPos_x == Ss.BlSz.myMpiPos_x) && (minMpiPos_y == Ss.BlSz.myMpiPos_y) && (minMpiPos_z == Ss.BlSz.myMpiPos_z))
		{
			// // write header
			std::fstream outHeader;
			// // dummy string here, when using the full VTK API, data can be compressed
			// // here, no compression used
			std::string compressor("");
			// // open pvti header file
			outHeader.open(headerfile_name.c_str(), std::ios_base::out);
			outHeader << "<?xml version=\"1.0\"?>" << std::endl;
			outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
			outHeader << "  <PImageData WholeExtent=\"";
			outHeader << minMpiPos_x * VTI.nbX << " " << (maxMpiPos_x + 1) * VTI.nbX * int(Ss.BlSz.DimX) << " ";
			outHeader << minMpiPos_y * VTI.nbY << " " << (maxMpiPos_y + 1) * VTI.nbY * int(Ss.BlSz.DimY) << " ";
			outHeader << minMpiPos_z * VTI.nbZ << " " << (maxMpiPos_z + 1) * VTI.nbZ * int(Ss.BlSz.DimZ) << "\" GhostLevel=\"0\" "
					  << "Origin=\""
					  << Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
					  << "Spacing=\""
					  << dx << " " << dy << " " << dz << "\">"
					  << std::endl;
			outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;
			for (int iVar = 0; iVar < varout.size(); iVar++)
				outHeader << "      <PDataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"" << varout[iVar].name << "\"/>" << std::endl;
			outHeader << "    </PCellData>" << std::endl;
			// // Out put for 2D && 3D;
			for (int iPiece = 0; iPiece < Ss.nRanks; ++iPiece)
				if (OutRanks[iPiece] >= 0)
				{
					std::ostringstream pieceFormat;
					pieceFormat.width(5);
					pieceFormat.fill('0');
					pieceFormat << iPiece;
					std::string pieceFilename = temp_name;
#ifdef USE_MPI
					pieceFilename += "_rank_" + pieceFormat.str();
#endif
					pieceFilename += +".vti";
					// get MPI coords corresponding to MPI rank iPiece
					int coords[3] = {0, 0, 0};
#ifdef USE_MPI
					Ss.mpiTrans->communicator->getCoords(iPiece, Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ, coords);
#endif
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

					if (3 == Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ)
					{
						if (coords[2] == 0)
							outHeader << 0 << " " << VTI.nbZ << " ";
						else
							outHeader << coords[2] * VTI.nbZ << " " << coords[2] * VTI.nbZ + VTI.nbZ << " ";
					}
					else
						outHeader << 0 << " " << 0;

					outHeader << "\" Source=\"";
					outHeader << pieceFilename << "\"/>" << std::endl;
				}
			outHeader << "</PImageData>" << std::endl;
			outHeader << "</VTKFile>" << std::endl;
			// // close header file
			outHeader.close();
		} // end writing pvti header
	}

	if (OutRanks[rank] >= 0)
	{
		int num = 0;
		for (size_t ni = 0; ni < nranks; ni++)
		{
			if (OutRanks[rank] >= 0)
				num += 1;
		}
		unsigned long long need = num * varout.size() * (VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(T) + 4); // Bytes
		if (disk_avail<disk::B>(Ss.WorkDir, need, "Partial output of rank: " + std::to_string(rank)))
		{
			std::fstream outFile;
			outFile.open(file_name.c_str(), std::ios_base::out);
			// write xml data header
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
			for (int iVar = 0; iVar < varout.size(); iVar++)
			{
				outFile << "     <DataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"";
				outFile << varout[iVar].name << "\" format=\"appended\" offset=\""
						<< iVar * VTI.nbX * VTI.nbY * VTI.nbZ * sizeof(T) + iVar * sizeof(unsigned int)
						<< "\" />" << std::endl;
			}
			outFile << "    </CellData>" << std::endl;
			outFile << "  </Piece>" << std::endl;
			outFile << "  </ImageData>" << std::endl;
			outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
			// write the leading undescore
			outFile << "_";

			// then write heavy data (column major format)
			{
				for (size_t iv = 0; iv < varout.size(); iv++)
					varout[iv].vti_binary<T>(Ss.BlSz, outFile, VTI);
			} // End Var Output

			outFile << "  </AppendedData>" << std::endl;
			outFile << "</VTKFile>" << std::endl;
			outFile.close();
		}
	}
}

// Need (Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ > 1)
template <typename T>
void XFLUIDS::Output_cvti(std::vector<OutVar> &varout, OutSlice &pos, OutString &osr)
{
	real_t dx = 0.0, dy = 0.0, dz = 0.0;
	int xmin = 0, ymin = 0, xmax = 0, ymax = 0, zmin = 0, zmax = 0, mx = 0, my = 0, mz = 0;
	if (pos.OutDirX && Ss.BlSz.DimX)
		xmin = Ss.BlSz.myMpiPos_x * VTI.nbX, xmax = Ss.BlSz.myMpiPos_x * VTI.nbX + VTI.nbX, dx = Ss.BlSz.dx;
	if (pos.OutDirY && Ss.BlSz.DimY)
		ymin = Ss.BlSz.myMpiPos_y * VTI.nbY, ymax = Ss.BlSz.myMpiPos_y * VTI.nbY + VTI.nbY, dy = Ss.BlSz.dy;
	if (pos.OutDirZ && Ss.BlSz.DimZ)
		zmin = (Ss.BlSz.myMpiPos_z * VTI.nbZ), zmax = (Ss.BlSz.myMpiPos_z * VTI.nbZ + VTI.nbZ), dz = Ss.BlSz.dz;

	std::string file_name;
	std::string temp_name = "./CVTI_" + outputPrefix + "_Step_Time_" + osr.stepFormat.str() + "." + osr.timeFormat.str();
	file_name = OutputDir + "/" + temp_name;

	OutSize CVTI = VTI;
	int *OutRanks = new int[nranks]{0};
#ifdef USE_MPI
	GetCPT_OutRanks(OutRanks, CVTI, pos);
	file_name = file_name + "_rank_" + osr.rankFormat.str();
#endif
	file_name += ".vti";

	std::string headerfile_name = OutputDir + "/CVTI_" + outputPrefix + "_Time_" + osr.timeFormat.str() + ".pvti";
	mx = (pos.OutDirX) ? Ss.BlSz.mx : 0;
	my = (pos.OutDirY) ? Ss.BlSz.my : 0;
	mz = (pos.OutDirZ) ? Ss.BlSz.mz : 0;
	if (0 == rank) // write header
	{
		std::fstream outHeader;
		std::string compressor("");
		// open pvti header file
		outHeader.open(headerfile_name.c_str(), std::ios_base::out);
		outHeader << "<?xml version=\"1.0\"?>" << std::endl;
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
		for (int iVar = 0; iVar < varout.size(); iVar++)
			outHeader << "      <PDataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"" << varout[iVar].name << "\"/>" << std::endl;
		outHeader << "    </PCellData>" << std::endl;

		// Out put for 2D && 3D;
		for (int iPiece = 0; iPiece < Ss.nRanks; ++iPiece)
			if (OutRanks[iPiece] >= 0)
			{
				std::ostringstream pieceFormat;
				pieceFormat.width(5);
				pieceFormat.fill('0');
				pieceFormat << iPiece;
				std::string pieceFilename = temp_name;
#ifdef USE_MPI
				pieceFilename += "_rank_" + pieceFormat.str();
#endif
				pieceFilename += +".vti";

				// get MPI coords corresponding to MPI rank iPiece
				int coords[3] = {0, 0, 0};
				int OnbX = (pos.OutDirX) ? VTI.nbX : 0, OnbY = (pos.OutDirY) ? VTI.nbY : 0, OnbZ = (pos.OutDirZ) ? VTI.nbZ : 0;
#ifdef USE_MPI
				Ss.mpiTrans->communicator->getCoords(iPiece, Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ, coords);
#endif // end USE_MPI

				outHeader << " <Piece Extent=\"";
				// pieces in first line of column are different (due to the special
				// pvti file format with overlapping by 1 cell)
				if (Ss.BlSz.DimX)
				{
					if (coords[0] == 0)
						outHeader << 0 << " " << OnbX << " ";
					else
						outHeader << coords[0] * OnbX << " " << coords[0] * OnbX + OnbX << " ";
				}
				else
					outHeader << 0 << " " << 0;

				if (Ss.BlSz.DimY)
				{
					if (coords[1] == 0)
						outHeader << 0 << " " << OnbY << " ";
					else
						outHeader << coords[1] * OnbY << " " << coords[1] * OnbY + OnbY << " ";
				}
				else
					outHeader << 0 << " " << 0;

				if (Ss.BlSz.DimZ)
				{
					if (coords[2] == 0)
						outHeader << 0 << " " << OnbZ << " ";
					else
						outHeader << coords[2] * OnbZ << " " << coords[2] * OnbZ + OnbZ << " ";
				}
				else
					outHeader << 0 << " " << 0;
				outHeader << "\" Source=\"";
				outHeader << pieceFilename << "\"/>" << std::endl;
			}
		outHeader << "</PImageData>" << std::endl;
		outHeader << "</VTKFile>" << std::endl;
		// close header file
		outHeader.close();
	} // end writing pvti header

	if (OutRanks[rank] >= 0)
	{
		int num = 0;
		for (size_t ni = 0; ni < nranks; ni++)
		{
			if (OutRanks[rank] >= 0)
				num += 1;
		}
		unsigned long long need = num * varout.size() * (CVTI.nbX * CVTI.nbY * CVTI.nbZ * sizeof(T) + 4); // Bytes
		if (disk_avail<disk::B>(Ss.WorkDir, need, "Compress output of rank: " + std::to_string(rank)))
		{
			std::fstream outFile;
			outFile.open(file_name.c_str(), std::ios_base::out);
			// write xml data header
			outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
			outFile << "  <ImageData WholeExtent=\""
					<< xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << "\" "
					<< "Origin=\""
					<< Ss.BlSz.Domain_xmin << " " << Ss.BlSz.Domain_ymin << " " << Ss.BlSz.Domain_zmin << "\" "
					<< "Spacing=\""
					<< dx << " " << dy << " " << dz << "\">" << std::endl;
			outFile << "  <Piece Extent=\""
					<< xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << "\">" << std::endl;
			outFile << "    <PointData>\n";
			outFile << "    </PointData>\n";
			// write data in binary format
			outFile << "    <CellData>" << std::endl;
			for (int iVar = 0; iVar < varout.size(); iVar++)
			{
				outFile << "     <DataArray type=\"Float" << sizeof(T) * 8 << "\" Name=\"";
				outFile << varout[iVar].name << "\" format=\"appended\" offset=\""
						<< iVar * CVTI.nbX * CVTI.nbY * CVTI.nbZ * sizeof(T) + iVar * sizeof(unsigned int)
						<< "\" />" << std::endl;
			}
			outFile << "    </CellData>" << std::endl;
			outFile << "  </Piece>" << std::endl;
			outFile << "  </ImageData>" << std::endl;
			outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
			// write the leading undescore
			outFile << "_";

			// then write heavy data (column major format)
			{
				for (size_t iv = 0; iv < varout.size(); iv++)
					varout[iv].vti_binary<T>(Ss.BlSz, outFile, CVTI);
			} // End Var Output

			outFile << "  </AppendedData>" << std::endl;
			outFile << "</VTKFile>" << std::endl;
			outFile.close();
		}
	}

	delete[] OutRanks;
}

void XFLUIDS::Output_plt(int rank, OutString &osr, bool error)
{
	std::string file_name = OutputDir + "/PLT_" + outputPrefix + "_Step_Time_" + osr.stepFormat.str() + "." + osr.timeFormat.str();
#ifdef USE_MPI
	file_name += "_rank_" + osr.rankFormat.str();
#endif
	file_name += ".dat";

	// Init var names
	int Onbvar = 5 + (Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ) * 2; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES;
#endif // end COP

	std::map<int, std::string> variables_names;
	int index = 0;

	if (Ss.BlSz.DimX)
		variables_names[index] = "x[m]", index++;
	if (Ss.BlSz.DimY)
		variables_names[index] = "y[m]", index++;
	if (Ss.BlSz.DimZ)
		variables_names[index] = "z[m]", index++;
	if (Ss.BlSz.DimX)
		variables_names[index] = "<i>u</i>[m/s]", index++;
	if (Ss.BlSz.DimY)
		variables_names[index] = "<i>v</i>[m/s]", index++;
	if (Ss.BlSz.DimZ)
		variables_names[index] = "<i>w</i>[m/s]", index++;

	variables_names[index] = "<i><greek>r</greek></i>[kg/m<sup>3</sup>]", index++; // rho
	variables_names[index] = "<i>p</i>[Pa]", index++;							   // pressure
	variables_names[index] = "<i>T</i>[K]", index++;							   // temperature
	variables_names[index] = "<i>c</i>[m/s]", index++;							   // sound speed
	variables_names[index] = "<i><greek>g</greek></i>[-]", index++;				   // gamma
#ifdef COP
	for (size_t ii = Onbvar - Ss.BlSz.num_species; ii < Onbvar; ii++)
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
	out << "zone t='" << outputPrefix << "_" << osr.timeFormat.str();
#ifdef USE_MPI
	out << "_rank_" << std::to_string(rank);
#endif // end USE_MPI
	out << "', i= " << VTI.nbX + Ss.BlSz.DimX << ", j= " << VTI.nbY + Ss.BlSz.DimY << ", k= " << VTI.nbZ + Ss.BlSz.DimZ
		<< "  DATAPACKING=BLOCK, VARLOCATION=([" << Ss.BlSz.DimX + Ss.BlSz.DimY + Ss.BlSz.DimZ + 1
		<< "-" << Onbvar << "]=CELLCENTERED) SOLUTIONTIME= " << osr.timeFormat.str() << "\n";

	real_t dimx = Ss.BlSz.DimX, dimy = Ss.BlSz.DimY, dimz = Ss.BlSz.DimZ;
	if (Ss.BlSz.DimX)
	{
		for (int k = VTI.minZ; k < VTI.maxZ + Ss.BlSz.DimZ; k++)
			for (int j = VTI.minY; j < VTI.maxY + Ss.BlSz.DimY; j++)
			{
				for (int i = VTI.minX; i < VTI.maxX + Ss.BlSz.DimX; i++)
					out << dimx * ((i + posx) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin) << " ";
				out << "\n";
			}
	}

	if (Ss.BlSz.DimY)
	{
		for (int k = VTI.minZ; k < VTI.maxZ + Ss.BlSz.DimZ; k++)
			for (int j = VTI.minY; j < VTI.maxY + Ss.BlSz.DimY; j++)
			{
				for (int i = VTI.minX; i < VTI.maxX + Ss.BlSz.DimX; i++)
					out << dimy * ((j + posy) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin) << " ";
				out << "\n";
			}
	}

	if (Ss.BlSz.DimZ)
	{
		for (int k = VTI.minZ; k < VTI.maxZ + Ss.BlSz.DimZ; k++)
			for (int j = VTI.minY; j < VTI.maxY + Ss.BlSz.DimY; j++)
			{
				for (int i = VTI.minX; i < VTI.maxX + Ss.BlSz.DimX; i++)
					out << dimz * ((k + posz) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin) << " ";
				out << "\n";
			}
	}

	if (Ss.BlSz.DimX)
	{
		MARCO_POUTLOOP(fluids[0]->h_fstate.u[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	}
	if (Ss.BlSz.DimY)
	{
		MARCO_POUTLOOP(fluids[0]->h_fstate.v[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	}
	if (Ss.BlSz.DimZ)
	{
		MARCO_POUTLOOP(fluids[0]->h_fstate.w[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	}

	MARCO_POUTLOOP(fluids[0]->h_fstate.rho[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.p[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.T[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.c[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);
	MARCO_POUTLOOP(fluids[0]->h_fstate.gamma[Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i]);

#ifdef COP
	for (size_t n = 0; n < Ss.BlSz.num_species; n++)
		MARCO_POUTLOOP(fluids[0]->h_fstate.y[n + NUM_SPECIES * (Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i)]);
#endif
	out.close();
}

template <typename T>
void XFLUIDS::Output_cplt(std::vector<OutVar> &varout, OutSlice &pos, OutString &osr)
{
	int Cnbvar = 16;
#ifdef COP
	Cnbvar += NUM_SPECIES;
#endif

	OutSize CPT = VTI;
	int *OutRanks = new int[nranks];
	real_t temx = _DF(0.5) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin;
	real_t temy = _DF(0.5) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin;
	real_t temz = _DF(0.5) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin;
	real_t posx = -Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.X_inner);
	real_t posy = -Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Y_inner);
	real_t posz = -Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Z_inner);
#ifdef USE_MPI
	GetCPT_OutRanks(OutRanks, CPT, pos);
	if (OutRanks[rank] >= 0)
#endif // end USE_MPI
	{
		real_t *OutPoint = new real_t[Cnbvar]; // OutPoint: each point;
		std::string file_name = OutputDir + "/CPLT_" + outputPrefix + "_Step_Time_" + osr.stepFormat.str() +
								"." + osr.timeFormat.str() + "_" + osr.rankFormat.str() + ".dat";
		std::ofstream out(file_name);
		// // defining header for tecplot(plot software)
		out << "title='" << outputPrefix << "'\nvariables=";
		if (Ss.BlSz.DimX)
			out << "x[m], ";
		if (Ss.BlSz.DimY)
			out << "y[m], ";
		if (Ss.BlSz.DimZ)
			out << "z[m], ";
		out << "<i><greek>r</greek></i>[kg/m<sup>3</sup>], <i>p</i>[Pa], <i>c</i>[m/s]";
		if (Ss.BlSz.DimX)
			out << ", <i>u</i>[m/s]";
		if (Ss.BlSz.DimY)
			out << ", <i>v</i>[m/s]";
		if (Ss.BlSz.DimZ)
			out << ", <i>w</i>[m/s]";
		out << ", <i><greek>g</greek></i>[-], <i>T</i>[K], <i>e</i>[J]";
		if (Visc && (Ss.BlSz.DimS > 1))
			out << ", <i><greek>w</greek></i>|[s<sup>-1</sup>]"
				<< ", <i><greek>w</greek></i><sub>x</sub>[s<sup>-1</sup>]"
				<< ", <i><greek>w</greek></i><sub>y</sub>[s<sup>-1</sup>]"
				<< ", <i><greek>w</greek></i><sub>z</sub>[s<sup>-1</sup>]";
#ifdef COP
		for (size_t n = 0; n < Ss.BlSz.num_species; n++)
			out << ", <i>Y(" << Ss.species_name[n] << ")</i>[-]";
#endif
		out << "\n";
		out << "zone t='" << outputPrefix << "_" << osr.timeFormat.str();
#ifdef USE_MPI
		out << "_rank_" << std::to_string(rank);
#endif // end USE_MPI
		out << "', i= " << CPT.nbX << ", j= " << CPT.nbY << ", k= " << CPT.nbZ << ", SOLUTIONTIME= " << osr.timeFormat.str() << "\n";

		for (int k = CPT.minZ; k < CPT.maxZ; k++)
			for (int j = CPT.minY; j < CPT.maxY; j++)
				for (int i = CPT.minX; i < CPT.maxX; i++)
				{ //&& Ss.OutDirX//&& Ss.OutDirY//&& Ss.OutDirZ
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					int pos_x = i + posx, pos_y = j + posy, pos_z = k + posz;
					OutPoint[0] = (Ss.BlSz.DimX) ? (pos_x)*Ss.BlSz.dx + temx : 0.0;
					OutPoint[1] = (Ss.BlSz.DimY) ? (pos_y)*Ss.BlSz.dy + temy : 0.0;
					OutPoint[2] = (Ss.BlSz.DimZ) ? (pos_z)*Ss.BlSz.dz + temz : 0.0;
					OutPoint[3] = fluids[0]->h_fstate.rho[id];
					OutPoint[4] = fluids[0]->h_fstate.p[id];
					OutPoint[5] = fluids[0]->h_fstate.c[id];
					OutPoint[6] = fluids[0]->h_fstate.u[id];
					OutPoint[7] = fluids[0]->h_fstate.v[id];
					OutPoint[8] = fluids[0]->h_fstate.w[id];
					OutPoint[9] = fluids[0]->h_fstate.gamma[id];
					OutPoint[10] = fluids[0]->h_fstate.T[id];
					OutPoint[11] = fluids[0]->h_fstate.e[id];

					if (Visc)
					{
						OutPoint[12] = sqrt(fluids[0]->h_fstate.vx[id]);
						OutPoint[13] = fluids[0]->h_fstate.vxs[0][id];
						OutPoint[14] = fluids[0]->h_fstate.vxs[1][id];
						OutPoint[15] = fluids[0]->h_fstate.vxs[2][id];
					}

#if COP
					for (int n = 0; n < Ss.BlSz.num_species; n++)
						OutPoint[Cnbvar - NUM_SPECIES + n] = fluids[0]->h_fstate.y[n + NUM_SPECIES * id];
#endif
					if (Ss.BlSz.DimX) // x
						out << OutPoint[0] << " ";
					if (Ss.BlSz.DimY) // y
						out << OutPoint[1] << " ";
					if (Ss.BlSz.DimZ) // z
						out << OutPoint[2] << " ";
					// rho, p, c
					out << OutPoint[3] << " " << OutPoint[4] << " " << OutPoint[5] << " ";

					if (Ss.BlSz.DimX) // u
						out << OutPoint[6] << " ";
					if (Ss.BlSz.DimY) // v
						out << OutPoint[7] << " ";
					if (Ss.BlSz.DimZ) // w
						out << OutPoint[8] << " ";

					out << OutPoint[9] << " " << OutPoint[10] << " " << OutPoint[11] << " "; // gamma, T, e

					if (Visc && (Ss.BlSz.DimS > 1))
						out << OutPoint[12] << " " << OutPoint[13] << " " << OutPoint[14] << " " << OutPoint[15] << " "; // Vorticity
#if COP
					for (int n = 0; n < Ss.BlSz.num_species; n++)
						out << OutPoint[Cnbvar - NUM_SPECIES + n] << " "; // Yi
#endif
					out << "\n";
				}
		out.close();
		free(OutPoint);
	}

	delete[] OutRanks;
}