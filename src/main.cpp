//-----------------------------------------------------------------------------------------------------------------------
//   This file is part of PANDOLINS
//-----------------------------------------------------------------------------------------------------------------------
//  						    	   __^    ^
//  							      \.O        _\
// 							/|      <              _\
// 							/|      <              _\
//  						/|^ ^/               /
// 							/PANDOLINS\
// 							\/   \/  \/   \/
//
//  PANDOLINS is a MPI/CUDA/ROCm-parallelized C++ solver for interfacial-flow dynamics.
//  It allows for large-scale high-resolution sharp-interface modeling  of both incompressible
//  and compressible multiphase flows.
//
// This code is developed by Prof. S Pan's group at the School of Aeronautics,
//  Northwestern Polytechincal University.
//-----------------------------------------------------------------------------------------------------------------------
//
// LICENSE
//
// PANDOLINS - PArallel aND GPUs-oriented Large-scale INterfacial-flow Solvers
// Copyright (C) 2021 Shucheng Pan and contributors (see AUTHORS list)
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation version 3.
//
// CONTACT:
// shcuheng.pan@nwpu.edu.cn
//
//  Xi'an, China, June 20th, 2022
//----------------------------------------------------------------------------------------------------------------------

#include "include/global_class.h"

//-------------------------------------------------------------------------------------------------
//							main
//-------------------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	// Create MPI session if MPI enabled
#ifdef USE_MPI
	// MPI_Init(&argc, &argv);
	mpiUtils::GlobalMpiSession mpiSession(&argc, &argv); //
	int rank, nRanks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#else
	int rank = 0;
	int nRanks = 1;
#endif // USE_MPI
	std::string input_flie = std::string(argv[1]);
	ConfigMap configMap = broadcast_parameters(input_flie);
	// accelerator_selector device;
	auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
	sycl::queue q(device);
	Setup setup(configMap, q);

	SYCLSolver syclsolver(q, setup);
	syclsolver.AllocateMemory(q);
	syclsolver.InitialCondition(q);
	// boundary conditions
	syclsolver.BoundaryCondition(q, 0);
	// // update states by U
	syclsolver.UpdateStates(q, 0);
	// test updateflux
	// syclsolver.dt = 0.00001;
	// syclsolver.SinglePhaseSolverRK3rd(q);
	// time marching by SYCL device
	syclsolver.Evolution(q);
	return 0;
}