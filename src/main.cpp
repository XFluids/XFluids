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
#ifdef USE_MPI
	int rank, nRanks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#else
	int rank = 0;
	int nRanks = 1;
#endif // USE_MPI
	ConfigMap configMap = broadcast_parameters(std::string(argv[1]));
	// num_GPUS:number of GPU on this cluster, Pform_id: the first GPU's ID in all accelerators sycl detected
	int device_id = rank % num_GPUs + Pform_id;
	// accelerator_selector device;
	auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
	sycl::queue q(device);
	Setup setup(configMap, q);
	// Create MPI session if MPI enabled
	SYCLSolver syclsolver(q, setup);
	syclsolver.AllocateMemory(q);
	syclsolver.InitialCondition(q);
	syclsolver.Output_vti(q, 0, 100, 0);
	// boundary conditions
	syclsolver.BoundaryCondition(q, 0);
	syclsolver.Output_vti(q, 0, 200, 0);
	// // update states by U
	syclsolver.UpdateStates(q, 0);
	syclsolver.Output_vti(q, 0, 300, 0);
	// time marching by SYCL device
	syclsolver.Evolution(q);
	return 0;
}