//-----------------------------------------------------------------------------------------------------------------------
//   This file is a main implemrntation of LAMNSS
//-----------------------------------------------------------------------------------------------------------------------
//
//  LAMNSS is a parallelized C++ solver for multi-component reacting flow dynamics.
//  It allows for large-scale high-resolution sharp-interface modeling  of both incompressible
//  and compressible multiphase flows.
//
// This code is developed by Mr.Li from Prof. S Pan's group at the School of Aeronautics,
//  Northwestern Polytechincal University.
//
//-----------------------------------------------------------------------------------------------------------------------
//
// LICENSE
//
// LAMNSS - Large-scale Architecture-independent Multi-component reacting Navier-Strokes equation Solver
// Copyright (C) 2023 JinLing Li and contributors (see AUTHORS list)
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation version 3.
//
// CONTACT:
// ljl66623@mail.nwpu.edu.cn
//
//  Xi'an, China, August 20th, 2023
//----------------------------------------------------------------------------------------------------------------------
#include "global_class.h"

int main(int argc, char *argv[])
{
#ifdef USE_MPI
	// Create MPI session if MPI enabled
	int rank, nRanks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#else
	int rank = 0;
	int nRanks = 1;
#endif // USE_MPI
	std::string ini_path;
	if (argc < 2)
		ini_path = std::string(IniFile);
	else if (argc == 2)
		ini_path = std::string(argv[1]);
	else
		std::cout << "Too much argcs appended to EulerSYCL while running\n";

	ConfigMap configMap = broadcast_parameters(ini_path);
	// // accelerator_selector device;
	// // num_GPUS:number of GPU on this cluster, Pform_id: the first GPU's ID in all accelerators sycl detected
	int num_GPUs = configMap.getInteger("mpi", "NUM", 1);
	int device_id = rank % num_GPUs + Pform_id;
	auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
	sycl::queue q(device);
	// // Setup Initialize
	Setup setup(configMap, q);
	LAMNSS solver(setup);
	// // AllocateMemory
	solver.AllocateMemory(q);
	// //  Initialize original states
	solver.InitialCondition(q);
	// // boundary conditions
	solver.BoundaryCondition(q, 0);
	// // update states by U
	// std::cout << "1";
	solver.UpdateStates(q, 0, solver.physicalTime, solver.Iteration, "_Ini");
	// // time marching by SYCL device
	solver.Evolution(q);

#ifdef USE_MPI
	MPI_Finalize();
#endif // end USE_MPI
	return 0;
}