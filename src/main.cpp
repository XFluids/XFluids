//-----------------------------------------------------------------------------------------------------------------------
//   This file is a main implemrntation of XFLUIDS
//-----------------------------------------------------------------------------------------------------------------------
//
//  XFLUIDS is a parallelized C++ solver for multi-component reacting flow dynamics.
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
// XFLUIDS - Large-scale Architecture-independent Multi-component reacting Navier-Strokes equation Solver
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
#ifdef USE_MPI // Create MPI session if MPI enabled
	MPI_Init(&argc, &argv);
#endif // USE_MPI

	// // Setup Initialize
	Setup setup(argc, argv);
	// // Solver Construction
	XFLUIDS solver(setup);
	// // AllocateMemory
	solver.AllocateMemory(setup.q);
	// // Initialize original states
	solver.InitialCondition(setup.q);
	// // Boundary conditions
	solver.BoundaryCondition(setup.q);
	// // Update states by U
	solver.UpdateStates(setup.q);
	// // Time marching by SYCL device
	solver.Evolution(setup.q);

#ifdef USE_MPI
	MPI_Finalize();
#endif // end USE_MPI
	return 0;
}