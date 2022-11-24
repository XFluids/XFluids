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

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string.h>
#include <float.h>
#include <ctime>

#include "setup.h"
#include "fun.h"
#include "global_class.h"


//-------------------------------------------------------------------------------------------------
//							main
//-------------------------------------------------------------------------------------------------
int main()
{
	// Logger logger;
	// logger.LogSolverInfo();

	auto device = sycl::platform::get_platforms()[2].get_devices()[0];
	// accelerator_selector device;
	queue q(device, dpc_common::exception_handler);

	SYCLSolver syclsolver(q);
	syclsolver.AllocateMemory(q);
	syclsolver.InitialCondition(q);
	// //  boundary conditions
	// // syclsolver.BoundaryCondition(0);
	syclsolver.CopyDataFromDevice(q);
	syclsolver.Output(0);

	return 0;
}