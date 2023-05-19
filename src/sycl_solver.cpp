#include "global_class.h"

SYCLSolver::SYCLSolver(Setup &setup) : Ss(setup), dt(_DF(0.0))
{
	for (int n = 0; n < NumFluid; n++)
	{
		fluids[n] = new FluidSYCL(setup);
#if 1 != NumFluid
		fluids[n]->initialize(n);
#endif
	}
	if (Ss.OutBoundary)
	{
		OnbX = Ss.BlSz.Xmax;
		OminX = 0;
		OmaxX = Ss.BlSz.Xmax;

		OnbY = Ss.BlSz.Ymax;
		OminY = 0;
		OmaxY = Ss.BlSz.Ymax;

		OnbZ = Ss.BlSz.Zmax;
		OminZ = 0;
		OmaxZ = Ss.BlSz.Zmax;
	}
	else
	{
		OnbX = Ss.BlSz.X_inner;
		OminX = Ss.BlSz.Bwidth_X;
		OmaxX = Ss.BlSz.Xmax - Ss.BlSz.Bwidth_X;

		OnbY = Ss.BlSz.Y_inner;
		OminY = Ss.BlSz.Bwidth_Y;
		OmaxY = Ss.BlSz.Ymax - Ss.BlSz.Bwidth_Y;

		OnbZ = Ss.BlSz.Z_inner;
		OminZ = Ss.BlSz.Bwidth_Z;
		OmaxZ = Ss.BlSz.Zmax - Ss.BlSz.Bwidth_Z;
	}
}

void SYCLSolver::Evolution(sycl::queue &q)
{
	real_t physicalTime = 0.0;
	int Iteration = 0, OutNum = 1, rank = 0, TimeLoop = 0, nranks = 1;
	bool error_out = false, Stepstop = false;
	dt = 0;
#if USE_MPI
	rank = Ss.mpiTrans->myRank;
	nranks = Ss.mpiTrans->nProcs;
#endif // end USE_MPI

	float duration = 0.0f;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	while (TimeLoop < Ss.nOutTimeStamps)
	{
		real_t target_t = Ss.OutTimeStamps[TimeLoop]; // std::max(Ss.OutTimeStamp, TimeLoop * Ss.OutTimeStamp + Ss.OutTimeStart);
		TimeLoop++;
		while (physicalTime < target_t)
		{
			if (TimeLoop && Iteration % Ss.OutInterval == 0 && OutNum <= Ss.nOutput)
			{
				Output(q, rank, std::to_string(Iteration), physicalTime);
				OutNum++;
			}
			else if (error_out)
			{
				std::cout << "Errors Captured." << std::endl;
				Output(q, rank, "_error", physicalTime);
				std::exit(EXIT_FAILURE);
			}

			// get minmum dt, if MPI used, get the minimum of all ranks
#ifdef DEBUG
				// std::cout << "  sleep before ComputeTimeStep\n";
				// sleep(5);
#endif								 // end DEBUG
			dt = ComputeTimeStep(q); // 2.0e-6; //
#ifdef DEBUG
			// std::cout << "  sleep after ComputeTimeStep\n";
			// sleep(5);
#endif // end DEBUG
			if (rank == 0) // An iteration begins at the physicalTime output on screen and ends at physicalTime + dt, which is the physicalTime of the next iteration
				std::cout << "N=" << std::setw(7) << Iteration + 1 << "     beginning physicalTime: " << std::setw(16) << std::setprecision(8) << physicalTime;
#ifdef USE_MPI
			Ss.mpiTrans->communicator->synchronize();
			real_t temp;
			Ss.mpiTrans->communicator->allReduce(&dt, &temp, 1, Ss.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
			dt = temp;
			if (rank == 0)
				std::cout << "     mpi communicated";
#endif // end USE_MPI
			if (physicalTime + dt > target_t)
				dt = target_t - physicalTime;
			if (rank == 0)
				std::cout << "   dt: " << dt << " to do. \n";

			// solved the fluid with 3rd order Runge-Kutta method
			error_out = SinglePhaseSolverRK3rd(q);

#ifdef COP_CHEME
			Reaction(q, dt);
#endif // end COP_CHEME

			physicalTime += dt;
			Iteration++;
			Stepstop = Ss.nStepmax <= Iteration ? true : false; /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/
			if (Stepstop)
				break;
		}
			Output(q, rank, std::to_string(Iteration), physicalTime);
			if (Stepstop)
			break;
	}

	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<float, std::milli>(end_time - start_time).count() / 1000.0f;
#ifdef USE_MPI
	Ss.mpiTrans->communicator->synchronize();
	float Ttemp;
	Ss.mpiTrans->communicator->allReduce(&duration, &Ttemp, 1, mpiUtils::MpiComm::FLOAT, mpiUtils::MpiComm::SUM);
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
		std::cout << SelectDv << " runtime: " << std::setw(8) << std::setprecision(6) << duration / float(nranks) << std::endl;
		// std::cout << SelectDv << " runtime: " << std::setw(8) << std::setprecision(6) << duration  << " of rank: " << rank << std::endl;
#ifdef USE_MPI
	}
	Ss.mpiTrans->communicator->synchronize();
#endif
	// sleep(5);
}

bool SYCLSolver::SinglePhaseSolverRK3rd(sycl::queue &q)
{
	RungeKuttaSP3rd(q, 1);
	RungeKuttaSP3rd(q, 2);
	RungeKuttaSP3rd(q, 3);

	// estimate if rho is_nan or <0 or is_inf
	bool error = false;
#ifdef ESTIM_NAN
	for (int n = 0; n < NumFluid; n++)
	{
		if (fluids[n]->EstimateFluidNAN(q, 0))
		{
			error = true;
			break;
		}
	}
#endif // end
	return error;
}

void SYCLSolver::RungeKuttaSP3rd(sycl::queue &q, int flag)
{
	switch (flag)
	{
	case 1:
		// the fisrt step
		BoundaryCondition(q, 0);
		UpdateStates(q, 0);
		ComputeLU(q, 0);
		UpdateU(q, 1);
		break;
	case 2:
		// the second step
		BoundaryCondition(q, 1);
		UpdateStates(q, 1);
		ComputeLU(q, 1);
		UpdateU(q, 2);
		break;
	case 3:
		// the third step
		BoundaryCondition(q, 1);
		UpdateStates(q, 1);
		ComputeLU(q, 1);
		UpdateU(q, 3);
		break;
	}
}

real_t SYCLSolver::ComputeTimeStep(sycl::queue &q)
{
	real_t dt_ref = _DF(1.0e-10);
#if NumFluid == 1
	dt_ref = fluids[0]->GetFluidDt(q);
#elif NumFluid == 2
	dt_ref = fluids[0]->GetFluidDt(levelset);
	dt_ref = min(dt_ref, fluids[1]->GetFluidDt(levelset));
#endif

	return dt_ref;
}

void SYCLSolver::ComputeLU(sycl::queue &q, int flag)
{
	fluids[0]->ComputeFluidLU(q, flag);
}

void SYCLSolver::UpdateU(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->UpdateFluidURK3(q, flag, dt);
}

void SYCLSolver::BoundaryCondition(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->BoundaryCondition(q, Ss.Boundarys, flag);
}

void SYCLSolver::UpdateStates(sycl::queue &q, int flag)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->UpdateFluidStates(q, flag);
}

void SYCLSolver::AllocateMemory(sycl::queue &q)
{
	d_BCs = static_cast<BConditions *>(malloc_device(6 * sizeof(BConditions), q));

	q.memcpy(d_BCs, Ss.Boundarys, 6 * sizeof(BConditions));

	// host arrays for each fluid
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->AllocateFluidMemory(q);

	// levelset->AllocateLSMemory();
#ifdef DEBUG
	std::cout << "Debug version causes increase of global memory in GetDt.\n\n";
#endif // end DEBUG
}

void SYCLSolver::InitialCondition(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->InitialU(q);
}

#ifdef COP_CHEME
void SYCLSolver::Reaction(sycl::queue &q, real_t Time)
{
	UpdateStates(q, 0);
	fluids[0]->ODESolver(q, Time);
	BoundaryCondition(q, 0);
	UpdateStates(q, 0);
}
#endif // end COP_CHEME

void SYCLSolver::CopyDataFromDevice(sycl::queue &q)
{
	// copy mem from device to host
	int bytes = Ss.bytes;
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
		q.memcpy(fluids[n]->h_fstate.gamma, fluids[n]->d_fstate.gamma, bytes);
#ifdef COP
		// for (size_t i = 0; i < NUM_SPECIES; i++)
		// 	q.memcpy(fluids[n]->h_fstate.y[i], fluids[n]->d_fstate.y[i], bytes);
		q.memcpy(fluids[n]->h_fstate.y, fluids[n]->d_fstate.y, bytes * NUM_SPECIES);
#endif // COP
	}
	q.wait();
}

void SYCLSolver::Output(sycl::queue &q, int rank, std::string interation, real_t Time)
{
	real_t Itime = Time;
	// Write time in string timeFormat
	std::ostringstream timeFormat;
	timeFormat.width(11);
	timeFormat.fill('0');
	timeFormat << Itime;
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

#ifdef OUT_PLT
	Output_plt(q, rank, timeFormat, stepFormat, rankFormat);
#else
	Output_vti(q, rank, timeFormat, stepFormat, rankFormat);
#endif // end OUT_PLT

	if (rank == 0)
		std::cout << "Output has been done at Step = " << interation << std::endl;
}

void SYCLSolver::Output_vti(sycl::queue &q, int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat)
{
	CopyDataFromDevice(q); // only copy when output

	int xmin, ymin, xmax, ymax, zmin, zmax, mx, my, mz;
	real_t dx, dy, dz;

	xmin = DIM_X ? Ss.BlSz.myMpiPos_x * OnbX : 0;
	xmax = DIM_X ? Ss.BlSz.myMpiPos_x * OnbX + OnbX : 0;
	ymin = DIM_Y ? Ss.BlSz.myMpiPos_y * OnbY : 0;
	ymax = DIM_Y ? Ss.BlSz.myMpiPos_y * OnbY + OnbY : 0;
	zmin = DIM_Z ? (Ss.BlSz.myMpiPos_z * OnbZ) : 0;
	zmax = DIM_Z ? (Ss.BlSz.myMpiPos_z * OnbZ + OnbZ) : 0;
	dx = DIM_X ? Ss.BlSz.dx : 0.0;
	dy = DIM_Y ? Ss.BlSz.dy : 0.0;
	dz = DIM_Z ? Ss.BlSz.dz : 0.0;
	// Init var names
	int Onbvar = 5 + (DIM_X + DIM_Y + DIM_Z) * 2; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES;
#endif // end COP
	std::map<int, std::string> variables_names;
	int index = 0;
#if DIM_X
	variables_names[index] = "DIR-X";
	index++;
	variables_names[index] = "V-u";
	index++;
#endif // end DIM_X
#if DIM_Y
	variables_names[index] = "DIR-Y";
	index++;
	variables_names[index] = "V-v";
	index++;
#endif // end DIM_Y
#if DIM_Z
	variables_names[index] = "DIR-Z";
	index++;
	variables_names[index] = "V-w";
	index++;
#endif // end DIM_Z

	variables_names[index] = "V-c";
	index++;
	variables_names[index] = "rho";
	index++;
	variables_names[index] = "P";
	index++;
	variables_names[index] = "Gamma";
	index++;
	variables_names[index] = "T";
	index++;
#if 1 != NumFluid
	Onbar += NumFluid - 1;
	variables_names[index] = "phi";
	index++;
#endif
#ifdef COP
	for (size_t ii = Onbvar - NUM_SPECIES; ii < Onbvar; ii++)
	{
		variables_names[ii] = "y" + std::to_string(ii - Onbvar + NUM_SPECIES) + "_" + Ss.species_name[ii - Onbvar + NUM_SPECIES];
	}
#endif // COP

	std::string file_name;
	std::string headerfile_name;
	std::string outputPrefix = INI_SAMPLE;
#ifdef USE_MPI
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + "_Time_" + timeFormat.str() + "_rank_" + rankFormat.str() + ".vti"; //"_Time" + timeFormat.str() +
	headerfile_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + ".pvti";
#else
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + "_Time_" + timeFormat.str() + ".vti";
#endif
#ifdef USE_MPI
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
		outHeader << 0 << " " << mx * OnbX << " ";
		outHeader << 0 << " " << my * OnbY << " ";
		outHeader << 0 << " " << mz * OnbZ << "\" GhostLevel=\"0\" "
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
			std::string pieceFilename = "./" + outputPrefix + "_Step_" + stepFormat.str() + "_Time_" + timeFormat.str() + "_rank_" + pieceFormat.str() + ".vti";
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
				outHeader << 0 << " " << OnbX << " ";
			else
				outHeader << coords[0] * OnbX << " " << coords[0] * OnbX + OnbX << " ";

			if (coords[1] == 0)
				outHeader << 0 << " " << OnbY << " ";
			else
				outHeader << coords[1] * OnbY << " " << coords[1] * OnbY + OnbY << " ";
#if 3 == DIM_X + DIM_Y + DIM_Z
			if (coords[2] == 0)
				outHeader << 0 << " " << OnbZ << " ";
			else
				outHeader << coords[2] * OnbZ << " " << coords[2] * OnbZ + OnbZ << " ";
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
				<< iVar * OnbX * OnbY * OnbZ * sizeof(real_t) + iVar * sizeof(unsigned int)
				<< "\" />" << std::endl;
	}
	outFile << "    </CellData>" << std::endl;
	outFile << "  </Piece>" << std::endl;
	outFile << "  </ImageData>" << std::endl;
	outFile << "  <AppendedData encoding=\"raw\">" << std::endl;
	// write the leading undescore
	outFile << "_";
	// then write heavy data (column major format)
	unsigned int nbOfWords = OnbX * OnbY * OnbZ * sizeof(real_t);
	{
		//[0]x
#if DIM_X
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_X ? (i - Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.Xmax - Ss.BlSz.Bwidth_X - Ss.BlSz.Bwidth_X)) * dx + 0.5 * dx + Ss.BlSz.Domain_xmin : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		  //[1]u
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.u[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.u[id] : fluids[1]->h_fstate.u[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
#endif			  // end DIM_X
#if DIM_Y
		//[2]y
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_Y ? (j - Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Ymax - Ss.BlSz.Bwidth_Y - Ss.BlSz.Bwidth_Y)) * dy + 0.5 * dy + Ss.BlSz.Domain_ymin : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		  //[3]v
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.v[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.v[id] : fluids[1]->h_fstate.v[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
#endif			  // end DIM_Y
#if DIM_Z
		//[4]z
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_Z ? (k - Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Zmax - Ss.BlSz.Bwidth_Z - Ss.BlSz.Bwidth_Z)) * dz + 0.5 * dz + Ss.BlSz.Domain_zmin : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		  //[5]w
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.w[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.w[id] : fluids[1]->h_fstate.w[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
#endif			  // end DIM_Z
		//[6]rho
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.c[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.c[id] : fluids[1]->h_fstate.c[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[6]rho
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.rho[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.rho[id] : fluids[1]->h_fstate.rho[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[7]P
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.p[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.p[id] : fluids[1]->h_fstate.p[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
				  //[8]Gamma
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.gamma[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.gamma[id] : fluids[1]->h_fstate.gamma[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[9]T
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
#if 1 == NumFluid
					real_t tmp = fluids[0]->h_fstate.T[id];
#elif 2 == NumFluid
					real_t tmp = (levelset->h_phi[id] >= 0.0) ? fluids[0]->h_fstate.T[id] : fluids[1]->h_fstate.T[id];
#endif
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k

#if 2 == NumFluid
		  //[5]phi
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
					real_t tmp = levelset->h_phi[id];
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
#endif

#ifdef COP
		//[COP]yii
		for (int ii = 0; ii < NUM_SPECIES; ii++)
		{
			outFile.write((char *)&nbOfWords, sizeof(unsigned int));
			for (int k = OminZ; k < OmaxZ; k++)
			{
				for (int j = OminY; j < OmaxY; j++)
				{
					for (int i = OminX; i < OmaxX; i++)
					{
						int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;
						real_t tmp = fluids[0]->h_fstate.y[ii + NUM_SPECIES * id]; // h_fstate.y[ii][id];
						outFile.write((char *)&tmp, sizeof(real_t));
					} // for i
				}	  // for j
			}		  // for k
		}			  // for yii
#endif				  // end  COP
	} // End Var Output
	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	outFile.close();
}

void SYCLSolver::Output_plt(sycl::queue &q, int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat)
{
	CopyDataFromDevice(q); // only copy when output

	std::string file_name;
	std::string outputPrefix = INI_SAMPLE;
#ifdef USE_MPI
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + "_Time_" + timeFormat.str() + "_rank_" + rankFormat.str() + ".plt";
#else
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + "_Time_" + timeFormat.str() + ".plt";
#endif

	std::ofstream out(file_name);
	// defining header for tecplot(plot software)
	out << "title='View'"
		<< "\n"
		<< "variables="
#if DIM_X
		<< "x, "
#endif
#if DIM_Y
		<< "y, "
#endif
#if DIM_Z
		<< "z, "
#endif
#if DIM_X
		<< "u, "
#endif
#if DIM_Y
		<< "v, "
#endif
#if DIM_Z
		<< "w, "
#endif
		// << "c, "
		<< "p, rho, ";
#ifdef COP
	out << "gamma, T"; //
	for (size_t n = 0; n < NUM_SPECIES; n++)
		out << ", Y(" << Ss.species_name[n] << ")";
#endif
	out << "\n";
	out << "zone t='Step_" << stepFormat.str() << "_Time_" << timeFormat.str() << "', i= " << OnbX << ", j= " << OnbY << ", k= " << OnbZ << ", SOLUTIONTIME= " << timeFormat.str() << "\n";

	real_t offset_x = DIM_X ? -Ss.BlSz.Bwidth_X + 0.5 : 0.0;
	real_t offset_y = DIM_Y ? -Ss.BlSz.Bwidth_Y + 0.5 : 0.0;
	real_t offset_z = DIM_Z ? -Ss.BlSz.Bwidth_Z + 0.5 : 0.0;

#ifdef DEBUG
	out.precision(20);
#endif // end DEBUG
	real_t xc, yc, zc, cc, pc, rhoc, uc, vc, wc;
	for (int k = OminZ; k < OmaxZ; k++)
		for (int j = OminY; j < OmaxY; j++)
			for (int i = OminX; i < OmaxX; i++)
			{
				xc = (i + offset_x) * Ss.BlSz.dx + Ss.BlSz.Domain_xmin;
				yc = (j + offset_y) * Ss.BlSz.dy + Ss.BlSz.Domain_ymin;
				zc = (k + offset_z) * Ss.BlSz.dz + Ss.BlSz.Domain_zmin;

				int id = Ss.BlSz.Xmax * Ss.BlSz.Ymax * k + Ss.BlSz.Xmax * j + i;

				cc = fluids[0]->h_fstate.c[id];
				pc = fluids[0]->h_fstate.p[id];
				rhoc = fluids[0]->h_fstate.rho[id];
				uc = fluids[0]->h_fstate.u[id];
				vc = fluids[0]->h_fstate.v[id];
				wc = fluids[0]->h_fstate.w[id];
				out
#if DIM_X
					<< xc << " "
#endif
#if DIM_Y
					<< yc << " "
#endif // end DIM_Y
#if DIM_Z
					<< zc << " "
#endif // end DIM_Z
#if DIM_X
					<< uc << " "
#endif
#if DIM_Y
					<< vc << " "
#endif // end DIM_Y
#if DIM_Z
					<< wc << " "
#endif // end DIM_Z
	   // << cc << " "
					<< pc << " " << rhoc << " ";
#if COP
				out << fluids[0]->h_fstate.gamma[id] << " " << fluids[0]->h_fstate.T[id]; //
				for (int n = 0; n < NUM_SPECIES; n++)
					out << " " << fluids[0]->h_fstate.y[n + NUM_SPECIES * id]; // h_fstate.y[n][id];
#endif
				out << "\n";
			}
	out.close();
}
