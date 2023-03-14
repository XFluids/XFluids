#include "include/global_class.h"

SYCLSolver::SYCLSolver(sycl::queue &q, Setup &setup) : Ss(setup), dt(setup.dt)
{
	// Display the information of fluid materials
	for (int n = 0; n < NumFluid; n++)
	{
		fluids[n] = new FluidSYCL(setup);
		fluids[n]->initialize(n);
	}
}

void SYCLSolver::Evolution(sycl::queue &q)
{
	real_t physicalTime = 0.0;
	int Iteration = 0;
	int OutNum = 1;
	int rank = 0;
#if USE_MPI
	rank = Ss.mpiTrans->myRank;
#endif // end USE_MPI
	float duration = 0.0f;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	while (physicalTime < Ss.EndTime)
	{
		cout << "N=" << std::setw(6) << Iteration << " physicalTime: " << std::setw(10) << std::setprecision(8) << physicalTime << "	dt: " << dt << "\n";
		if (Iteration % Ss.OutInterval == 0 && OutNum <= Ss.nOutput)
		{
			Output_vti(q, rank, Iteration, physicalTime); // Output(q, physicalTime); //
			OutNum++;
		}
		if (Iteration == Ss.nStepmax)
			break;
		// get minmum dt, if MPI used, get the minimum of all ranks
		dt = 3.0e-6; // ComputeTimeStep(q); // debug for viscous flux // 5.0e-5;//0.001;//
#ifdef USE_MPI
		Ss.mpiTrans->communicator->synchronize();
		real_t temp;
		Ss.mpiTrans->communicator->allReduce(&dt, &temp, 1, Ss.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
		dt = temp;
#endif // end USE_MPI
		if (physicalTime + dt > Ss.EndTime)
			dt = Ss.EndTime - physicalTime;
		// solved the fluid with 3rd order Runge-Kutta method
		SinglePhaseSolverRK3rd(q);

#ifdef COP
#ifdef React
		// Reaction(q, dt); // TODO: without Reaction
#endif // React
#endif // COP
		physicalTime = physicalTime + dt;
		Iteration++;
	}
	Output_vti(q, rank, Iteration, physicalTime); // Output(q, physicalTime); //

	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<float, std::milli>(end_time - start_time).count();
#ifdef USE_MPI
	Ss.mpiTrans->communicator->synchronize();
	real_t temp;
	Ss.mpiTrans->communicator->allReduce(&duration, &temp, 1, Ss.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
	duration = temp;
	if (0 == Ss.mpiTrans->myRank)
#endif // end USE_MPI
		std::cout << SelectDv << " runtime(max of all ranks if USE_MPI): " << duration / 1000.0f << std::endl;
}

void SYCLSolver::SinglePhaseSolverRK3rd(sycl::queue &q)
{
	RungeKuttaSP3rd(q, 1);
	RungeKuttaSP3rd(q, 2);
	RungeKuttaSP3rd(q, 3);
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
	real_t dt_ref = _DF(0.0f);
#if NumFluid == 1
	dt_ref = fluids[0]->GetFluidDt(q);
#elif NumFluid == 2
// dt_ref = fluids[0]->GetFluidDt(levelset);
// dt_ref = min(dt_ref, fluids[1]->GetFluidDt(levelset));
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
	{
		fluids[n]->UpdateFluidStates(q, flag);
	}
}

void SYCLSolver::AllocateMemory(sycl::queue &q)
{
	d_BCs = static_cast<BConditions *>(malloc_device(6 * sizeof(BConditions), q));

	q.memcpy(d_BCs, Ss.Boundarys, 6 * sizeof(BConditions)).wait();

	// host arrays for each fluid
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->AllocateFluidMemory(q);

	// levelset->AllocateLSMemory();
}

void SYCLSolver::InitialCondition(sycl::queue &q)
{
	for (int n = 0; n < NumFluid; n++)
		fluids[n]->InitialU(q);
}

#ifdef COP
#ifdef React
void SYCLSolver::Reaction(sycl::queue &q, real_t Time)
{
	UpdateStates(q, 0);
	fluids[0]->ODESolver(q, Time);
	BoundaryCondition(q, 0);
	UpdateStates(q, 0);
}
#endif // React
#endif // COP

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
#ifdef COP
		for (size_t i = 0; i < NUM_SPECIES; i++)
		{
			q.memcpy(fluids[n]->h_fstate.y[i], fluids[n]->d_fstate.y[i], bytes);
		}
		q.memcpy(fluids[n]->h_fstate.T, fluids[n]->d_fstate.T, bytes);
#endif // COP
	}
	q.wait();
}

void SYCLSolver::Output_vti(sycl::queue &q, int rank, int interation, real_t Time)
{
	CopyDataFromDevice(q); // only copy when output
	real_t Itime = Time * 1.0e8;
	// Write time in string timeFormat
	std::ostringstream timeFormat;
	timeFormat.width(4);
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

	int xmin, ymin, xmax, ymax, zmin, zmax, mx, my, mz;
	int OnbX, OnbY, OnbZ, OminX, OminY, OminZ, OmaxX, OmaxY, OmaxZ;
	real_t dx, dy, dz;

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

		OnbZ = Ss.BlSz.Zmax;
		OminZ = Ss.BlSz.Bwidth_Z;
		OmaxZ = Ss.BlSz.Zmax - Ss.BlSz.Bwidth_Z;
	}
	xmin = Ss.BlSz.myMpiPos_x * OnbX;
	xmax = Ss.BlSz.myMpiPos_x * OnbX + OnbX;
	dx = Ss.BlSz.dx;
	ymin = Ss.BlSz.myMpiPos_y * OnbY;
	ymax = Ss.BlSz.myMpiPos_y * OnbY + OnbY;
	dy = Ss.BlSz.dy;
	zmin = Ss.BlSz.myMpiPos_z * OnbZ;
	zmax = Ss.BlSz.myMpiPos_z * OnbZ + OnbZ;
	dz = Ss.BlSz.dz;

	// Init var names
	int Onbvar = 8; // one fluid no COP
#ifdef COP
	Onbvar += NUM_SPECIES + 1;
#endif // end COP
#if 1 != NumFluid
	Onbar += NumFluid - 1;
#endif // end Multi
	std::map<int, std::string> variables_names;
	variables_names[0] = "DIR-X";
	variables_names[1] = "DIR-Y";
	variables_names[2] = "DIR-Z";
	variables_names[3] = "rho";
	variables_names[4] = "P";
	variables_names[5] = "u";
	variables_names[6] = "v";
	variables_names[7] = "w";
#ifdef COP
	variables_names[8] = "T";
	for (size_t ii = 9; ii < Onbvar; ii++)
	{
		variables_names[ii] = "y" + std::to_string(ii - 9) + "_" + Ss.species_name[ii - 9];
	}
#endif // COP
#if 2 == NumFluid
	variables_names[5] = "phi";
#endif

	std::string file_name;
	std::string headerfile_name;
	std::string outputPrefix = "FlowField";
#ifdef USE_MPI
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + "_mpi_" + rankFormat.str() + ".vti"; //"_Time" + timeFormat.str() +
	headerfile_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + ".pvti";
	// std::cout << file_name << "of rank " << rank << std::endl;
#else
	file_name = Ss.OutputDir + "/" + outputPrefix + "_Step_" + stepFormat.str() + ".vti";
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
				  << Ss.Domain_xmin << " " << Ss.Domain_ymin << " " << Ss.Domain_zmin << "\" "
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
			std::string pieceFilename = "./" + outputPrefix + "_Step_" + stepFormat.str() + "_mpi_" + pieceFormat.str() + ".vti";
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
			<< Ss.Domain_xmin << " " << Ss.Domain_ymin << " " << Ss.Domain_zmin << "\" "
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
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_X ? (i - Ss.BlSz.Bwidth_X + Ss.BlSz.myMpiPos_x * (Ss.BlSz.Xmax - Ss.BlSz.Bwidth_X - Ss.BlSz.Bwidth_X)) * dx + 0.5 * dx : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[1]y
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_Y ? (j - Ss.BlSz.Bwidth_Y + Ss.BlSz.myMpiPos_y * (Ss.BlSz.Ymax - Ss.BlSz.Bwidth_Y - Ss.BlSz.Bwidth_Y)) * dy + 0.5 * dy : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[2]z
		outFile.write((char *)&nbOfWords, sizeof(unsigned int));
		for (int k = OminZ; k < OmaxZ; k++)
		{
			for (int j = OminY; j < OmaxY; j++)
			{
				for (int i = OminX; i < OmaxX; i++)
				{
					real_t tmp = DIM_Z ? (k - Ss.BlSz.Bwidth_Z + Ss.BlSz.myMpiPos_z * (Ss.BlSz.Zmax - Ss.BlSz.Bwidth_Z - Ss.BlSz.Bwidth_Z)) * dz + 0.5 * dz : 0.0;
					outFile.write((char *)&tmp, sizeof(real_t));
				} // for i
			}	  // for j
		}		  // for k
		//[3]rho
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
		//[4]P
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
		//[5]u
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
		//[6]v
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
		//[7]w
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

#ifdef COP
		//[8]T
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
						real_t tmp = fluids[0]->h_fstate.y[ii][id];
						outFile.write((char *)&tmp, sizeof(real_t));
					} // for i
				}	  // for j
			}		  // for k
		}			  // for yii
#endif				  // end  COP

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
	} // End Var Output
	outFile << "  </AppendedData>" << std::endl;
	outFile << "</VTKFile>" << std::endl;
	outFile.close();
	std::cout << "Output of rank: " << rank << " has been done at Step = " << interation << std::endl;
}

void SYCLSolver::Output(sycl::queue &q, real_t Time)
{
	CopyDataFromDevice(q); // only copy when output
	int Xmax = Ss.BlSz.Xmax;
	int Ymax = Ss.BlSz.Ymax;
	int Zmax = Ss.BlSz.Zmax;
	int X_inner = Ss.BlSz.X_inner;
	int Y_inner = Ss.BlSz.Y_inner;
	int Z_inner = Ss.BlSz.Z_inner;
	int Bwidth_X = Ss.BlSz.Bwidth_X;
	int Bwidth_Y = Ss.BlSz.Bwidth_Y;
	int Bwidth_Z = Ss.BlSz.Bwidth_Z;
	real_t dx = Ss.BlSz.dx;
	real_t dy = Ss.BlSz.dy;
	real_t dz = Ss.BlSz.dz;

	real_t Itime;
	char file_name[50], file_list[50];

	// produce output file name
	Itime = Time * 1.0e6;
	strcpy(file_name, "./outdata/flowfield_");
	sprintf(file_list, "%d", (int)Itime);
	strcat(file_name, file_list);
	strcat(file_name, ".plt");
	ofstream out(file_name);
	// defining header for tecplot(plot software)
	out << "title='View'"
		<< "\n";
	int LEN = 0;
#if (DIM_X + DIM_Y + DIM_Z == 1)
	LEN = 2;
	out << "variables=x, u, p, rho"
		<< "\n";
#elif (DIM_X + DIM_Y + DIM_Z == 2)
	LEN = 2;
	out << "variables=x, y, u, v, p, rho"
		<< "\n";
#elif (DIM_X + DIM_Y + DIM_Z == 3)
	LEN = 2;
	out << "variables=x, y, z, u, v, w, p, rho"
		<< "\n";
#endif
	out << "zone t='filed', i=" << X_inner + DIM_X << ", j=" << Y_inner + DIM_Y << ", k=" << Z_inner + DIM_Z << "  DATAPACKING=BLOCK, VARLOCATION=([";
	int pos_s = DIM_X + DIM_Y + DIM_Z + 1;
	out << pos_s << "-";
	out << 2 * pos_s - 1 + LEN - 1 << "]=CELLCENTERED) SOLUTIONTIME=" << Time << "\n";

	int ii = Xmax - Bwidth_X + DIM_X - 1;
	int jj = Ymax - Bwidth_Y + DIM_Y - 1;
	int kk = Zmax - Bwidth_Z + DIM_Z - 1;

#if DIM_X
	for (int k = Bwidth_Z; k <= kk; k++)
	{
		for (int j = Bwidth_Y; j <= jj; j++)
		{
			for (int i = Bwidth_X; i <= ii; i++)
			{
				out << (i - Bwidth_X) * dx << " ";
			}
			out << "\n";
		}
	}
#endif
#if DIM_Y
	for (int k = Bwidth_Z; k <= kk; k++)
	{
		for (int j = Bwidth_Y; j <= jj; j++)
		{
			for (int i = Bwidth_X; i <= ii; i++)
			{
				out << (j - Bwidth_Y) * dy << " ";
			}
			out << "\n";
		}
	}
#endif
#if DIM_Z
	for (int k = Bwidth_Z; k <= kk; k++)
	{
		for (int j = Bwidth_Y; j <= jj; j++)
		{
			for (int i = Bwidth_X; i <= ii; i++)
			{
				out << (k - Bwidth_Z) * dz << " ";
			}
			out << "\n";
		}
	}
#endif

#if DIM_X
	// u
	for (int k = Bwidth_Z; k < Zmax - Bwidth_Z; k++)
	{
		for (int j = Bwidth_Y; j < Ymax - Bwidth_Y; j++)
		{
			for (int i = Bwidth_X; i < Xmax - Bwidth_X; i++)
			{
				int id = Xmax * Ymax * k + Xmax * j + i;
				// if(levelset->h_phi[id]>= 0.0)
				out << fluids[0]->h_fstate.u[id] << " ";
				// else
				// 	out<<fluids[1]->h_fstate.u[id]<<" ";
			}
			out << "\n";
		}
	}
#endif

#if DIM_Y
	// v
	for (int k = Bwidth_Z; k < Zmax - Bwidth_Z; k++)
	{
		for (int j = Bwidth_Y; j < Ymax - Bwidth_Y; j++)
		{
			for (int i = Bwidth_X; i < Xmax - Bwidth_X; i++)
			{
				int id = Xmax * Ymax * k + Xmax * j + i;
				// if(levelset->h_phi[id]>= 0.0)
				out << fluids[0]->h_fstate.v[id] << " ";
				// else
				// 	out<<fluids[1]->h_fstate.v[id]<<" ";
			}
			out << "\n";
		}
	}
#endif

#if DIM_Z
	// w
	for (int k = Bwidth_Z; k < Zmax - Bwidth_Z; k++)
	{
		for (int j = Bwidth_Y; j < Ymax - Bwidth_Y; j++)
		{
			for (int i = Bwidth_X; i < Xmax - Bwidth_X; i++)
			{
				int id = Xmax * Ymax * k + Xmax * j + i;
				// if(levelset->h_phi[id]>= 0.0)
				out << fluids[0]->h_fstate.w[id] << " ";
				// else
				// 	out<<fluids[1]->h_fstate.w[id]<<" ";
			}
			out << "\n";
		}
	}
#endif

	// P
	for (int k = Bwidth_Z; k < Zmax - Bwidth_Z; k++)
	{
		for (int j = Bwidth_Y; j < Ymax - Bwidth_Y; j++)
		{
			for (int i = Bwidth_X; i < Xmax - Bwidth_X; i++)
			{
				int id = Xmax * Ymax * k + Xmax * j + i;
				// if(levelset->h_phi[id]>= 0.0)
				out << fluids[0]->h_fstate.p[id] << " ";
				// else
				// 	out<<fluids[1]->h_fstate.p[id]<<" ";
			}
			out << "\n";
		}
	}

	// rho
	for (int k = Bwidth_Z; k < Zmax - Bwidth_Z; k++)
	{
		for (int j = Bwidth_Y; j < Ymax - Bwidth_Y; j++)
		{
			for (int i = Bwidth_X; i < Xmax - Bwidth_X; i++)
			{
				int id = Xmax * Ymax * k + Xmax * j + i;
				// if(levelset->h_phi[id]>= 0.0)
				out << fluids[0]->h_fstate.rho[id] << " ";
				// else
				// 	out<<fluids[1]->h_fstate.rho[id]<<" ";
			}
			out << "\n";
		}
	}

	// //phi
	// for(int k=Bwidth_Z; k<Zmax-Bwidth_Z; k++){
	// 	for(int j=Bwidth_Y; j<Ymax-Bwidth_Y; j++){
	// 		for(int i=Bwidth_X; i<Xmax-Bwidth_X; i++)
	// 		{
	// 			int id = Xmax*Ymax*k + Xmax*j + i;
	// 			out<<levelset->h_phi[id]<<" ";
	// 		}
	// 		out<<"\n";
	// 	}
	// }

	out.close();
}