#include "mpiPacks.h"

MpiTrans::MpiTrans(Block &bl, BConditions const Boundarys[6])
{
	mx = bl.mx, my = bl.my, mz = bl.mz;
	// runtime determination if we are using float ou double (for MPI communication)
	data_type = typeid(1.0).name() == typeid(_DF(1.0)).name() ? mpiUtils::MpiComm::DOUBLE : mpiUtils::MpiComm::FLOAT;
	// check that parameters are consistent
	bool error = false;
	error |= (mx < 1);
	error |= (my < 1);
	error |= (mz < 1);
	// get world communicator size and check it is consistent with mesh grid sizes
	MPI_Comm_group(MPI_COMM_WORLD, &comm_world);
	nProcs = MpiComm::world().getNProc();
	if (nProcs != mx * my * mz)
	{
		std::cerr << "ERROR: mx * my * mz = " << mx * my * mz << " must match with nRanks given to mpirun !!!\n";
		abort();
	}
	communicator = new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);

	// get my MPI rank inside topology    // get my MPI rank inside topology
	myRank = communicator->getRank();
	{
		// get coordinates of myRank inside topology
		// myMpiPos[0] is between 0 and mx-1 // myMpiPos[1] is between 0 and my-1 // myMpiPos[2] is between 0 and mz-1
		int mpiPos[3] = {0, 0, 0};
		communicator->getMyCoords(&mpiPos[0]);
		bl.myMpiPos_x = mpiPos[0];
		bl.myMpiPos_y = mpiPos[1];
		bl.myMpiPos_z = mpiPos[2];
	}

	// compute MPI ranks of our neighbors and set default boundary condition types
	nNeighbors = 2 * (DIM_X + DIM_Y + DIM_Z);
	neighborsRank[X_MIN] = DIM_X ? communicator->getNeighborRank<X_MIN>() : 0;
	neighborsRank[X_MAX] = DIM_X ? communicator->getNeighborRank<X_MAX>() : 0;
	neighborsRank[Y_MIN] = DIM_Y ? communicator->getNeighborRank<Y_MIN>() : 0;
	neighborsRank[Y_MAX] = DIM_Y ? communicator->getNeighborRank<Y_MAX>() : 0;
	neighborsRank[Z_MIN] = DIM_Z ? communicator->getNeighborRank<Z_MIN>() : 0;
	neighborsRank[Z_MAX] = DIM_Z ? communicator->getNeighborRank<Z_MAX>() : 0;

	neighborsBC[X_MIN] = DIM_X ? BC_COPY : BC_UNDEFINED;
	neighborsBC[X_MAX] = DIM_X ? BC_COPY : BC_UNDEFINED;
	neighborsBC[Y_MIN] = DIM_Y ? BC_COPY : BC_UNDEFINED;
	neighborsBC[Y_MAX] = DIM_Y ? BC_COPY : BC_UNDEFINED;
	neighborsBC[Z_MIN] = DIM_Z ? BC_COPY : BC_UNDEFINED;
	neighborsBC[Z_MAX] = DIM_Z ? BC_COPY : BC_UNDEFINED;

	// Identify outside boundaries for mpi rank at edge of each direction in all mpi nRanks world
#ifdef DIM_X
	if (bl.myMpiPos_x == 0) // X_MIN boundary
		neighborsBC[X_MIN] = Boundarys[XMIN];
	if (bl.myMpiPos_x == mx - 1) // X_MAX boundary
		neighborsBC[X_MAX] = Boundarys[XMAX];
#endif // end DIM_X
#ifdef DIM_Y
	if (bl.myMpiPos_y == 0) // Y_MIN boundary
		neighborsBC[Y_MIN] = Boundarys[YMIN];
	if (bl.myMpiPos_y == my - 1) // Y_MAX boundary
		neighborsBC[Y_MAX] = Boundarys[YMAX];
#endif // end DIM_Y
#ifdef DIM_Z
	if (bl.myMpiPos_z == 0) // Z_MIN boundary
		neighborsBC[Z_MIN] = Boundarys[ZMIN];
	if (bl.myMpiPos_z == mz - 1) // Z_MAX boundary
		neighborsBC[Z_MAX] = Boundarys[ZMAX];
#endif // end DIM_Z
} // MpiTrans::MpiTrans
// =======================================================
// =======================================================
int MpiTrans::Get_RankGroupXY(MPI_Group &group, const int posx, const int posy)
{
	int *members = new int[mz], coords[3];
	for (size_t k = 0; k < mz; k++)
	{
		coords[0] = posx, coords[1] = posy, coords[2] = k;
		members[k] = communicator->getCartRank(coords);
	}

	MPI_Group_incl(comm_world, mz, members, &group);
	return members[0];
} // MPI_Group MpiTrans::Get_RankGroupXY
// =======================================================
// =======================================================
int MpiTrans::Get_RankGroupXZ(MPI_Group &group, const int posx, const int posz)
{
	int *members = new int[my], coords[3];
	for (size_t j = 0; j < my; j++)
	{
		coords[0] = posx, coords[1] = j, coords[2] = posz;
		members[j] = communicator->getCartRank(coords);
	}

	MPI_Group_incl(comm_world, my, members, &group);
	return members[0];
} // MPI_Group MpiTrans::Get_RankGroupXZ
// =======================================================
// =======================================================
int MpiTrans::Get_RankGroupYZ(MPI_Group &group, const int posy, const int posz)
{
	int *members = new int[mx], coords[3];
	for (size_t i = 0; i < mx; i++)
	{
		coords[0] = i, coords[1] = posy, coords[2] = posz;
		members[i] = communicator->getCartRank(coords);
	}

	MPI_Group_incl(comm_world, mx, members, &group);
	return members[0];
} // MPI_Group MpiTrans::Get_RankGroupXY
// =======================================================
// =======================================================
void MpiTrans::Get_RankGroupX(MPI_Group &group, const int pos)
{
	int size = my * mz, *members = new int[size], coords[3];
	for (size_t j = 0; j < my; j++)
		for (size_t k = 0; k < mz; k++)
		{
			coords[0] = pos, coords[1] = j, coords[2] = k;
			members[j * mz + k] = communicator->getCartRank(coords);
		}

	MPI_Group_incl(comm_world, size, members, &group);
} // MPI_Group MpiTrans::Get_RankGroupX
// =======================================================
// =======================================================
void MpiTrans::Get_RankGroupY(MPI_Group &group, const int pos)
{
	int size = mx * mz, *members = new int[size], coords[3];
	for (size_t i = 0; i < mx; i++)
		for (size_t k = 0; k < mz; k++)
		{
			coords[0] = i, coords[1] = pos, coords[2] = k;
			members[i * mz + k] = communicator->getCartRank(coords);
		}

	MPI_Group_incl(comm_world, size, members, &group);
} // MPI_Group MpiTrans::Get_RankGroupY
// =======================================================
// =======================================================
void MpiTrans::Get_RankGroupZ(MPI_Group &group, const int pos)
{
	int size = mx * my, *members = new int[size], coords[3];
	for (size_t i = 0; i < mx; i++)
		for (size_t j = 0; j < my; j++)
		{
			coords[0] = i, coords[1] = j, coords[2] = pos;
			members[i * my + j] = communicator->getCartRank(coords);
		}

	MPI_Group_incl(comm_world, size, members, &group);
} // MPI_Group MpiTrans::Get_RankGroupZ
// =======================================================
// =======================================================
void MpiTrans::GroupallReduce(void *input, void *result, int inputCount, int type, int op, MPI_Group group)
{
	MPI_Comm group_comm;
	MPI_Op mpiOp = communicator->getOp(op);
	MPI_Datatype mpiType = communicator->getDataType(type);
	MPI_Comm_create(MPI_COMM_WORLD, group, &group_comm);
	MPI_Allreduce(input, result, inputCount, mpiType, mpiOp, group_comm);
}
// =======================================================
// =======================================================
long double MpiTrans::AllocMemory(middle::device_t &q, Block &bl, const int N)
{
	long double temp = 0.0;
	if (N != 0)
	{
#ifndef EXPLICIT_ALLOC
		h_mpiData = middle::MallocHost<MpiData>(h_mpiData, 1, q);
#endif // end EXPLICIT_ALLOC
#if DIM_X
		Ghost_CellSz_x = bl.Bwidth_X * bl.Ymax * bl.Zmax * N;
		// Ghost_DataSz_x = Ghost_CellSz_x * sizeof(real_t);
		temp += Ghost_CellSz_x * sizeof(real_t) * 4.0 / 1024.0 / 1024.0;
// =======================================================
#ifdef EXPLICIT_ALLOC
		// =======================================================
#ifndef AWARE_MPI
		h_mpiData.TransBufSend_xmin = middle::MallocHost<real_t>(h_mpiData.TransBufSend_xmin, Ghost_CellSz_x, q);
		h_mpiData.TransBufRecv_xmin = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_xmin, Ghost_CellSz_x, q);
		h_mpiData.TransBufRecv_xmax = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_xmax, Ghost_CellSz_x, q);
		h_mpiData.TransBufSend_xmax = middle::MallocHost<real_t>(h_mpiData.TransBufSend_xmax, Ghost_CellSz_x, q);
#endif // end AWARE_MPI
		d_mpiData.TransBufSend_xmin = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_xmin, Ghost_CellSz_x, q);
		d_mpiData.TransBufRecv_xmin = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_xmin, Ghost_CellSz_x, q);
		d_mpiData.TransBufRecv_xmax = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_xmax, Ghost_CellSz_x, q);
		d_mpiData.TransBufSend_xmax = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_xmax, Ghost_CellSz_x, q);
		// =======================================================
#else
		h_mpiData->TransBufSend_xmin = middle::MallocHost<real_t>(h_mpiData->TransBufSend_xmin, Ghost_CellSz_x, q);
		h_mpiData->TransBufRecv_xmin = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_xmin, Ghost_CellSz_x, q);
		h_mpiData->TransBufRecv_xmax = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_xmax, Ghost_CellSz_x, q);
		h_mpiData->TransBufSend_xmax = middle::MallocHost<real_t>(h_mpiData->TransBufSend_xmax, Ghost_CellSz_x, q);
#endif // end EXPLICIT_ALLOC
// =======================================================
#endif // end DIM_X

#if DIM_Y
		Ghost_CellSz_y = bl.Bwidth_Y * bl.Xmax * bl.Zmax * N;
		// Ghost_DataSz_y = Ghost_CellSz_y * sizeof(real_t);
		temp += Ghost_CellSz_y * sizeof(real_t) * 4.0 / 1024.0 / 1024.0;
// =======================================================
#ifdef EXPLICIT_ALLOC
		// =======================================================
#ifndef AWARE_MPI
		h_mpiData.TransBufSend_ymin = middle::MallocHost<real_t>(h_mpiData.TransBufSend_ymin, Ghost_CellSz_y, q);
		h_mpiData.TransBufRecv_ymin = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_ymin, Ghost_CellSz_y, q);
		h_mpiData.TransBufRecv_ymax = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_ymax, Ghost_CellSz_y, q);
		h_mpiData.TransBufSend_ymax = middle::MallocHost<real_t>(h_mpiData.TransBufSend_ymax, Ghost_CellSz_y, q);
#endif // end AWARE_MPI
		d_mpiData.TransBufSend_ymin = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_ymin, Ghost_CellSz_y, q);
		d_mpiData.TransBufRecv_ymin = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_ymin, Ghost_CellSz_y, q);
		d_mpiData.TransBufRecv_ymax = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_ymax, Ghost_CellSz_y, q);
		d_mpiData.TransBufSend_ymax = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_ymax, Ghost_CellSz_y, q);
		// =======================================================
#else
		h_mpiData->TransBufSend_ymin = middle::MallocHost<real_t>(h_mpiData->TransBufSend_ymin, Ghost_CellSz_y, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_y, q));
		h_mpiData->TransBufRecv_ymin = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_ymin, Ghost_CellSz_y, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_y, q));
		h_mpiData->TransBufRecv_ymax = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_ymax, Ghost_CellSz_y, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_y, q));
		h_mpiData->TransBufSend_ymax = middle::MallocHost<real_t>(h_mpiData->TransBufSend_ymax, Ghost_CellSz_y, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_y, q));
#endif // end EXPLICIT_ALLOC
// ======================================================
#endif // end DIM_Y

#if DIM_Z
		Ghost_CellSz_z = bl.Bwidth_Z * bl.Xmax * bl.Ymax * N;
		// Ghost_DataSz_z = Ghost_CellSz_z * sizeof(real_t);
		temp += Ghost_CellSz_z * sizeof(real_t) * 4.0 / 1024.0 / 1024.0;
// =======================================================
#ifdef EXPLICIT_ALLOC
		// =======================================================
#ifndef AWARE_MPI
		h_mpiData.TransBufSend_zmin = middle::MallocHost<real_t>(h_mpiData.TransBufSend_zmin, Ghost_CellSz_z, q);
		h_mpiData.TransBufRecv_zmin = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_zmin, Ghost_CellSz_z, q);
		h_mpiData.TransBufRecv_zmax = middle::MallocHost<real_t>(h_mpiData.TransBufRecv_zmax, Ghost_CellSz_z, q);
		h_mpiData.TransBufSend_zmax = middle::MallocHost<real_t>(h_mpiData.TransBufSend_zmax, Ghost_CellSz_z, q);
#endif // end AWARE_MPI
		d_mpiData.TransBufSend_zmin = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_zmin, Ghost_CellSz_z, q);
		d_mpiData.TransBufRecv_zmin = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_zmin, Ghost_CellSz_z, q);
		d_mpiData.TransBufRecv_zmax = middle::MallocDevice<real_t>(d_mpiData.TransBufRecv_zmax, Ghost_CellSz_z, q);
		d_mpiData.TransBufSend_zmax = middle::MallocDevice<real_t>(d_mpiData.TransBufSend_zmax, Ghost_CellSz_z, q);
		// =======================================================
#else
		h_mpiData->TransBufSend_zmin = middle::MallocHost<real_t>(h_mpiData->TransBufSend_zmin, Ghost_CellSz_z, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_z, q));
		h_mpiData->TransBufRecv_zmin = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_zmin, Ghost_CellSz_z, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_z, q));
		h_mpiData->TransBufRecv_zmax = middle::MallocHost<real_t>(h_mpiData->TransBufRecv_zmax, Ghost_CellSz_z, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_z, q));
		h_mpiData->TransBufSend_zmax = middle::MallocHost<real_t>(h_mpiData->TransBufSend_zmax, Ghost_CellSz_z, q); // static_cast<real_t *>(sycl::malloc_host(Ghost_DataSz_z, q));
#endif // end EXPLICIT_ALLOC
// ======================================================
#endif // end DIM_Z

#ifndef EXPLICIT_ALLOC
		d_mpiData = middle::MallocDevice<MpiData>(d_mpiData, 1, q);
		middle::MemCpy(d_mpiData, h_mpiData, sizeof(MpiData), q, middle::MemCpy_t::HtD);
// #else
// ======================================================
// #ifdef AWARE_MPI // needn't host buffer for aware-mpi enabled, only explicit alloc needed
// #if DIM_X
//         middle::Free(h_mpiData.TransBufSend_xmin, q);
//         middle::Free(h_mpiData.TransBufRecv_xmin, q);
//         middle::Free(h_mpiData.TransBufRecv_xmax, q);
//         middle::Free(h_mpiData.TransBufSend_xmax, q);
// #endif // end DIM_X
// #if DIM_Y
//         middle::Free(h_mpiData.TransBufSend_ymin, q);
//         middle::Free(h_mpiData.TransBufRecv_ymin, q);
//         middle::Free(h_mpiData.TransBufRecv_ymax, q);
//         middle::Free(h_mpiData.TransBufSend_ymax, q);
// #endif // end DIM_Y
// #if DIM_Z
//         middle::Free(h_mpiData.TransBufSend_zmin, q);
//         middle::Free(h_mpiData.TransBufRecv_zmin, q);
//         middle::Free(h_mpiData.TransBufRecv_zmax, q);
//         middle::Free(h_mpiData.TransBufSend_zmax, q);
// #endif // end DIM_Z
// #endif // AWARE_MPI
// ======================================================
#endif // end EXPLICIT_ALLOC
	}
	return temp;
} // MpiTrans::AllocMem
// =======================================================
// =======================================================
#ifdef EXPLICIT_ALLOC
void MpiTrans::MpiBufCpy(MpiData dest, MpiData src, middle::device_t &q)
{
#if DIM_X
	middle::MemCpy<real_t>(dest.TransBufSend_xmin, src.TransBufSend_xmin, Ghost_CellSz_x, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_xmin, src.TransBufRecv_xmin, Ghost_CellSz_x, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_xmax, src.TransBufRecv_xmax, Ghost_CellSz_x, q);
	middle::MemCpy<real_t>(dest.TransBufSend_xmax, src.TransBufSend_xmax, Ghost_CellSz_x, q);
#endif // end DIM_X

#if DIM_Y
	middle::MemCpy<real_t>(dest.TransBufSend_ymin, src.TransBufSend_ymin, Ghost_CellSz_y, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_ymin, src.TransBufRecv_ymin, Ghost_CellSz_y, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_ymax, src.TransBufRecv_ymax, Ghost_CellSz_y, q);
	middle::MemCpy<real_t>(dest.TransBufSend_ymax, src.TransBufSend_ymax, Ghost_CellSz_y, q);
#endif // end DIM_Y

#if DIM_Z
	middle::MemCpy<real_t>(dest.TransBufSend_zmin, src.TransBufSend_zmin, Ghost_CellSz_z, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_zmin, src.TransBufRecv_zmin, Ghost_CellSz_z, q);
	middle::MemCpy<real_t>(dest.TransBufRecv_zmax, src.TransBufRecv_zmax, Ghost_CellSz_z, q);
	middle::MemCpy<real_t>(dest.TransBufSend_zmax, src.TransBufSend_zmax, Ghost_CellSz_z, q);
#endif // end DIM_Z
}
#endif // end EXPLICIT_ALLOC
// =======================================================
// =======================================================
void MpiTrans::MpiTransBuf(middle::device_t &q, Direction Dir)
{
#ifdef EXPLICIT_ALLOC
// =======================================================
#ifndef AWARE_MPI
	MpiBufCpy(h_mpiData, d_mpiData, q);
#endif // end AWARE_MPI
// =======================================================
#else
	middle::MemCpy(h_mpiData, d_mpiData, sizeof(MpiData), q, middle::MemCpy_t::DtH); // q.memcpy(h_mpiData, d_mpiData, sizeof(MpiData));
#endif // end EXPLICIT_ALLOC

	switch (Dir)
	{
	case XDIR:
	{
#if DIM_X
#ifdef EXPLICIT_ALLOC
// =======================================================
#ifndef AWARE_MPI
		real_t *inptr_TransBufSend_xmin = h_mpiData.TransBufSend_xmin;
		real_t *inptr_TransBufSend_xmax = h_mpiData.TransBufSend_xmax;
		real_t *inptr_TransBufRecv_xmin = h_mpiData.TransBufRecv_xmin;
		real_t *inptr_TransBufRecv_xmax = h_mpiData.TransBufRecv_xmax;
#else
		real_t *inptr_TransBufSend_xmin = d_mpiData.TransBufSend_xmin;
		real_t *inptr_TransBufSend_xmax = d_mpiData.TransBufSend_xmax;
		real_t *inptr_TransBufRecv_xmin = d_mpiData.TransBufRecv_xmin;
		real_t *inptr_TransBufRecv_xmax = d_mpiData.TransBufRecv_xmax;
#endif // end AWARE_MPI
// =======================================================
#else

		// =======================================================
		// #ifndef AWARE_MPI

		real_t *inptr_TransBufSend_xmin = h_mpiData->TransBufSend_xmin;
		real_t *inptr_TransBufSend_xmax = h_mpiData->TransBufSend_xmax;
		real_t *inptr_TransBufRecv_xmin = h_mpiData->TransBufRecv_xmin;
		real_t *inptr_TransBufRecv_xmax = h_mpiData->TransBufRecv_xmax;

// #else
// #define inptr_TransBufSend_xmin d_mpiData->TransBufSend_xmin
// #define inptr_TransBufSend_xmax d_mpiData->TransBufSend_xmax
// #define inptr_TransBufRecv_xmin d_mpiData->TransBufRecv_xmin
// #define inptr_TransBufRecv_xmax d_mpiData->TransBufRecv_xmax
// #endif // end AWARE_MPI
// =======================================================
#endif // end EXPLICIT_ALLOC
		communicator->synchronize();
		communicator->sendrecv(inptr_TransBufSend_xmin, Ghost_CellSz_x, data_type, neighborsRank[X_MIN], 100,
							   inptr_TransBufRecv_xmax, Ghost_CellSz_x, data_type, neighborsRank[X_MAX], 100);
		communicator->sendrecv(inptr_TransBufSend_xmax, Ghost_CellSz_x, data_type, neighborsRank[X_MAX], 200,
							   inptr_TransBufRecv_xmin, Ghost_CellSz_x, data_type, neighborsRank[X_MIN], 200);
#endif // end DIM_X
	}
	break;

	case YDIR:
	{
#if DIM_Y
#ifdef EXPLICIT_ALLOC
// =======================================================
#ifndef AWARE_MPI
		real_t *inptr_TransBufSend_ymin = h_mpiData.TransBufSend_ymin;
		real_t *inptr_TransBufSend_ymax = h_mpiData.TransBufSend_ymax;
		real_t *inptr_TransBufRecv_ymin = h_mpiData.TransBufRecv_ymin;
		real_t *inptr_TransBufRecv_ymax = h_mpiData.TransBufRecv_ymax;
#else
		real_t *inptr_TransBufSend_ymin = d_mpiData.TransBufSend_ymin;
		real_t *inptr_TransBufSend_ymax = d_mpiData.TransBufSend_ymax;
		real_t *inptr_TransBufRecv_ymin = d_mpiData.TransBufRecv_ymin;
		real_t *inptr_TransBufRecv_ymax = d_mpiData.TransBufRecv_ymax;
#endif // end AWARE_MPI
// =======================================================
#else
		// =======================================================
		// #ifndef AWARE_MPI
		real_t *inptr_TransBufSend_ymin = h_mpiData->TransBufSend_ymin;
		real_t *inptr_TransBufSend_ymax = h_mpiData->TransBufSend_ymax;
		real_t *inptr_TransBufRecv_ymin = h_mpiData->TransBufRecv_ymin;
		real_t *inptr_TransBufRecv_ymax = h_mpiData->TransBufRecv_ymax;
// #else
// #define inptr_TransBufSend_ymin d_mpiData->TransBufSend_ymin
// #define inptr_TransBufSend_ymax d_mpiData->TransBufSend_ymax
// #define inptr_TransBufRecv_ymin d_mpiData->TransBufRecv_ymin
// #define inptr_TransBufRecv_ymax d_mpiData->TransBufRecv_ymax
// #endif // end AWARE_MPI
// =======================================================
#endif // end EXPLICIT_ALLOC

		communicator->synchronize();
		communicator->sendrecv(inptr_TransBufSend_ymin, Ghost_CellSz_y, data_type, neighborsRank[Y_MIN], 100,
							   inptr_TransBufRecv_ymax, Ghost_CellSz_y, data_type, neighborsRank[Y_MAX], 100);
		communicator->sendrecv(inptr_TransBufSend_ymax, Ghost_CellSz_y, data_type, neighborsRank[Y_MAX], 200,
							   inptr_TransBufRecv_ymin, Ghost_CellSz_y, data_type, neighborsRank[Y_MIN], 200);
#endif // end DIM_Y
	}
	break;

	case ZDIR:
	{
#if DIM_Z
#ifdef EXPLICIT_ALLOC
// =======================================================
#ifndef AWARE_MPI
		real_t *inptr_TransBufSend_zmin = h_mpiData.TransBufSend_zmin;
		real_t *inptr_TransBufSend_zmax = h_mpiData.TransBufSend_zmax;
		real_t *inptr_TransBufRecv_zmin = h_mpiData.TransBufRecv_zmin;
		real_t *inptr_TransBufRecv_zmax = h_mpiData.TransBufRecv_zmax;
#else
		real_t *inptr_TransBufSend_zmin = d_mpiData.TransBufSend_zmin;
		real_t *inptr_TransBufSend_zmax = d_mpiData.TransBufSend_zmax;
		real_t *inptr_TransBufRecv_zmin = d_mpiData.TransBufRecv_zmin;
		real_t *inptr_TransBufRecv_zmax = d_mpiData.TransBufRecv_zmax;
#endif // end AWARE_MPI
// =======================================================
#else
		// =======================================================
		// #ifndef AWARE_MPI
		real_t *inptr_TransBufSend_zmin = h_mpiData->TransBufSend_zmin;
		real_t *inptr_TransBufSend_zmax = h_mpiData->TransBufSend_zmax;
		real_t *inptr_TransBufRecv_zmin = h_mpiData->TransBufRecv_zmin;
		real_t *inptr_TransBufRecv_zmax = h_mpiData->TransBufRecv_zmax;
// #else
// #define inptr_TransBufSend_zmin d_mpiData->TransBufSend_zmin
// #define inptr_TransBufSend_zmax d_mpiData->TransBufSend_zmax
// #define inptr_TransBufRecv_zmin d_mpiData->TransBufRecv_zmin
// #define inptr_TransBufRecv_zmax d_mpiData->TransBufRecv_zmax
// #endif // end AWARE_MPI
// =======================================================
#endif // end EXPLICIT_ALLOC

		communicator->synchronize();
		communicator->sendrecv(inptr_TransBufSend_zmin, Ghost_CellSz_z, data_type, neighborsRank[Z_MIN], 100,
							   inptr_TransBufRecv_zmax, Ghost_CellSz_z, data_type, neighborsRank[Z_MAX], 100);
		communicator->sendrecv(inptr_TransBufSend_zmax, Ghost_CellSz_z, data_type, neighborsRank[Z_MAX], 200,
							   inptr_TransBufRecv_zmin, Ghost_CellSz_z, data_type, neighborsRank[Z_MIN], 200);
#endif // end DIM_Z
	}
	break;
	} // end switch()

	communicator->synchronize();

#ifdef EXPLICIT_ALLOC
// =======================================================
#ifndef AWARE_MPI
	MpiBufCpy(d_mpiData, h_mpiData, q);
#endif // end AWARE_MPI
// =======================================================
#else
	middle::MemCpy(d_mpiData, h_mpiData, sizeof(MpiData), q, middle::MemCpy_t::HtD); // q.memcpy(d_mpiData, h_mpiData, sizeof(MpiData));
#endif // end EXPLICIT_ALLOC
}

int MpiTrans::MpiAllReduce(int &var, int Option)
{
	int temp = var;
#ifdef USE_MPI
	communicator->synchronize();
	communicator->allReduce(&var, &temp, 1, mpiUtils::MpiComm::INT, Option);
	communicator->synchronize();
#endif // end USE_MPI
	var = temp;
	return temp;
}

real_t MpiTrans::MpiAllReduce(real_t &var, int Option)
{
	real_t temp = var;
#ifdef USE_MPI
	communicator->synchronize();
	communicator->allReduce(&var, &temp, 1, data_type, Option);
	communicator->synchronize();
#endif // end USE_MPI
	var = temp;
	return temp;
}

bool MpiTrans::MpiBocastTrue(const bool mayTrue)
{
	int root, maybe_root = mayTrue ? myRank : 0, error_t = mayTrue ? 1 : 0;
#ifdef USE_MPI
	communicator->synchronize();
	communicator->allReduce(&maybe_root, &root, 1, mpiUtils::MpiComm::INT, mpiUtils::MpiComm::MAX);
	communicator->synchronize();
	communicator->bcast(&(error_t), 1, mpiUtils::MpiComm::INT, root);
	communicator->synchronize();
#endif // end USE_MPI
	return error_t;
}