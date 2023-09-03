#pragma once

#include "global_setup.h"
#include "mpiUtils/MpiCommCart.h"
using namespace mpiUtils;

typedef struct
{
#if DIM_X
	// send && receive of Data-block-buffer at the posion of Xmin && Xmax
	real_t *TransBufSend_xmin, *TransBufSend_xmax, *TransBufRecv_xmin, *TransBufRecv_xmax;
#endif
#if DIM_Y
	real_t *TransBufSend_ymin, *TransBufSend_ymax, *TransBufRecv_ymin, *TransBufRecv_ymax;
#endif
#if DIM_Z
	real_t *TransBufSend_zmin, *TransBufSend_zmax, *TransBufRecv_zmin, *TransBufRecv_zmax;
#endif
} MpiData;
/**
 * Attention: MpiTrans only allocate memory automatically and transfer messages when the trans-function be called,
 * you must assignment the values in buffer youself
 */
struct MpiTrans
{
	// runtime determination if we are using float or double (for MPI communication)
	int data_type;
	// MPI rank of current process
	int myRank;
	// number of MPI processes
	int nProcs;
	// number of MPI process neighbors (4 in 2D and 6 in 3D)
	int nNeighbors;
	// MPI rank of adjacent MPI processes
	int neighborsRank[6];
	// CellSz: total number of the cell points transferred by Mpi ranks, needed by mpisendrecv-function
	// DataSz: total sizeof-data(bytes) of all physical arguments in these cell points used for malloc memory
	int mx, my, mz;
	MPI_Group comm_world; // dynamically set groupx, groupy, groupz;
#if DIM_X
	// CellSz: total number of the cell points transferred by Mpi ranks, needed by mpisendrecv-function
	// DataSz: total sizeof-data(bytes) of all physical arguments in these cell points used for malloc memory
	int Ghost_CellSz_x, Ghost_DataSz_x;
#endif
#if DIM_Y
	int Ghost_CellSz_y, Ghost_DataSz_y;
#endif
#if DIM_Z
	int Ghost_CellSz_z, Ghost_DataSz_z;
#endif
	// MPI communicator in a cartesian virtual topology
	MpiCommCart *communicator;
	// Boundary condition type with adjacent domains (corresponding to neighbor MPI processes)
	BConditions neighborsBC[6];
	// Data buffer for aware or nan-aware-mpi : aware-mpi trans device buffer directly but nan-aware-mpi trans host buffer only
	// Struct init delete host memory
#ifdef EXPLICIT_ALLOC
	MpiData d_mpiData, h_mpiData;
	void MpiBufCpy(MpiData dest, MpiData src, middle::device_t &q);
#else
	MpiData *d_mpiData, *h_mpiData;
#endif // end EXPLICIT_ALLOC
	MpiTrans(Block &bl, BConditions const Boundarys[6]);
	long double AllocMemory(middle::device_t &q, Block &bl, const int N); // might not need to allocate mem as in LAMNSS
	void MpiTransBuf(middle::device_t &q, Direction Dir);
	int MpiAllReduce(int &var, int Option);
	real_t MpiAllReduce(real_t &var, int Option);
	bool MpiBocastTrue(const bool mayTrue);
	void Get_RankGroupX(MPI_Group &group, const int pos);
	void Get_RankGroupY(MPI_Group &group, const int pos);
	void Get_RankGroupZ(MPI_Group &group, const int pos);
	int Get_RankGroupXY(MPI_Group &group, const int posx, const int posy);
	int Get_RankGroupXZ(MPI_Group &group, const int posx, const int posz);
	int Get_RankGroupYZ(MPI_Group &group, const int posy, const int posz);
	void GroupallReduce(void *input, void *result, int inputCount, int type, int op, MPI_Group group);
};