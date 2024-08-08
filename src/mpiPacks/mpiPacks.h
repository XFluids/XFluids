#pragma once

#include "middle.hpp"
#include "global_setup.h"
#include "mpiUtils/MpiCommCart.h"

using namespace mpiUtils;

typedef struct
{
	// send && receive of Data-block-buffer at the posion of Xmin && Xmax
	real_t *TransBufSend_xmin, *TransBufSend_xmax, *TransBufRecv_xmin, *TransBufRecv_xmax;
	real_t *TransBufSend_ymin, *TransBufSend_ymax, *TransBufRecv_ymin, *TransBufRecv_ymax;
	real_t *TransBufSend_zmin, *TransBufSend_zmax, *TransBufRecv_zmin, *TransBufRecv_zmax;
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
	// number of MPI ranks at each dir
	int mx, my, mz;
	// number of MPI process neighbors (4 in 2D and 6 in 3D)
	int nNeighbors;
	// MPI rank of adjacent MPI processes
	int neighborsRank[6];

	// CellSz: total number of the cell points transferred by Mpi ranks, needed by mpisendrecv-function
	int Ghost_CellSz_x, Ghost_CellSz_y, Ghost_CellSz_z;
	// DataSz: total sizeof-data(bytes) of all physical arguments in these cell points used for malloc memory
	int Ghost_DataSz_x, Ghost_DataSz_y, Ghost_DataSz_z;

	Block mbl; // Block
	// Dynamically set groupx, groupy, groupz;
	MPI_Group comm_world;
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

	MpiTrans() {};
	~MpiTrans() {};
	MpiTrans(Block &bl, BConditions const Boundarys[6]);
	long double AllocMemory(middle::device_t &q, Block &bl, const int N); // might not need to allocate mem as in XFLUIDS
	void MpiTransBuf(middle::device_t &q, Direction Dir);
	int MpiAllReduce(int &var, int Option);
	real_t MpiAllReduce(real_t &var, int Option);
	void Get_RankGroupX(MPI_Group &group, const int pos);
	void Get_RankGroupY(MPI_Group &group, const int pos);
	void Get_RankGroupZ(MPI_Group &group, const int pos);
	int Get_RankGroupXY(MPI_Group &group, const int posx, const int posy);
	int Get_RankGroupXZ(MPI_Group &group, const int posx, const int posz);
	int Get_RankGroupYZ(MPI_Group &group, const int posy, const int posz);
	bool BocastTrue(const bool mayTrue);
	void BocastGroup2All(void *target, int type, int *group_ranks);
	void allReduce(void *input, void *result, int inputCount, int type, int op, MPI_Comm group_comm);
	void GroupallReduce(void *input, void *result, int inputCount, int type, int op, int *group_ranks, bool bocast = false);
};

// // 打印某个进程组中进程在MPI_COMM_WORLD中的进程号
// void printf_ranknumber_in_world(MPI_Group group, MPI_Group world_group)
// {
// 	int size;
// 	int *rank1;
// 	int *rank2;
// 	MPI_Group_size(group, &size);
// 	rank1 = (int *)malloc(size * sizeof(int));
// 	rank2 = (int *)malloc(size * sizeof(int));
// 	for (int i = 0; i < size; i++)
// 	{
// 		rank1[i] = i;
// 	}
// 	MPI_Group_translate_ranks(group, size, rank1, world_group, rank2);
// 	for (int j = 0; j < size; j++)
// 	{
// 		printf("%d,", rank2[j]);
// 	}
// 	printf("\n");
// 	MPI_Group_free(&group);
// }
