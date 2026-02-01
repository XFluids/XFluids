#include "MpiCommCart.h"
#include <iostream>

namespace mpiUtils
{
  // =======================================================
  // =======================================================
  MpiCommCart::MpiCommCart(int mx, int my, int mz, int isPeriodic, int allowReorder)
      : MpiComm(), mx_(mx), my_(my), mz_(mz)
  {
    int dims[NDIM_3D] = {mx, my, mz};
    int periods[NDIM_3D] = {isPeriodic, isPeriodic, isPeriodic};

    // Create virtual topology cartesian 3D
    // We still call this to create the communicator with correct topology info
    errCheck(MPI_Cart_create(MPI_COMM_WORLD, NDIM_3D, dims, periods, allowReorder, &comm_), "MPI_Cart_create");

    // Update nProc_ and myRank_ from the new communicator
    init();

    // [Fix for Intel MPI Crash]
    // Intel MPI may optimize 2x1x1 topology to 1D, causing MPI_Cart_coords to crash 
    // even with correct maxdims.
    // Instead of querying MPI, we calculate coordinates manually. 
    // This is mathematically safe for Cartesian topologies.
    // Logic: Rank = z * (mx*my) + y * (mx) + x
    
    int rank_tmp = myRank_;
    
    // 1. Calculate Z
    // stride_z = mx * my
    int stride_z = mx_ * my_;
    myCoords_[2] = rank_tmp / stride_z;
    rank_tmp = rank_tmp % stride_z;
    
    // 2. Calculate Y
    // stride_y = mx
    myCoords_[1] = rank_tmp / mx_;
    
    // 3. Calculate X
    myCoords_[0] = rank_tmp % mx_;

    // Debug output to verify (Optional)
    // std::cout << "[MpiCommCart] Rank " << myRank_ 
    //           << " Manual Coords: (" << myCoords_[0] << "," 
    //           << myCoords_[1] << "," << myCoords_[2] << ")" << std::endl;
  }
} // namespace mpiUtils