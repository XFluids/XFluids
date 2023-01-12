#include "MpiCommCart.h"

namespace mpiUtils
{
  // =======================================================
  // =======================================================
  MpiCommCart::MpiCommCart(int mx, int my, int mz, int isPeriodic, int allowReorder)
      : MpiComm(), mx_(mx), my_(my), mz_(mz)
  {
    int dims[NDIM_3D] = {mx, my, mz};
    int periods[NDIM_3D] = {isPeriodic, isPeriodic, isPeriodic};

    // create virtual topology cartesian 3D
    errCheck(MPI_Cart_create(MPI_COMM_WORLD, NDIM_3D, dims, periods, allowReorder, &comm_), "MPI_Cart_create");

    // fill nProc_ and myRank_
    init();

    // get cartesian coordinates (myCoords_) of current process (myRank_)
    getCoords(myRank_, NDIM_3D, myCoords_);
  }
} // namespace mpiUtils
