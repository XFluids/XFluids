#include "../setupini.h"
#include "wheels/fworkdir.hpp"

// =======================================================
// // // struct Setup Member function definitions
// =======================================================
Setup::Setup(int argc, char **argv, int rank, int nranks) : myRank(rank), nRanks(nranks), apa(argc, argv)
{
#ifdef USE_MPI // Create MPI session if MPI enabled
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif         // USE_MPI
    ReWrite(); // rewrite parameters using appended options
    // // sycl::queue construction
    q = sycl::queue(sycl::platform::get_platforms()[DeviceSelect[1]].get_devices()[DeviceSelect[2]]);
    // // get Work directory
    WorkDir = getWorkDir(std::string(argv[0]), "XFLUIDS");
    // // NOTE: read_grid
    grid = Gridread(q, BlSz, WorkDir + "/" + std::string(INI_SAMPLE), myRank, nRanks);

    // /*begin runtime read , fluid && compoent characteristics set*/
    ReadSpecies();
    if (ReactSources)
        ReadReactions();
        /*end runtime read*/

// read && caculate coffes for visicity
#if Visc
    GetFitCoefficient();
#endif

    init(); // Ini
#ifdef USE_MPI
    mpiTrans = new MpiTrans(BlSz, Boundarys);
    mpiTrans->communicator->synchronize();
#endif // end USE_MPI
    {
        std::cout << "Selected Device: " << middle::DevInfo(q) << "  of rank: " << myRank << std::endl;
    }
    CpyToGPU();
} // Setup::Setup end
