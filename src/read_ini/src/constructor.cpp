#include "../setupini.h"
#include "wheels/fworkdir.hpp"

// =======================================================
// // // struct Setup Member function definitions
// =======================================================
size_t Setup::adv_id = 0;
size_t Setup::sbm_id = 0;
bool Setup::adv_push = true;
std::vector<std::vector<Assign>> Setup::adv_nd;

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
#ifdef USE_MPI
    mpiTrans->communicator->synchronize();
    if (0 == myRank)
#endif // end USE_MPI
    {
#if __VENDOR_SUBMIT__
        vendorDeviceProp prop;
        CheckGPUErrors(vendorGetDeviceProperties(&prop, DeviceSelect[2]));
        // printf("  PCI Bus ID= %d; PCI Device ID= %d; PCI domain ID= %d; ECC= %d \n", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID, prop.ECCEnabled);
        printf("  Global Memory size= %2.1f GB;  clock freqency= %d khz;\n", double(prop.totalGlobalMem >> 20) / 1024.0, prop.memoryClockRate / 1000);
        printf("  Warp Size= %d;", prop.warpSize);
        printf("  multiProcessorCount= %d \n", prop.multiProcessorCount);
        printf("  32bit register number per block= %d = 256 x %d\n", prop.regsPerBlock, prop.regsPerBlock / 256);
        printf("  L2 cache size= %d MB;", prop.l2CacheSize / 1024 / 1024);
        printf("  Shared Memory per block: %ld KB\n", prop.sharedMemPerBlock / 1024);
        printf("  maxBlocksPerMultiProcessor= %d; maxThreadsPerMultiProcessor= %d= %d x %d;\n",
               prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor, prop.maxThreadsPerMultiProcessor / prop.maxBlocksPerMultiProcessor);
        printf("  Max threads per block= %d;  Max Threads in 3D= %d x %d x %d;", prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size : %d x %d x %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
#endif
        std::cout << "<---------------------------------------------------> \n";
        std::cout << "Setup_ini is copying buffers into Device . ";
    }

    CpyToGPU();
} // Setup::Setup end
