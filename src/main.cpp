#include "global_class.h"

int main(int argc, char *argv[])
{
#ifdef USE_MPI
	int rank, nRanks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#else
	int rank = 0;
	int nRanks = 1;
#endif // USE_MPI
	ConfigMap configMap = broadcast_parameters(std::string(argv[1]));
	// num_GPUS:number of GPU on this cluster, Pform_id: the first GPU's ID in all accelerators sycl detected
	int device_id = rank % num_GPUs + Pform_id;
	// accelerator_selector device;
	auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
	sycl::queue q(device);
	Setup setup(configMap, q);
	// Create MPI session if MPI enabled
	SYCLSolver syclsolver(setup);
	syclsolver.AllocateMemory(q);
	// syclsolver.InitialCondition(q);
	// // boundary conditions
	// syclsolver.BoundaryCondition(q, 0);
	// // update states by U
	// syclsolver.UpdateStates(q, 0);
	// // time marching by SYCL device
	// syclsolver.Evolution(q);
	return 0;
}