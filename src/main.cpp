#include "global_class.h"

int main(int argc, char *argv[])
{
#ifdef USE_MPI
	// Create MPI session if MPI enabled
	int rank, nRanks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#else
	int rank = 0;
	int nRanks = 1;
#endif // USE_MPI
	std::string ini_path;
	if (argc < 2)
		ini_path = std::string(IniFile);
	else if (argc == 2)
		ini_path = std::string(argv[1]);
	else
		std::cout << "Too much argcs appended to EulerSYCL while running\n";

	ConfigMap configMap = broadcast_parameters(ini_path);
	// num_GPUS:number of GPU on this cluster, Pform_id: the first GPU's ID in all accelerators sycl detected
	int device_id = rank % num_GPUs + Pform_id;
	// accelerator_selector device;
	auto device = sycl::platform::get_platforms()[device_id].get_devices()[0];
	sycl::queue q(device);
	Setup setup(configMap, q);
	SYCLSolver syclsolver(setup);
	// AllocateMemory
	// std::cout << "sleep(1)\n";
	// // sleep(10);
	syclsolver.AllocateMemory(q);
	// std::cout << "sleep(2)\n";
	// // sleep(10);
	//  Initialize original states
	syclsolver.InitialCondition(q);
	// boundary conditions
	syclsolver.BoundaryCondition(q, 0);
	// update states by U
	syclsolver.UpdateStates(q, 0);
	// time marching by SYCL device
	// std::cout << "sleep(3)\n";
	// // sleep(10);
	syclsolver.Evolution(q);
	return 0;
}