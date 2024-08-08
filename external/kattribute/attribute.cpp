#include "attribute.h"

#if defined(__HIPSYCL_ENABLE_HIP_TARGET__) || (__HIPSYCL_ENABLE_CUDA_TARGET__)
void GetKernelAttributes(const void *Func_ptr, std::string Func_name)
{
#ifdef USE_MPI
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (0 == rank)
#endif // end USE_MPI
	{
		vendorFuncAttributes attr;
		CheckGPUErrors(vendorFuncGetAttributes(&attr, Func_ptr));
		printf(">>>>> Kernel name: %s\n", Func_name.c_str());
		printf("Max threads per block: %d \n", attr.maxThreadsPerBlock);
		printf("Max dynamic shared mem: %d bytes \n", attr.maxDynamicSharedSizeBytes);
		printf("Constant mem usage: %d bytes;      register usage: %d \n", (int)(attr.constSizeBytes), attr.numRegs);
		printf("Local mem usage(register spilling): %d byte;          shared mem usage: %d bytes \n", (int)(attr.localSizeBytes), (int)(attr.sharedSizeBytes));
		printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
	}
}
#endif