#include "timer.h"

float OutThisTime(std::chrono::high_resolution_clock::time_point start_time)
{
	float duration = 0.0f;
	{
		std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration<float, std::milli>(end_time - start_time).count() / 1000.0f;
	}
	return duration;
}

#ifdef USE_MPI
    float OutThisTime(double start_time){
        return (MPI_Wtime() - start_time);
    }
#endif // USE_MPI