#pragma once

#include <chrono>
#ifdef USE_MPI
    #include <mpi.h>
#endif // USE_MPI

extern float OutThisTime(std::chrono::high_resolution_clock::time_point start_time);

#ifdef USE_MPI
    extern float OutThisTime(double start_time);
#endif // USE_MPI
