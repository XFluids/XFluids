#pragma once
#include "setup.h"
#include "sycl_kernels.h"
#include "fun.h"

// SYCL head files
#include <CL/sycl.hpp>
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

void InitializeFluidStates(sycl::queue &q, array<int, 3> WG, array<int, 3> WI, MaterialProperty *material, FlowData &fdata, 
                            Real* U, Real* U1, Real* LU,
                            Real* FluxF, Real* FluxG, Real* FluxH, 
                            Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                            Real const dx, Real const dy, Real const dz);