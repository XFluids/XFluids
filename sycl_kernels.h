#pragma once
#include "fun.h"

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

// 被SYCL内核调用的函数需要加"extern SYCL_EXTERNAL"
extern SYCL_EXTERNAL void InitialStatesKernel(int i, int j, int k, MaterialProperty* material, Real*  U, Real*  U1, Real*  LU, 
                                                    Real*  FluxF, Real*  FluxG, Real*  FluxH, 
                                                    Real*  FluxFw, Real*  FluxGw, Real*  FluxHw,
                                                    Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c,
                                                    Real dx, Real dy, Real dz);

extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real* rho, Real* u, Real* v, 
                                            Real* w, Real* H, Real dx, Real dy);


extern SYCL_EXTERNAL void testkernel(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c, Real const dx, Real const dy, Real const dz);