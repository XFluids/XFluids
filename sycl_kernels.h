#pragma once
#include "fun.h"

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

// 被SYCL内核调用的函数需要加"extern SYCL_EXTERNAL"
extern SYCL_EXTERNAL 
void InitialStatesKernel(int i, int j, int k, MaterialProperty* material, Real*  U, Real*  U1, Real*  LU, 
                                                    Real*  FluxF, Real*  FluxG, Real*  FluxH, 
                                                    Real*  FluxFw, Real*  FluxGw, Real*  FluxHw,
                                                    Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c,
                                                    Real dx, Real dy, Real dz);

extern SYCL_EXTERNAL 
void ReconstructFluxX(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real* rho, Real* u, Real* v, 
                                            Real* w, Real* H, Real dx);


extern SYCL_EXTERNAL 
void testkernel(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c, Real const dx, Real const dy, Real const dz);

extern SYCL_EXTERNAL 
void FluidBCKernelX(int i, int j, int k, BConditions const BC, Real* d_UI, int const mirror_offset, int const index_inner, int const sign);

extern SYCL_EXTERNAL 
void FluidBCKernelY(int i, int j, int k, BConditions const BC, Real* d_UI, int const mirror_offset, int const index_inner, int const sign);

extern SYCL_EXTERNAL 
void FluidBCKernelZ(int i, int j, int k, BConditions const BC, Real* d_UI, int const  mirror_offset, int const index_inner, int const sign);


extern SYCL_EXTERNAL 
void UpdateFuidStatesKernel(int i, int j, int k, Real*  UI, Real*  FluxF, Real*  FluxG, Real*  FluxH, Real*  rho, Real*  p, Real*  c, Real*  H, Real*  u, Real*  v, Real*  w, Real const Gamma);

extern SYCL_EXTERNAL 
void UpdateURK3rdKernel(int i, int j, int k, Real* U, Real* U1, Real* LU, Real const dt, int flag);

extern SYCL_EXTERNAL 
void UpdateFluidLU(int i, int j, int k, Real* LU, Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                    Real const dx, Real const dy, Real const dz);

extern SYCL_EXTERNAL 
void GetLocalEigen(int i, int j, int k, Real AA, Real BB, Real CC, Real* eigen_local, Real* u, Real* v, Real* w, Real* c);
