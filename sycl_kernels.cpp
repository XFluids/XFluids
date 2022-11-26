#include "device_func.h"
#include "setup.h"
#include "sycl_kernels.h"

void InitialStatesKernel(int i, int j, int k, MaterialProperty* material, Real*  U, Real*  U1, Real*  LU, 
                                                    Real*  FluxF, Real*  FluxG, Real*  FluxH, 
                                                    Real*  FluxFw, Real*  FluxGw, Real*  FluxHw,
                                                    Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c,
                                                    Real dx, Real dy, Real dz)
{
    int id = Xmax*Ymax*k + Xmax*j + i;

    #if DIM_X
    if(i >= Xmax)
        return;
    #endif
    #if DIM_Y
    if(j >= Ymax)
        return;
    #endif
    #if DIM_Z
    if(k >= Zmax)
        return;
    #endif

	#if USE_DP
	Real one_float = 1.0;
	Real half_float = 0.5;
	Real zero_float = 0.0;
	#else
	Real one_float = 1.0f;
	Real half_float = 0.5f;
	Real zero_float = 0.0f;
	#endif

    Real x = (i-Bwidth_X)*dx + half_float*dx;
    Real y = (j-Bwidth_Y)*dy + half_float*dy;
    Real z = (k-Bwidth_Z)*dz + half_float*dz;

    // 1d shock tube case
    #if USE_DP
    if(x < 0.5*DOMAIN_length) {
        rho[id] = 1.0;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        p[id] = 1.0;
    }
    else{
        rho[id] = 0.125;
        u[id] = 0.0;
        v[id] = 0.0;
        w[id] = 0.0;
        p[id] = 0.1;
    }
    #else
    if(x < 0.5f*DOMAIN_length) {
        rho[id] = 1.0f;
        u[id] = 0.0f;
        v[id] = 0.0f;
        w[id] = 0.0f;
        p[id] = 1.0f;
    }
    else{
        rho[id] = 0.125f;
        u[id] = 0.0f;
        v[id] = 0.0f;
        w[id] = 0.0f;
        p[id] = 0.1f;
    }
    #endif

    // // 2d sod case
    // d_rho[id] = 1.0; d_u[id] = 0;
	// d_v[id] = 0;		d_p[id] = 4.0e-13;
	// if(x<=dx && y<=dy)
	// 	d_p[id] = 9.79264/dx/dy*10000.0;

    // // two-phase
    // if(material.Rgn_ind>0.5){
    //     rho[id] = 0.125;
    //     u[id] = 0.0;
    //     v[id] = 0.0;
    //     w[id] = 0.0;
    //     p[id] = 0.1;
    // }
    // else
    // {
    //     rho[id] = 1.0;
    //     u[id] = 0.0;
    //     v[id] = 0.0;
    //     w[id] = 0.0;
    //     p[id] = 1.0;
    // }

    U[Emax*id+0] = rho[id];
    U[Emax*id+1] = rho[id]*u[id];
    U[Emax*id+2] = rho[id]*v[id];
    U[Emax*id+3] = rho[id]*w[id];
    //EOS was included
    if(material->Mtrl_ind == 0)
        U[Emax*id+4] = p[id] /(material->Gamma-one_float) + half_float*rho[id]*(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
    else
        U[Emax*id+4] = (p[id] + material->Gamma*(material->B-material->A))/(material->Gamma-one_float) 
                                            + half_float*rho[id]*(u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
    
    H[id]		= (U[Emax*id+4] + p[id])/rho[id];
    c[id]		= material->Mtrl_ind == 0 ? sqrt(material->Gamma*p[id]/rho[id]) : sqrt(material->Gamma*(p[id] + material->B - material->A)/rho[id]);

    //initial flux terms F, G, H
    FluxF[Emax*id+0] = U[Emax*id+1];
    FluxF[Emax*id+1] = U[Emax*id+1]*u[id] + p[id];
    FluxF[Emax*id+2] = U[Emax*id+1]*v[id];
    FluxF[Emax*id+3] = U[Emax*id+1]*w[id];
    FluxF[Emax*id+4] = (U[Emax*id+4] + p[id])*u[id];
    
    FluxG[Emax*id+0] = U[Emax*id+2];
    FluxG[Emax*id+1] = U[Emax*id+2]*u[id];
    FluxG[Emax*id+2] = U[Emax*id+2]*v[id] + p[id];
    FluxG[Emax*id+3] = U[Emax*id+2]*w[id];
    FluxG[Emax*id+4] = (U[Emax*id+4] + p[id])*v[id];

    FluxH[Emax*id+0] = U[Emax*id+3];
    FluxH[Emax*id+1] = U[Emax*id+3]*u[id];
    FluxH[Emax*id+2] = U[Emax*id+3]*v[id];
    FluxH[Emax*id+3] = U[Emax*id+3]*w[id] + p[id];
    FluxH[Emax*id+4] = (U[Emax*id+4] + p[id])*w[id];

    // Real fraction = material->Rgn_ind > 0.5 ? vof[id] : 1.0 - vof[id];

    //give intial value for the interval matrixes
    for(int n=0; n<Emax; n++){
	    LU[Emax*id+n] = 0.0; //incremental of one time step
	    U1[Emax*id+n] = U[Emax*id+n]; //intermediate conwervatives

        // CnsrvU[Emax*id+n] = U[Emax*id+n]*fraction;
        // CnsrvU1[Emax*id+n] = CnsrvU[Emax*id+n];

	    FluxFw[Emax*id+n] = 0.0; //numerical flux F
	    FluxGw[Emax*id+n] = 0.0; //numerical flux G
        FluxHw[Emax*id+n] = 0.0; //numerical flux H
    }
}

// add "sycl::nd_item<3> item" for get_global_id
// add "stream const s" for output
extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real* rho, Real* u, Real* v, 
                                            Real* w, Real* H, Real dx, Real dy)
{
    // // 利用get_global_id获得全局指标
    // int i = item.get_global_id(0) + Bwidth_X - 1;
    // int j = item.get_global_id(1) + Bwidth_Y;
	// int k = item.get_global_id(2) + Bwidth_Z;

    int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*j + i + 1;

	// cout<<"i,j,k = "<<i<<", "<<j<<", "<<k<<", "<<Emax*(Xmax*j + i-2)+0<<"\n";
	// printf("%f", UI[0]);

    if(i>= X_inner+Bwidth_X)
        return;

    // Real eigen_l[Emax][Emax], eigen_r[Emax][Emax];

	// //preparing some interval value for roe average
	// Real D	=	sqrt(rho[id_r]/rho[id_l]);
	// #if USE_DP
	// Real D1	=	1.0 / (D + 1.0);
	// #else
	// Real D1	=	1.0f / (D + 1.0f);
	// #endif
	// Real _u	=	(u[id_l] + D*u[id_r])*D1;
	// Real _v	=	(v[id_l] + D*v[id_r])*D1;
	// Real _w	=	(w[id_l] + D*w[id_r])*D1;
	// Real _H	=	(H[id_l] + D*H[id_r])*D1;
	// Real _rho = sqrt(rho[id_r]*rho[id_l]);

    // RoeAverage_x(eigen_l, eigen_r, _rho, _u, _v, _w, _H, D, D1);

    // Real uf[10],  ff[10], pp[10], mm[10];
    // Real f_flux, _p[Emax][Emax];

	// // construct the right value & the left value scalar equations by characteristic reduction			
	// // at i+1/2 in x direction
    // // #pragma unroll Emax
	// for(int n=0; n<Emax; n++){
    //     #if USE_DP
    //     Real eigen_local_max = 0.0;
    //     #else
    //     Real eigen_local_max = 0.0f;
    //     #endif
    //     for(int m=-2; m<=3; m++){
    //         int id_local = Xmax*Ymax*k + Xmax*j + i + m;
    //         eigen_local_max = sycl::max(eigen_local_max, fabs(eigen_local[Emax*id_local+n]));//local lax-friedrichs	
    //     }

	// 	for(int m=i-3; m<=i+4; m++){	// 3rd oder and can be modified
    //         int id_local = Xmax*Ymax*k + Xmax*j + m;
    //         #if USE_DP
	// 		uf[m-i+3] = 0.0;
	// 		ff[m-i+3] = 0.0;
    //         #else
	// 		uf[m-i+3] = 0.0f;
	// 		ff[m-i+3] = 0.0f;
    //         #endif

	// 		for(int n1=0; n1<Emax; n1++){
	// 			uf[m-i+3] = uf[m-i+3] + UI[Emax*id_local+n1]*eigen_l[n][n1];
	// 			ff[m-i+3] = ff[m-i+3] + Fx[Emax*id_local+n1]*eigen_l[n][n1];
	// 		}
	// 		// for local speed
	// 		pp[m-i+3] = 0.5f*(ff[m-i+3] + eigen_local_max*uf[m-i+3]);
	// 		mm[m-i+3] = 0.5f*(ff[m-i+3] - eigen_local_max*uf[m-i+3]);
    //     }

	// 	// calculate the scalar numerical flux at x direction
    //     #if USE_DP
    //     f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0;
    //     #else
    //     f_flux = (weno5old_P(&pp[3], dx) + weno5old_M(&mm[3], dx))/6.0f;
    //     #endif

	// 	// get Fp
	// 	for(int n1=0; n1<Emax; n1++)
	// 		_p[n][n1] = f_flux*eigen_r[n1][n];
    // }

	// // reconstruction the F-flux terms
	// for(int n=0; n<Emax; n++){
    //     #if USE_DP
    //     Real fluxx = 0.0;
    //     #else
    //     Real fluxx = 0.0f;
    //     #endif
	// 	for(int n1=0; n1<Emax; n1++) {
    //         fluxx += _p[n1][n];
	// 	}
    //     Fxwall[Emax*id_l+n] = fluxx;
	// }
}

extern SYCL_EXTERNAL void testkernel(int i, int j, int k, Real* UI, Real* Fx, Real* Fxwall, Real* eigen_local, Real*  u, Real*  v, Real*  w, Real*  rho,
                                                    Real*  p, Real*  H, Real*  c, Real const dx, Real const dy, Real const dz)
{
   int id_l = Xmax*Ymax*k + Xmax*j + i;
    int id_r = Xmax*Ymax*k + Xmax*j + i + 1;

	// cout<<"i,j,k = "<<i<<", "<<j<<", "<<k<<", "<<Emax*(Xmax*j + i-2)+0<<"\n";
	// printf("%f", UI[0]);

    if(i>= X_inner+Bwidth_X)
        return;

}