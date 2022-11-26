#pragma once

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

#include "setup.h"
#include "fun.h"

using namespace std;
using namespace sycl;

/**
 * @brief Obtain state at a grid point
 */
void GetStates(Real UI[Emax], Real &rho, Real &u, Real &v, Real &w, Real &p, Real &H, Real &c, Real const Gamma)
{
	rho	=	UI[0];
	#if USE_DP
	Real rho1 =	1.0/rho;
	#else
	Real rho1 =	1.0f/rho;
	#endif
	u	=	UI[1]*rho1;
	v	=	UI[2]*rho1;
	w	=	UI[3]*rho1;
	
	//EOS was included
	#if USE_DP
	p	=	(Gamma-1.0)*(UI[4] - 0.5*rho*(u*u + v*v + w*w));
	#else
	p	=	(Gamma-1.0f)*(UI[4] - 0.5f*rho*(u*u + v*v + w*w));
	#endif
	H	=	(UI[4] + p)*rho1;
	c	=	sqrt(Gamma*p*rho1);
}

/**
 * @brief  Obtain fluxes at a grid point
 */
void GetPhysFlux(Real UI[Emax], Real *FluxF, Real *FluxG, Real *FluxH, Real const rho, Real const u, Real const v, Real const w, Real const p, Real const H, Real const c)
{
	FluxF[0] = UI[1];
	// *(FluxF+0) = UI[1];
	FluxF[1] = UI[1]*u + p;
	FluxF[2] = UI[1]*v;
	FluxF[3] = UI[1]*w;
	FluxF[4] = (UI[4] + p)*u;

	FluxG[0] = UI[2];
	FluxG[1] = UI[2]*u;
	FluxG[2] = UI[2]*v + p;
	FluxG[3] = UI[2]*w;
	FluxG[4] = (UI[4] + p)*v;

	FluxH[0] = UI[3];
	FluxH[1] = UI[3]*u;
	FluxH[2] = UI[3]*v;
	FluxH[3] = UI[3]*w + p;
	FluxH[4] = (UI[4] + p)*w;
}

inline void RoeAverage_x(Real eigen_l[Emax][Emax], Real eigen_r[Emax][Emax], Real const _rho, Real const _u, Real const _v, Real const _w, 
	Real const _H, Real const D, Real const D1)
{
	//preparing some interval value
	#if USE_DP
	Real one_float = 1.0;
	Real half_float = 0.5;
	Real zero_float = 0.0;
	#else
	Real one_float = 1.0f;
	Real half_float = 0.5f;
	Real zero_float = 0.0f;
	#endif

	Real _Gamma = Gamma - one_float;
	Real _rho1 = one_float / _rho;
	Real q2 = _u*_u + _v*_v + _w*_w;
	Real c2 = _Gamma*(_H - half_float*q2);
	Real _c = sqrt(c2);
	Real _c1_rho = half_float*_rho / _c;
	Real c21_Gamma = _Gamma / c2;
	Real _c1_rho1_Gamma = _Gamma*_rho1 / _c;

	// left eigen vectors
	eigen_l[0][0] = one_float - half_float*c21_Gamma*q2;
	eigen_l[0][1] = c21_Gamma*_u;
	eigen_l[0][2] = c21_Gamma*_v;
	eigen_l[0][3] = c21_Gamma*_w;
	eigen_l[0][4] = -c21_Gamma;
	
	eigen_l[1][0] = -_w*_rho1;
	eigen_l[1][1] = zero_float;
	eigen_l[1][2] = zero_float;
	eigen_l[1][3] = _rho1;
	eigen_l[1][4] = zero_float;

	eigen_l[2][0] = _v*_rho1;
	eigen_l[2][1] = zero_float;
	eigen_l[2][2] = -_rho1;
	eigen_l[2][3] = zero_float;
	eigen_l[2][4] = zero_float;

	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _u*_rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma*_u + _rho1;
	eigen_l[3][2] = -_c1_rho1_Gamma*_v;
	eigen_l[3][3] = -_c1_rho1_Gamma*_w;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _u*_rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma*_u - _rho1;
	eigen_l[4][2] = -_c1_rho1_Gamma*_v;
	eigen_l[4][3] = -_c1_rho1_Gamma*_w;
	eigen_l[4][4] = _c1_rho1_Gamma;

	//right eigen vectors
	eigen_r[0][0] = one_float;
	eigen_r[0][1] = zero_float;
	eigen_r[0][2] = zero_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;
	
	eigen_r[1][0] = _u;
	eigen_r[1][1] = zero_float;
	eigen_r[1][2] = zero_float;
	eigen_r[1][3] = _c1_rho*(_u + _c);
	eigen_r[1][4] = _c1_rho*(_u - _c);

	eigen_r[2][0] = _v;
	eigen_r[2][1] = zero_float;
	eigen_r[2][2] = -_rho;
	eigen_r[2][3] = _c1_rho*_v;
	eigen_r[2][4] = _c1_rho*_v;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _rho;
	eigen_r[3][2] = zero_float;
	eigen_r[3][3] = _c1_rho*_w;
	eigen_r[3][4] = _c1_rho*_w;

	eigen_r[4][0] = half_float*q2;
	eigen_r[4][1] = _rho*_w;
	eigen_r[4][2] = -_rho*_v;
	eigen_r[4][3] = _c1_rho*(_H + _u*_c);
	eigen_r[4][4] = _c1_rho*(_H - _u*_c);
}

inline void RoeAverage_y(Real eigen_l[Emax][Emax], Real eigen_r[Emax][Emax], Real const _rho, Real const _u, Real const _v, Real const _w, 
	Real const _H, Real const D, Real const D1)
{
	//preparing some interval value
	#if USE_DP
	Real one_float = 1.0;
	Real half_float = 0.5;
	Real zero_float = 0.0;
	#else
	Real one_float = 1.0f;
	Real half_float = 0.5f;
	Real zero_float = 0.0f;
	#endif

	Real _Gamma = Gamma - one_float;
	Real _rho1 = one_float / _rho;
	Real q2 = _u*_u + _v*_v + _w*_w;
	Real c2 = _Gamma*(_H - half_float*q2);
	Real _c = sqrt(c2);
	Real _c1_rho = half_float*_rho / _c;
	Real c21_Gamma = _Gamma / c2;
	Real _c1_rho1_Gamma = _Gamma*_rho1 / _c;

	// left eigen vectors 
	eigen_l[0][0] = _w*_rho1;
	eigen_l[0][1] = zero_float;
	eigen_l[0][2] = zero_float;
	eigen_l[0][3] = - _rho1;
	eigen_l[0][4] = zero_float;
	
	eigen_l[1][0] = one_float - half_float*c21_Gamma*q2;
	eigen_l[1][1] = c21_Gamma*_u;
	eigen_l[1][2] = c21_Gamma*_v;
	eigen_l[1][3] = c21_Gamma*_w;
	eigen_l[1][4] = - c21_Gamma;

	eigen_l[2][0] = - _u*_rho1;
	eigen_l[2][1] = _rho1;
	eigen_l[2][2] = zero_float;
	eigen_l[2][3] = zero_float;
	eigen_l[2][4] = zero_float;

	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _v*_rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma*_u;
	eigen_l[3][2] = -_c1_rho1_Gamma*_v + _rho1;
	eigen_l[3][3] = -_c1_rho1_Gamma*_w;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _v*_rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma*_u;
	eigen_l[4][2] = -_c1_rho1_Gamma*_v - _rho1;
	eigen_l[4][3] = -_c1_rho1_Gamma*_w;
	eigen_l[4][4] = _c1_rho1_Gamma;

	//right eigen vectors
	eigen_r[0][0] = zero_float;
	eigen_r[0][1] = one_float;
	eigen_r[0][2] = zero_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;
	
	eigen_r[1][0] = zero_float;
	eigen_r[1][1] = _u;
	eigen_r[1][2] = _rho;
	eigen_r[1][3] = _c1_rho*_u;
	eigen_r[1][4] = _c1_rho*_u;

	eigen_r[2][0] = zero_float;
	eigen_r[2][1] = _v;
	eigen_r[2][2] = zero_float;
	eigen_r[2][3] = _c1_rho*(_v + _c);
	eigen_r[2][4] = _c1_rho*(_v - _c);

	eigen_r[3][0] = - _rho;
	eigen_r[3][1] = _w;
	eigen_r[3][2] = zero_float;
	eigen_r[3][3] = _c1_rho*_w;
	eigen_r[3][4] = _c1_rho*_w;

	eigen_r[4][0] = - _rho*_w;
	eigen_r[4][1] = half_float*q2;
	eigen_r[4][2] = _rho*_u;
	eigen_r[4][3] = _c1_rho*(_H + _v*_c);
	eigen_r[4][4] = _c1_rho*(_H - _v*_c);
}

inline void RoeAverage_z(Real eigen_l[Emax][Emax], Real eigen_r[Emax][Emax], Real const _rho, Real const _u, Real const _v, Real const _w, 
	Real const _H, Real const D, Real const D1)
{
	//preparing some interval value
	#if USE_DP
	Real one_float = 1.0;
	Real half_float = 0.5;
	Real zero_float = 0.0;
	#else
	Real one_float = 1.0f;
	Real half_float = 0.5f;
	Real zero_float = 0.0f;
	#endif

	Real _Gamma = Gamma - one_float;
	Real _rho1 = one_float / _rho;
	Real q2 = _u*_u + _v*_v + _w*_w;
	Real c2 = _Gamma*(_H - half_float*q2);
	Real _c = sqrt(c2);
	Real _c1_rho = half_float*_rho / _c;
	Real c21_Gamma = _Gamma / c2;
	Real _c1_rho1_Gamma = _Gamma*_rho1 / _c;

	// left eigen vectors 
	eigen_l[0][0] = - _v*_rho1;
	eigen_l[0][1] = zero_float;
	eigen_l[0][2] = _rho1;
	eigen_l[0][3] = zero_float;
	eigen_l[0][4] = zero_float;
	
	eigen_l[1][0] = _u*_rho1;
	eigen_l[1][1] = - _rho1;
	eigen_l[1][2] = zero_float;
	eigen_l[1][3] = zero_float;
	eigen_l[1][4] = zero_float;

	eigen_l[2][0] = one_float - half_float*c21_Gamma*q2; 
	eigen_l[2][1] = c21_Gamma*_u; 
	eigen_l[2][2] = c21_Gamma*_v; 
	eigen_l[2][3] = c21_Gamma*_w; 
	eigen_l[2][4] = - c21_Gamma;

	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _w*_rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma*_u;
	eigen_l[3][2] = -_c1_rho1_Gamma*_v;
	eigen_l[3][3] = -_c1_rho1_Gamma*_w + _rho1;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _w*_rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma*_u;
	eigen_l[4][2] = -_c1_rho1_Gamma*_v;
	eigen_l[4][3] = -_c1_rho1_Gamma*_w - _rho1;
	eigen_l[4][4] = _c1_rho1_Gamma;

	//right eigen vectors
	eigen_r[0][0] = zero_float;
	eigen_r[0][1] = zero_float;
	eigen_r[0][2] = one_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;
	
	eigen_r[1][0] = zero_float;
	eigen_r[1][1] = - _rho;
	eigen_r[1][2] = _u;
	eigen_r[1][3] = _c1_rho*_u;
	eigen_r[1][4] = _c1_rho*_u;

	eigen_r[2][0] = _rho;
	eigen_r[2][1] = zero_float;
	eigen_r[2][2] = _v;
	eigen_r[2][3] = _c1_rho*_v;
	eigen_r[2][4] = _c1_rho*_v;

	eigen_r[3][0] = zero_float;
	eigen_r[3][1] = zero_float;
	eigen_r[3][2] = _w;
	eigen_r[3][3] = _c1_rho*(_w + _c);
	eigen_r[3][4] = _c1_rho*(_w - _c);

	eigen_r[4][0] = _rho*_v;
	eigen_r[4][1] = -_rho*_u;
	eigen_r[4][2] = half_float*q2;
	eigen_r[4][3] = _c1_rho*(_H + _w*_c);
	eigen_r[4][4] = _c1_rho*(_H - _w*_c);
}

/**
 * @brief the 5th WENO Scheme
 * 
 * @param f 
 * @param delta 
 * @return Real 
 */
Real weno5old_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real a1, a2, a3;
//	Real w1, w2, w3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1); 
	v5 = *(f + k + 2);

	//smoothness indicator
//	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
//	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
//	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
//	   + 3.0*(v2 - v4)*(v2 - v4);
//	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
//	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	
        //weights
//      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
//      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
//      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
//      Real tw1 = 1.0/(a1 + a2 +a3); 
//      w1 = a1*tw1;
//      w2 = a2*tw1;
//      w3 = a3*tw1;

//      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
//      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
//      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

//      return (a1+a2+a3)/6.0;

        //return weighted average
//      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
  //              + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
    //            + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;

		#if USE_DP
		a1 = v1 - 2.0*v2 + v3;
        Real s1 = 13.0*a1*a1;
        a1 = v1 - 4.0*v2 + 3.0*v3;
        s1 += 3.0*a1*a1;
        a1 = v2 - 2.0*v3 + v4;
        Real s2 = 13.0*a1*a1;
        a1 = v2 - v4;
        s2 += 3.0*a1*a1;
        a1 = v3 - 2.0*v4 + v5;
        Real s3 = 13.0*a1*a1;
        a1 = 3.0*v3 - 4.0*v4 + v5;
        s3 += 3.0*a1*a1;
		#else
		a1 = v1 - 2.0f*v2 + v3;
        Real s1 = 13.0f*a1*a1;
        a1 = v1 - 4.0f*v2 + 3.0f*v3;
        s1 += 3.0f*a1*a1;
        a1 = v2 - 2.0f*v3 + v4;
        Real s2 = 13.0f*a1*a1;
        a1 = v2 - v4;
        s2 += 3.0f*a1*a1;
        a1 = v3 - 2.0f*v4 + v5;
        Real s3 = 13.0f*a1*a1;
        a1 = 3.0f*v3 - 4.0f*v4 + v5;
        s3 += 3.0f*a1*a1;
		#endif

    	// a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
    	// a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
    	// a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
    	// Real tw1 = 1.0/(a1 + a2 +a3); 
    	// a1 = a1*tw1;
    	// a2 = a2*tw1;
    	// a3 = a3*tw1;
        Real tol = 1.0e-6;
		#if USE_DP
        a1 = 0.1*(tol + s2)*(tol + s2)*(tol + s3)*(tol + s3);
        a2 = 0.2*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
        a3 = 0.3*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
        Real tw1 = 1.0/(a1 + a2 +a3);
		#else
        a1 = 0.1f*(tol + s2)*(tol + s2)*(tol + s3)*(tol + s3);
        a2 = 0.2f*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
        a3 = 0.3f*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
        Real tw1 = 1.0f/(a1 + a2 +a3);
		#endif

    	a1 = a1*tw1;
    	a2 = a2*tw1;
    	a3 = a3*tw1;

		#if USE_DP
    	s1 = a1*(2.0*v1 - 7.0*v2 + 11.0*v3);
    	s2 = a2*(-v2 + 5.0*v3 + 2.0*v4);
    	s3 = a3*(2.0*v3 + 5.0*v4 - v5);
		#else
    	s1 = a1*(2.0f*v1 - 7.0f*v2 + 11.0f*v3);
    	s2 = a2*(-v2 + 5.0f*v3 + 2.0f*v4);
    	s3 = a3*(2.0f*v3 + 5.0f*v4 - v5);
		#endif

    	// return (s1+s2+s3)/6.0;
		return (s1+s2+s3);

        // a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
        // a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
        // a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
        // Real tw1 = 6.0*(a1 + a2 + a3);
        // return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
}
/**
 * @brief the 5th WENO Scheme
 * 
 * @param f 
 * @param delta 
 * @return Real 
 */
Real weno5old_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real a1, a2, a3;
//	Real w1, w2, w3;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1); 
	v5 = *(f + k - 2);

	//smoothness indicator
//	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
//	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
//	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
//	   + 3.0*(v2 - v4)*(v2 - v4);
//	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
//	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

        //weights
//      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
//      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
//      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
//      Real tw1 = 1.0/(a1 + a2 +a3); 
//      w1 = a1*tw1;
//      w2 = a2*tw1;
//      w3 = a3*tw1;

//      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
//      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
//      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

//      return (a1+a2+a3)/6.0;

        //return weighted average
//      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
//                + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
//                + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;	

		#if USE_DP
        a1 = v1 - 2.0*v2 + v3;
        Real s1 = 13.0*a1*a1;
        a1 = v1 - 4.0*v2 + 3.0*v3;
        s1 += 3.0*a1*a1;
        a1 = v2 - 2.0*v3 + v4;
        Real s2 = 13.0*a1*a1;
        a1 = v2 - v4;
        s2 += 3.0*a1*a1;
        a1 = v3 - 2.0*v4 + v5;
        Real s3 = 13.0*a1*a1;
        a1 = 3.0*v3 - 4.0*v4 + v5;
        s3 += 3.0*a1*a1;
		#else
        a1 = v1 - 2.0f*v2 + v3;
        Real s1 = 13.0f*a1*a1;
        a1 = v1 - 4.0f*v2 + 3.0f*v3;
        s1 += 3.0f*a1*a1;
        a1 = v2 - 2.0f*v3 + v4;
        Real s2 = 13.0f*a1*a1;
        a1 = v2 - v4;
        s2 += 3.0f*a1*a1;
        a1 = v3 - 2.0f*v4 + v5;
        Real s3 = 13.0f*a1*a1;
        a1 = 3.0f*v3 - 4.0f*v4 + v5;
        s3 += 3.0f*a1*a1;
		#endif

		//  a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
		//  a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
		//  a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
		//  Real tw1 = 1.0/(a1 + a2 +a3); 
		//  a1 = a1*tw1;
		//  a2 = a2*tw1;
		//  a3 = a3*tw1;
        Real tol = 1.0e-6;
		#if USE_DP
		a1 = 0.1*(tol+ s2)*(tol + s2)*(tol + s3)*(tol + s3);
        a2 = 0.2*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
        a3 = 0.3*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
        Real tw1 = 1.0/(a1 + a2 +a3);
		#else
		a1 = 0.1f*(tol+ s2)*(tol + s2)*(tol + s3)*(tol + s3);
        a2 = 0.2f*(tol + s1)*(tol + s1)*(tol + s3)*(tol + s3);
        a3 = 0.3f*(tol + s1)*(tol + s1)*(tol + s2)*(tol + s2);
        Real tw1 = 1.0f/(a1 + a2 +a3);
		#endif
         a1 = a1*tw1;
         a2 = a2*tw1;
         a3 = a3*tw1;

		#if USE_DP
		s1 = a1*(2.0*v1 - 7.0*v2 + 11.0*v3);
		s2 = a2*(-v2 + 5.0*v3 + 2.0*v4);
		s3 = a3*(2.0*v3 + 5.0*v4 - v5);
		#else
		s1 = a1*(2.0f*v1 - 7.0f*v2 + 11.0f*v3);
		s2 = a2*(-v2 + 5.0f*v3 + 2.0f*v4);
		s3 = a3*(2.0f*v3 + 5.0f*v4 - v5);
		#endif
 
		//  return (s1+s2+s3)/6.0;
		return (s1+s2+s3);

        //  a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
        //  a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
        //  a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
        //  Real tw1 = 6.0*(a1 + a2 + a3);
        //  return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
}