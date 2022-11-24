#include "global_function.h"

using namespace std;

Real KroneckerDelta(const int i, const int j)
{
	Real f = i ==j ? 1 : 0;
	return f;
}
/**
 * @brief Roe average of value u_l and u_r
 * 
 * @param ul left value
 * @param ur right value
 * @param D 
 * @return Real 
 */
Real RoeAverage(const Real ul, const Real ur, const Real D, const Real D1)
{
	return (ul + D*ur)*D1;
}
/**
 * @brief van Leer limiter
 * 
 * @param r 
 * @return Real 
 */
Real van_Leer(Real r)
{
	return (r+std::abs(r))/(1.0+std::abs(r));
}
/**
 * @brief van Albada limiter
 * 
 * @param r 
 * @return Real 
 */
Real van_Albada(Real r)
{
	return (r*r+r)/(1.0+r*r);
}
/**
 * @brief the minmod limiter
 * 
 * @param r 
 * @return Real 
 */
Real minmod(Real r)
{
	Real minmod = 0;
    Real aa = 1.0;
    if(r>0)
		minmod = std::min(r,aa);
	return minmod;
}
/**
 * @brief the MUSCL reconstruction
 * 
 * @param p 
 * @param LL 
 * @param RR 
 * @param flag 
 */
void MUSCL(Real p[4], Real &LL, Real &RR, int flag)
{
	Real tol = 1e-20, k=1.0/3.0;
	Real a0 = p[1] - p[0], a1 = p[2] - p[1], a2 = p[3] - p[2];
	if(a0 == -tol || a2 == -tol || a1 == -tol)
		tol *= 0.1;

	Real r1 = a1/(a0+tol), r2 = a1/(a2+tol), r11 = a0/(a1+tol), r22 = a2/(a1+tol);
	Real LL1 = 0, LL2 = 0, LR1 = 0, LR2 = 0;
	switch(flag){
		case 1:
			LL1 = minmod(r1);	LL2 = minmod(r11);
			LR1 = minmod(r2);	LR2 = minmod(r22);
		break;
		case 2:
			LL1 = van_Leer(r1);	LL2 = van_Leer(r11);
			LR1 = van_Leer(r2);	LR2 = van_Leer(r22);
		break;
		case 3:
			LL1 = van_Albada(r1);	LL2 = van_Albada(r11);
			LR1 = van_Albada(r2);	LR2 = van_Albada(r22);
		break;						
	}
    LL = p[1] + 0.25*( (1.0-k)*LL1 + (1.0+k)*LL2*r1 )*a0;
    RR = p[2] - 0.25*( (1.0-k)*LR1 + (1.0+k)*LR2*r2 )*a2;
}
/**
 * @brief upwind scheme
 * 
 * @param f 
 * @param delta 
 * @return Real 
 */
Real upwind_P(Real *f, Real delta)
{
	return *f;
}
Real upwind_M(Real *f, Real delta)
{
	return *(f+1);
}
Real linear_3rd_P(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    Real vv = -(-2.0*v4-5.0*v3+v2)/6.0;

	return vv;
}
Real linear_3rd_M(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    Real vv = (-v5+5.0*v4+2*v3)/6.0;

	return vv;
}
Real linear_5th_P(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    Real vv = (2.0*v1 - 13.0*v2 + 47.0*v3 + 27.0*v4 - 3.0*v5)/60.0;

	return vv;
}
Real linear_5th_M(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    Real vv = (2.0*v6 - 13.0*v5 + 47.0*v4 + 27.0*v3 - 3.0*v2)/60.0;

	return vv;
}
Real linear_2th(Real *f, Real delta)
{
    Real v1 = *f;
    Real v2 = *(f + 1); 
    Real vv = (v1 + v2)/2.0;

	return vv;
}
Real linear_4th(Real *f, Real delta)
{
    Real v1 = *(f - 1);
    Real v2 = *f;
    Real v3 = *(f + 1); 
    Real v4 = *(f + 2);
    Real vv = (-v1 + 7.0*v2 + 7.0*v3 - v4)/12.0;

	return vv;
}
Real linear_6th(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    Real vv = (v1 - 8.0*v2 + 37.0*v3 + 37.0*v4 - 8.0*v5 + v6)/60.0;

	return vv;
}
//-------------------------------------------------------------------------------------------------
//  linear scheme for hybrid method
//-------------------------------------------------------------------------------------------------
Real du_upwind5(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    return (v1 - 5.0*v2 + 10.0*v3 - 10.0*v4 + 5.0*v5 - v6)/60.0;
}
//-------------------------------------------------------------------------------------------------
//  linear scheme for hybrid method
//-------------------------------------------------------------------------------------------------
Real f2_upwind5(Real *f, Real delta)
{
    Real v1 = *(f - 2);
    Real v2 = *(f - 1);
    Real v3 = *f;
    Real v4 = *(f + 1); 
    Real v5 = *(f + 2);
    Real v6 = *(f + 3);
    return (v1 - 8.0*v2 + 37.0*v3 + 37.0*v4 - 8.0*v5 + v6)/60.0;
}

Real weno_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real s1, s2, s3;
	Real a1, a2, a3, w1, w2, w3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	//smoothness indicator
	s1 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
	   + 0.25*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	s2 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
	   + 0.25*(v2 - v4)*(v2 - v4);
	s3 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
	   + 0.25*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

	//weights
	a1 = 0.1/(1.0e-6 + s1)/(1.0e-15 + s1);
	a2 = 0.6/(1.0e-6 + s2)/(1.0e-15 + s2);
	a3 = 0.3/(1.0e-6 + s3)/(1.0e-15 + s3);

	w1 = a1/(a1 + a2 + a3);
	w2 = a2/(a1 + a2 + a3);
	w3 = a3/(a1 + a2 + a3);

	//return weighted average
	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
}
Real weno_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real s1, s2,s3;
	Real a1, a2,a3, w1, w2, w3;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);

	//smoothness indicator
	s1 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
	   + 0.25*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	s2 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
	   + 0.25*(v2 - v4)*(v2 - v4);
	s3 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
	   + 0.25*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

	//weights
	a1 = 0.1/(1.0e-6 + s1)/(1.0e-15 + s1);
	a2 = 0.6/(1.0e-6 + s2)/(1.0e-15 + s2);
	a3 = 0.3/(1.0e-6 + s3)/(1.0e-15 + s3);

	w1 = a1/(a1 + a2 +a3);
	w2 = a2/(a1 + a2 +a3);
	w3 = a3/(a1 + a2 +a3);

	//return weighted average
	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
}

// Real weno_P(Real *f, Real delta)
// {
// 	int k;
// 	Real v1, v2, v3, v4, v5;
// 	Real a1, a2, a3, w1, w2, w3;

// 	//assign value to v1, v2,...
// 	k = 0;
// 	v1 = *(f + k - 2);
// 	v2 = *(f + k - 1);
// 	v3 = *(f + k);
// 	v4 = *(f + k + 1); 
// 	v5 = *(f + k + 2);

// 	//smoothness indicator
// 	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
// 	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
// 	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
// 	   + 3.0*(v2 - v4)*(v2 - v4);
// 	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
// 	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

// 	//weights
// 	a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
// 	a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
// 	a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
// 	Real tw1 = 1.0/(a1 + a2 +a3); 
// 	w1 = a1*tw1;
// 	w2 = a2*tw1;
// 	w3 = a3*tw1;

// 	//return weighted average
// 	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
// 		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
// 		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
// }
// Real weno_M(Real *f, Real delta)
// {
// 	int k;
// 	Real v1, v2, v3, v4, v5;
// 	Real a1, a2, a3, w1, w2, w3;

// 	//assign value to v1, v2,...
// 	k = 1;
// 	v1 = *(f + k + 2);
// 	v2 = *(f + k + 1);
// 	v3 = *(f + k);
// 	v4 = *(f + k - 1); 
// 	v5 = *(f + k - 2);

// 	//smoothness indicator
// 	Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
// 	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
// 	Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
// 	   + 3.0*(v2 - v4)*(v2 - v4);
// 	Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
// 	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

// 	//weights
// 	a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
// 	a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
// 	a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
// 	Real tw1 = 1.0/(a1 + a2 +a3); 
// 	w1 = a1*tw1;
// 	w2 = a2*tw1;
// 	w3 = a3*tw1;

// 	//return weighted average
// 	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
// 		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
// 		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
// }


/**
 * @brief  WENO-AO(5,3) scheme from
 *                  Balsara et al., An efficient class of WENO schemes with adaptive order. (2016)
 *                  Kumar et al., Simple smoothness indicator and multi-level adaptive order WENO scheme for hyperbolic conservation laws. (2018)
 */
Real WENOAO53_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real a1, a2, a3, a5, w1, w2, w3, w5;

	//assign value to v1, v2,...
 	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1); 
	v5 = *(f + k + 2);

    //smoothness indicator
    Real s11 = 1.0/2.0*v1 - 2.0*v2 + 3.0/2.0*v3, s12 = 1.0/2.0*v1 - 1.0*v2 + 1.0/2.0*v3;
    Real s21 = -1.0/2.0*v2 + 1.0/2.0*v4, s22 = 1.0/2.0*v2 - v3 + 1.0/2.0*v4;
    Real s31 = -3.0/2.0*v3 + 2.0*v4 -1.0/2.0*v5, s32 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s51 =  11.0/120.0 * v1 - 82.0/120.0 * v2 + 82.0/120.0 * v4 - 11.0/120.0 * v5;
    Real s52 = -3.0/56.0 * v1 + 40.0/56.0 * v2 + -74.0/56.0 * v3 + 40.0/56.0 * v4  -3.0/56.0 * v5;
    Real s53 = -1.0/12.0 * v1 + 2.0/12.0 * v2 - 2.0/12.0 * v4 + 1.0/12.0 * v5;
    Real s54 = 1.0/24.0 * v1 - 4.0/24.0 * v2 + 6.0/24.0 * v3  - 4.0/24.0 * v4 + 1.0/24.0 * v5;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s5 = 1.0 * (s51 + 1.0/10.0 * s53) * (s51 + 1.0/10.0 * s53)
                            + 13.0/3.0 * (s52 + 123.0/455.0 * s54) * (s52 + 123.0/455.0 * s54)
                            + 781.0/20.0 * s53 * s53 + 1421461.0/2275.0 * s54 *s54;

    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau = (std::abs( s5 - s1 ) + std::abs( s5 - s2 ) + std::abs( s5 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_5 = 0.85;
    a1 = coef_weights_1 * ( 1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)) );
    a2 = coef_weights_2 * ( 1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)) );
    a3 = coef_weights_3 * ( 1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)) );
    a5 = coef_weights_5 * ( 1.0 + (tau * tau) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)) );
    Real tw1 = 1.0/(a1 + a2 +a3 + a5);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;
    w5 = a5 * tw1;

    // Compute coefficients of the Legendre basis polynomial
    Real u0 = v3;
    Real u1 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                            + w1 * s11 + w2 * s21 + w3 * s31;
    Real u2 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                            + w1 * s12 + w2 * s22 + w3 * s32;
    Real u3 = (w5 / coef_weights_5) * s53;
    Real u4 = (w5 / coef_weights_5) * s54;   
    // Return value of reconstructed polynomial
    return  u0 + u1 * 1.0/2.0 + u2 * 1.0/6.0 + u3 * 1.0/20.0 + u4 * 1.0/70.0;
}

Real WENOAO53_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real a1, a2, a3, a5, w1, w2, w3, w5;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1); 
	v5 = *(f + k - 2);

   //smoothness indicator
    Real s11 = 1.0/2.0*v1 - 2.0*v2 + 3.0/2.0*v3, s12 = 1.0/2.0*v1 - 1.0*v2 + 1.0/2.0*v3;
    Real s21 = -1.0/2.0*v2 + 1.0/2.0*v4, s22 = 1.0/2.0*v2 - v3 + 1.0/2.0*v4;
    Real s31 = -3.0/2.0*v3 + 2.0*v4 -1.0/2.0*v5, s32 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s51 =  11.0/120.0 * v1 - 82.0/120.0 * v2 + 82.0/120.0 * v4 - 11.0/120.0 * v5;
    Real s52 = -3.0/56.0 * v1 + 40.0/56.0 * v2 + -74.0/56.0 * v3 + 40.0/56.0 * v4  -3.0/56.0 * v5;
    Real s53 = -1.0/12.0 * v1 + 2.0/12.0 * v2 - 2.0/12.0 * v4 + 1.0/12.0 * v5;
    Real s54 = 1.0/24.0 * v1 - 4.0/24.0 * v2 + 6.0/24.0 * v3  - 4.0/24.0 * v4 + 1.0/24.0 * v5;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s5 = 1.0 * (s51 + 1.0/10.0 * s53) * (s51 + 1.0/10.0 * s53)
                            + 13.0/3.0 * (s52 + 123.0/455.0 * s54) * (s52 + 123.0/455.0 * s54)
                            + 781.0/20.0 * s53 * s53 + 1421461.0/2275.0 * s54 *s54;

    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau = (std::abs( s5 - s1 ) + std::abs( s5 - s2 ) + std::abs( s5 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_5 = 0.85;
    a1 = coef_weights_1 * ( 1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)) );
    a2 = coef_weights_2 * ( 1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)) );
    a3 = coef_weights_3 * ( 1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)) );
    a5 = coef_weights_5 * ( 1.0 + (tau * tau) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)) );
    Real tw1 = 1.0/(a1 + a2 +a3 + a5);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;
    w5 = a5 * tw1;

    // Compute coefficients of the Legendre basis polynomial
    Real u0 = v3;
    Real u1 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                            + w1 * s11 + w2 * s21 + w3 * s31;
    Real u2 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                            + w1 * s12 + w2 * s22 + w3 * s32;
    Real u3 = (w5 / coef_weights_5) * s53;
    Real u4 = (w5 / coef_weights_5) * s54;
    // Return value of reconstructed polynomial
    return  u0 + u1 * 1.0/2.0 + u2 * 1.0/6.0 + u3 * 1.0/20.0 + u4 * 1.0/70.0;
}
/**
 * @brief  WENO-AO(7,3) scheme from Balsara (2016)
 */
Real WENOAO73_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6, v7;
    Real u0, u1, u2, u3, u4, u5, u6;
    Real a1, a2, a3, a7;
    Real w1, w2, w3, w7;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 3);
	v2 = *(f + k - 2);
    v3 = *(f + k - 1);
    v4 = *(f + k); 
    v5 = *(f + k + 1);
    v6 = *(f + k + 2);
    v7 = *(f + k + 3);

    //smoothness indicator
    Real s11 = 1.0/2.0*v2 - 2.0*v3 + 3.0/2.0*v4, s12 = 1.0/2.0*v2 - 1.0*v3 + 1.0/2.0*v4;
    Real s21 = -1.0/2.0*v3 + 1.0/2.0*v5, s22 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s31 = -3.0/2.0*v4 + 2.0*v5 -1.0/2.0*v6, s32 = 1.0/2.0*v4 - v5 + 1.0/2.0*v6;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s71 = -191.0/10080.0 * v1 + 1688.0/10080.0 * v2 - 7843.0/10080.0 * v3 + 7843.0/10080.0 * v5 - 1688.0/10080.0 * v6 + 191.0/10080.0 * v7;
    Real s72 =  79.0/10080.0 * v1 - 1014.0/10080.0 * v2 + 8385.0/10080.0 * v3 - 14900.0/10080.0 * v4 + 8385.0/10080.0 * v5 - 1014.0/10080.0 * v6 + 79.0/10080.0 * v7;
    Real s73 = 5.0/216.0 * v1 - 38.0/216.0 * v2 + 61.0/216.0 * v3 - 61.0/216.0 * v5 + 38.0/216.0 * v6 - 5.0/216.0 * v7;
    Real s74 = -13.0/1584.0 * v1 + 144.0/1584.0 * v2 + 459.0/1584.0 * v3 + 656.0/1584.0 * v4 - 459.0/1584.0 * v5 + 144.0/1584.0 * v6 - 13.0/1584.0 * v7;
    Real s75 = -1.0/240.0 * v1 + 4.0/240.0 * v2 - 5.0/240.0 * v3 + 5.0/240.0 * v5 - 4.0/240.0 * v6 + 1/240.0 * v7;
    Real s76 = 1.0/720.0 * v1 - 6.0/720.0 * v2 + 15.0/720.0 * v3 - 20.0/720.0 * v4 + 15.0/720.0 * v5 - 6.0/720.0 * v6 + 1/720.0 * v7;
    Real s7 = 1.0 * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75) * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75)
                        + 13.0/3.0 * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76) * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76)
                        + 781.0/20.0 * (s73 + 26045.0/49203.0 * s75) * (s73 + 26045.0/49203.0 * s75)
                        + 1421461.0/2275.0 * (s74 + 81596225.0/93816426.0 * s76) * (s74 + 81596225.0/93816426.0 * s76)
                        + 21520059541.0/1377684.0 * s75 * s75 + 15510384942580921.0/27582029244.0 * s76 * s76;
    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau = (std::abs( s7 - s1 ) + std::abs( s7 - s2 ) + std::abs( s7 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_7 = 0.85;
    a1 = coef_weights_1 * ( 1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)) );
    a2 = coef_weights_2 * ( 1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)) );
    a3 = coef_weights_3 * ( 1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)) );
    a7 = coef_weights_7 * ( 1.0 + (tau * tau) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)) );
    Real tw1 = 1.0/(a1 + a2 +a3 + a7);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;
    w7 = a7 * tw1;

    // Compute coefficients of the Legendre basis polynomial
    u0 = v4;
    u1 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
            + w1 * s11 + w2 * s21 + w3 * s31;
    u2 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
            + w1 * s12 + w2 * s22 + w3 * s32;
    u3 = (w7 / coef_weights_7) * s73;
    u4 = (w7 / coef_weights_7) * s74;
    u5 = (w7 / coef_weights_7) * s75;
    u6 = (w7 / coef_weights_7) * s76;

    // Return value of reconstructed polynomial
    return  u0 + u1 * 1.0/2.0 + u2 * 1.0/6.0 + u3 * 1.0/20.0 + u4 * 1.0/70.0 + u5 * 1.0/252.0 + u6 * 1.0/924.0;    
}
Real WENOAO73_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6, v7;
    Real u0, u1, u2, u3, u4, u5, u6;
    Real a1, a2, a3, a7;
    Real w1, w2, w3, w7;

	//assign value to v1, v2,...
	k = 1;
    v1 = *(f + k + 3);
    v2 = *(f + k + 2);
    v3 = *(f + k + 1);
    v4 = *(f + k); 
    v5 = *(f + k - 1);
    v6 = *(f + k - 2);
    v7 = *(f + k - 3);

    //smoothness indicator
    Real s11 = 1.0/2.0*v2 - 2.0*v3 + 3.0/2.0*v4, s12 = 1.0/2.0*v2 - 1.0*v3 + 1.0/2.0*v4;
    Real s21 = -1.0/2.0*v3 + 1.0/2.0*v5, s22 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s31 = -3.0/2.0*v4 + 2.0*v5 -1.0/2.0*v6, s32 = 1.0/2.0*v4 - v5 + 1.0/2.0*v6;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s71 = -191.0/10080.0 * v1 + 1688.0/10080.0 * v2 - 7843.0/10080.0 * v3 + 7843.0/10080.0 * v5 - 1688.0/10080.0 * v6 + 191.0/10080.0 * v7;
    Real s72 =  79.0/10080.0 * v1 - 1014.0/10080.0 * v2 + 8385.0/10080.0 * v3 - 14900.0/10080.0 * v4 + 8385.0/10080.0 * v5 - 1014.0/10080.0 * v6 + 79.0/10080.0 * v7;
    Real s73 = 5.0/216.0 * v1 - 38.0/216.0 * v2 + 61.0/216.0 * v3 - 61.0/216.0 * v5 + 38.0/216.0 * v6 - 5.0/216.0 * v7;
    Real s74 = -13.0/1584.0 * v1 + 144.0/1584.0 * v2 + 459.0/1584.0 * v3 + 656.0/1584.0 * v4 - 459.0/1584.0 * v5 + 144.0/1584.0 * v6 - 13.0/1584.0 * v7;
    Real s75 = -1.0/240.0 * v1 + 4.0/240.0 * v2 - 5.0/240.0 * v3 + 5.0/240.0 * v5 - 4.0/240.0 * v6 + 1/240.0 * v7;
    Real s76 = 1.0/720.0 * v1 - 6.0/720.0 * v2 + 15.0/720.0 * v3 - 20.0/720.0 * v4 + 15.0/720.0 * v5 - 6.0/720.0 * v6 + 1/720.0 * v7;
    Real s7 = 1.0 * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75) * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75)
                        + 13.0/3.0 * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76) * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76)
                        + 781.0/20.0 * (s73 + 26045.0/49203.0 * s75) * (s73 + 26045.0/49203.0 * s75)
                        + 1421461.0/2275.0 * (s74 + 81596225.0/93816426.0 * s76) * (s74 + 81596225.0/93816426.0 * s76)
                        + 21520059541.0/1377684.0 * s75 * s75 + 15510384942580921.0/27582029244.0 * s76 * s76;
    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau = (std::abs( s7 - s1 ) + std::abs( s7 - s2 ) + std::abs( s7 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_7 = 0.85;
    a1 = coef_weights_1 * ( 1.0 + (tau * tau) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)) );
    a2 = coef_weights_2 * ( 1.0 + (tau * tau) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)) );
    a3 = coef_weights_3 * ( 1.0 + (tau * tau) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)) );
    a7 = coef_weights_7 * ( 1.0 + (tau * tau) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)) );
    Real tw1 = 1.0/(a1 + a2 +a3 + a7);
    w1 = a1 * tw1;
    w2 = a2 * tw1;
    w3 = a3 * tw1;
    w7 = a7 * tw1;

    // Compute coefficients of the Legendre basis polynomial
    u0 = v4;
    u1 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
            + w1 * s11 + w2 * s21 + w3 * s31;
    u2 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
            + w1 * s12 + w2 * s22 + w3 * s32;
    u3 = (w7 / coef_weights_7) * s73;
    u4 = (w7 / coef_weights_7) * s74;
    u5 = (w7 / coef_weights_7) * s75;
    u6 = (w7 / coef_weights_7) * s76;

    // Return value of reconstructed polynomial
    return  u0 + u1 * 1.0/2.0 + u2 * 1.0/6.0 + u3 * 1.0/20.0 + u4 * 1.0/70.0 + u5 * 1.0/252.0 + u6 * 1.0/924.0;    
}
/**
 * @brief  WENO-AO(7,5,3) scheme from Balsara (2016)
 */
Real WENOAO753_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6, v7;
    Real u0_5, u1_5, u2_5, u3_5, u4_5;
    Real u0_7, u1_7, u2_7, u3_7, u4_7, u5_7, u6_7;
    Real a1_5, a2_5, a3_5, a1_7, a2_7, a3_7, a5, a7;
    Real w1_5, w2_5, w3_5, w1_7, w2_7, w3_7, w5, w7;    

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 3);
	v2 = *(f + k - 2);
    v3 = *(f + k - 1);
    v4 = *(f + k); 
    v5 = *(f + k + 1);
    v6 = *(f + k + 2);
    v7 = *(f + k + 3);

	//smoothness indicator
    Real s11 = 1.0/2.0*v2 - 2.0*v3 + 3.0/2.0*v4, s12 = 1.0/2.0*v2 - 1.0*v3 + 1.0/2.0*v4;
    Real s21 = -1.0/2.0*v3 + 1.0/2.0*v5, s22 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s31 = -3.0/2.0*v4 + 2.0*v5 -1.0/2.0*v6, s32 = 1.0/2.0*v4 - v5 + 1.0/2.0*v6;
    Real s51 =  11.0/120.0 * v2 - 82.0/120.0 * v3 + 82.0/120.0 * v5 - 11.0/120.0 * v6;
    Real s52 = -3.0/56.0 * v2 + 40.0/56.0 * v3 + -74.0/56.0 * v4 + 40.0/56.0 * v5  -3.0/56.0 * v6;
    Real s53 = -1.0/12.0 * v2 + 2.0/12.0 * v3 - 2.0/12.0 * v5 + 1.0/12.0 * v6;
    Real s54 = 1.0/24.0 * v2 - 4.0/24.0 * v3 + 6.0/24.0 * v4  - 4.0/24.0 * v5 + 1.0/24.0 * v6;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s5 = 1.0 * (s51 + 1.0/10.0 * s53) * (s51 + 1.0/10.0 * s53)
                            + 13.0/3.0 * (s52 + 123.0/455.0 * s54) * (s52 + 123.0/455.0 * s54)
                            + 781.0/20.0 * s53 * s53 + 1421461.0/2275.0 * s54 *s54;

    Real s71 = -191.0/10080.0 * v1 + 1688.0/10080.0 * v2 - 7843.0/10080.0 * v3 + 7843.0/10080.0 * v5 - 1688.0/10080.0 * v6 + 191.0/10080.0 * v7;
    Real s72 =  79.0/10080.0 * v1 - 1014.0/10080.0 * v2 + 8385.0/10080.0 * v3 - 14900.0/10080.0 * v4 + 8385.0/10080.0 * v5 - 1014.0/10080.0 * v6 + 79.0/10080.0 * v7;
    Real s73 = 5.0/216.0 * v1 - 38.0/216.0 * v2 + 61.0/216.0 * v3 - 61.0/216.0 * v5 + 38.0/216.0 * v6 - 5.0/216.0 * v7;
    Real s74 = -13.0/1584.0 * v1 + 144.0/1584.0 * v2 + 459.0/1584.0 * v3 + 656.0/1584.0 * v4 - 459.0/1584.0 * v5 + 144.0/1584.0 * v6 - 13.0/1584.0 * v7;
    Real s75 = -1.0/240.0 * v1 + 4.0/240.0 * v2 - 5.0/240.0 * v3 + 5.0/240.0 * v5 - 4.0/240.0 * v6 + 1/240.0 * v7;
    Real s76 = 1.0/720.0 * v1 - 6.0/720.0 * v2 + 15.0/720.0 * v3 - 20.0/720.0 * v4 + 15.0/720.0 * v5 - 6.0/720.0 * v6 + 1/720.0 * v7;
    Real s7 = 1.0 * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75) * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75)
                        + 13.0/3.0 * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76) * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76)
                        + 781.0/20.0 * (s73 + 26045.0/49203.0 * s75) * (s73 + 26045.0/49203.0 * s75)
                        + 1421461.0/2275.0 * (s74 + 81596225.0/93816426.0 * s76) * (s74 + 81596225.0/93816426.0 * s76)
                        + 21520059541.0/1377684.0 * s75 * s75 + 15510384942580921.0/27582029244.0 * s76 * s76;

    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau_7 = (std::abs( s7 - s1 ) + std::abs( s7 - s2 ) + std::abs( s7 - s3 )) / 3.0;
    Real tau_5 = (std::abs( s5 - s1 ) + std::abs( s5 - s2 ) + std::abs( s5 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_5 = 0.85;
    Real coef_weights_7 = 0.85;
    
    a1_7 = coef_weights_1 * ( 1.0 + (tau_7 * tau_7) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
    a2_7 = coef_weights_2 * ( 1.0 + (tau_7 * tau_7) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
    a3_7 = coef_weights_3 * ( 1.0 + (tau_7 * tau_7) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
    a7 = coef_weights_7 * ( 1.0 + (tau_7 * tau_7) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
    a1_5 = coef_weights_1 * ( 1.0 + (tau_5 * tau_5) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
    a2_5 = coef_weights_2 * ( 1.0 + (tau_5 * tau_5) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
    a3_5 = coef_weights_3 * ( 1.0 + (tau_5 * tau_5) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
    a5 = coef_weights_5 * ( 1.0 + (tau_5 * tau_5) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));

    Real one_a_sum_7 = 1.0 / (a1_7 + a2_7 + a3_7 + a7);
    Real one_a_sum_5 = 1.0 / (a1_5 + a2_5 + a3_5 + a5);

    w1_7 = a1_7 * one_a_sum_7;
    w2_7 = a2_7 * one_a_sum_7;
    w3_7 = a3_7 * one_a_sum_7;
    w7 = a7 * one_a_sum_7;

    w1_5 = a1_5 * one_a_sum_5;
    w2_5 = a2_5 * one_a_sum_5;
    w3_5 = a3_5 * one_a_sum_5;
    w5 = a5 * one_a_sum_5;

    // Compute coefficients of the Legendre basis polynomial of order 7
     u0_7 = v4;
     u1_7 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                + w1_7 * s11 + w2_7 * s21 + w3_7 * s31;
     u2_7 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                 + w1_7 * s12 + w2_7 * s22 + w3_7 * s32;
     u3_7 = (w7 / coef_weights_7) * s73;
     u4_7 = (w7 / coef_weights_7) * s74;
     u5_7 = (w7 / coef_weights_7) * s75;
     u6_7 = (w7 / coef_weights_7) * s76;

     // Compute coefficients of the Legendre basis polynomial of order 5
     u0_5 = v4;
     u1_5 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                + w1_5 * s11 + w2_5 * s21 + w3_5 * s31;
     u2_5 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                + w1_5 * s12 + w2_5 * s22 + w3_5 * s32;
     u3_5 = (w5 / coef_weights_5) * s53;
     u4_5 = (w5 / coef_weights_5) * s54;

    // Compute values of reconstructed Legendre basis polynomials
    Real polynomial_7 = u0_7 + u1_7 * 1.0/2.0 + u2_7 * 1.0/6.0 + u3_7 * 1.0/20.0 + u4_7 * 1.0/70.0 + u5_7 * 1.0/252.0 + u6_7 * 1.0/924.0;
    Real polynomial_5 = u0_5 + u1_5 * 1.0/2.0 + u2_5 * 1.0/6.0 + u3_5 * 1.0/20.0 + u4_5 * 1.0/70.0;

    // Compute normalized weights for hybridization Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real sigma = std::abs( s7 - s5 );
    Real b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
    Real b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

    Real one_b_sum = b7 + b5;

    Real w_ao_7 = b7 / one_b_sum;
    Real w_ao_5 = b5 / one_b_sum;

    // Return value of hybridized reconstructed polynomial
    return  (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}

Real WENOAO753_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6, v7;
    Real u0_5, u1_5, u2_5, u3_5, u4_5;
    Real u0_7, u1_7, u2_7, u3_7, u4_7, u5_7, u6_7;
    Real a1_5, a2_5, a3_5, a1_7, a2_7, a3_7, a5, a7;
    Real w1_5, w2_5, w3_5, w1_7, w2_7, w3_7, w5, w7;

	//assign value to v1, v2,...
	k = 1;
    v1 = *(f + k + 3);
    v2 = *(f + k + 2);
    v3 = *(f + k + 1);
    v4 = *(f + k); 
    v5 = *(f + k - 1);
    v6 = *(f + k - 2);
    v7 = *(f + k - 3);

	//smoothness indicator
    Real s11 = 1.0/2.0*v2 - 2.0*v3 + 3.0/2.0*v4, s12 = 1.0/2.0*v2 - 1.0*v3 + 1.0/2.0*v4;
    Real s21 = -1.0/2.0*v3 + 1.0/2.0*v5, s22 = 1.0/2.0*v3 - v4 + 1.0/2.0*v5;
    Real s31 = -3.0/2.0*v4 + 2.0*v5 -1.0/2.0*v6, s32 = 1.0/2.0*v4 - v5 + 1.0/2.0*v6;
    Real s51 =  11.0/120.0 * v2 - 82.0/120.0 * v3 + 82.0/120.0 * v5 - 11.0/120.0 * v6;
    Real s52 = -3.0/56.0 * v2 + 40.0/56.0 * v3 + -74.0/56.0 * v4 + 40.0/56.0 * v5  -3.0/56.0 * v6;
    Real s53 = -1.0/12.0 * v2 + 2.0/12.0 * v3 - 2.0/12.0 * v5 + 1.0/12.0 * v6;
    Real s54 = 1.0/24.0 * v2 - 4.0/24.0 * v3 + 6.0/24.0 * v4  - 4.0/24.0 * v5 + 1.0/24.0 * v6;
	Real s1 = 1.0*s11*s11 + 13.0/3.0*s12*s12;
	Real s2 = 1.0*s21*s21 + 13.0/3.0*s22*s22;
	Real s3 = 1.0*s31*s31 + 13.0/3.0*s32*s32;
    Real s5 = 1.0 * (s51 + 1.0/10.0 * s53) * (s51 + 1.0/10.0 * s53)
                            + 13.0/3.0 * (s52 + 123.0/455.0 * s54) * (s52 + 123.0/455.0 * s54)
                            + 781.0/20.0 * s53 * s53 + 1421461.0/2275.0 * s54 *s54;

    Real s71 = -191.0/10080.0 * v1 + 1688.0/10080.0 * v2 - 7843.0/10080.0 * v3 + 7843.0/10080.0 * v5 - 1688.0/10080.0 * v6 + 191.0/10080.0 * v7;
    Real s72 =  79.0/10080.0 * v1 - 1014.0/10080.0 * v2 + 8385.0/10080.0 * v3 - 14900.0/10080.0 * v4 + 8385.0/10080.0 * v5 - 1014.0/10080.0 * v6 + 79.0/10080.0 * v7;
    Real s73 = 5.0/216.0 * v1 - 38.0/216.0 * v2 + 61.0/216.0 * v3 - 61.0/216.0 * v5 + 38.0/216.0 * v6 - 5.0/216.0 * v7;
    Real s74 = -13.0/1584.0 * v1 + 144.0/1584.0 * v2 + 459.0/1584.0 * v3 + 656.0/1584.0 * v4 - 459.0/1584.0 * v5 + 144.0/1584.0 * v6 - 13.0/1584.0 * v7;
    Real s75 = -1.0/240.0 * v1 + 4.0/240.0 * v2 - 5.0/240.0 * v3 + 5.0/240.0 * v5 - 4.0/240.0 * v6 + 1/240.0 * v7;
    Real s76 = 1.0/720.0 * v1 - 6.0/720.0 * v2 + 15.0/720.0 * v3 - 20.0/720.0 * v4 + 15.0/720.0 * v5 - 6.0/720.0 * v6 + 1/720.0 * v7;
    Real s7 = 1.0 * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75) * (s71 + 1.0/10.0 * s73 + 1.0/126.0 * s75)
                        + 13.0/3.0 * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76) * (s72 + 123.0/455.0 * s74 + 85.0/2002.0 * s76)
                        + 781.0/20.0 * (s73 + 26045.0/49203.0 * s75) * (s73 + 26045.0/49203.0 * s75)
                        + 1421461.0/2275.0 * (s74 + 81596225.0/93816426.0 * s76) * (s74 + 81596225.0/93816426.0 * s76)
                        + 21520059541.0/1377684.0 * s75 * s75 + 15510384942580921.0/27582029244.0 * s76 * s76;

    // Compute normalized weights Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real tau_7 = (std::abs( s7 - s1 ) + std::abs( s7 - s2 ) + std::abs( s7 - s3 )) / 3.0;
    Real tau_5 = (std::abs( s5 - s1 ) + std::abs( s5 - s2 ) + std::abs( s5 - s3 )) / 3.0;
    Real coef_weights_1 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_2 = (1 - 0.85) * 0.85;
    Real coef_weights_3 = (1 - 0.85) * (1 - 0.85) / 2.0;
    Real coef_weights_5 = 0.85;
    Real coef_weights_7 = 0.85;
    
    a1_7 = coef_weights_1 * ( 1.0 + (tau_7 * tau_7) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
    a2_7 = coef_weights_2 * ( 1.0 + (tau_7 * tau_7) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
    a3_7 = coef_weights_3 * ( 1.0 + (tau_7 * tau_7) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
    a7 = coef_weights_7 * ( 1.0 + (tau_7 * tau_7) / ((s7 + 2.0e-16) * (s7 + 2.0e-16)));
    a1_5 = coef_weights_1 * ( 1.0 + (tau_5 * tau_5) / ((s1 + 2.0e-16) * (s1 + 2.0e-16)));
    a2_5 = coef_weights_2 * ( 1.0 + (tau_5 * tau_5) / ((s2 + 2.0e-16) * (s2 + 2.0e-16)));
    a3_5 = coef_weights_3 * ( 1.0 + (tau_5 * tau_5) / ((s3 + 2.0e-16) * (s3 + 2.0e-16)));
    a5 = coef_weights_5 * ( 1.0 + (tau_5 * tau_5) / ((s5 + 2.0e-16) * (s5 + 2.0e-16)));

    Real one_a_sum_7 = 1.0 / (a1_7 + a2_7 + a3_7 + a7);
    Real one_a_sum_5 = 1.0 / (a1_5 + a2_5 + a3_5 + a5);

    w1_7 = a1_7 * one_a_sum_7;
    w2_7 = a2_7 * one_a_sum_7;
    w3_7 = a3_7 * one_a_sum_7;
    w7 = a7 * one_a_sum_7;

    w1_5 = a1_5 * one_a_sum_5;
    w2_5 = a2_5 * one_a_sum_5;
    w3_5 = a3_5 * one_a_sum_5;
    w5 = a5 * one_a_sum_5;

    // Compute coefficients of the Legendre basis polynomial of order 7
     u0_7 = v4;
     u1_7 = (w7 / coef_weights_7) * (s71 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                + w1_7 * s11 + w2_7 * s21 + w3_7 * s31;
     u2_7 = (w7 / coef_weights_7) * (s72 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                 + w1_7 * s12 + w2_7 * s22 + w3_7 * s32;
     u3_7 = (w7 / coef_weights_7) * s73;
     u4_7 = (w7 / coef_weights_7) * s74;
     u5_7 = (w7 / coef_weights_7) * s75;
     u6_7 = (w7 / coef_weights_7) * s76;

     // Compute coefficients of the Legendre basis polynomial of order 5
     u0_5 = v4;
     u1_5 = (w5 / coef_weights_5) * (s51 - coef_weights_1 * s11 - coef_weights_2 * s21 - coef_weights_3 * s31)
                + w1_5 * s11 + w2_5 * s21 + w3_5 * s31;
     u2_5 = (w5 / coef_weights_5) * (s52 - coef_weights_1 * s12 - coef_weights_2 * s22 - coef_weights_3 * s32)
                + w1_5 * s12 + w2_5 * s22 + w3_5 * s32;
     u3_5 = (w5 / coef_weights_5) * s53;
     u4_5 = (w5 / coef_weights_5) * s54;

    // Compute values of reconstructed Legendre basis polynomials
    Real polynomial_7 = u0_7 + u1_7 * 1.0/2.0 + u2_7 * 1.0/6.0 + u3_7 * 1.0/20.0 + u4_7 * 1.0/70.0 + u5_7 * 1.0/252.0 + u6_7 * 1.0/924.0;
    Real polynomial_5 = u0_5 + u1_5 * 1.0/2.0 + u2_5 * 1.0/6.0 + u3_5 * 1.0/20.0 + u4_5 * 1.0/70.0;

    // Compute normalized weights for hybridization Note: Borges et al. suggest an epsilon value of 1e-40 to minimize the influence. We use machine precision instead.
    Real sigma = std::abs( s7 - s5 );
    Real b7 = 2.0e-16 * (1 + sigma / (s7 + 2.0e-16));
    Real b5 = (1 - 2.0e-16) * (1 + sigma / (s5 + 2.0e-16));

    Real one_b_sum = b7 + b5;

    Real w_ao_7 = b7 / one_b_sum;
    Real w_ao_5 = b5 / one_b_sum;

    // Return value of hybridized reconstructed polynomial
    return  (w_ao_7 / 2.0e-16) * (polynomial_7 - (1 - 2.0e-16) * polynomial_5) + w_ao_5 * polynomial_5;
}

Real Weno5L2_P(Real *f, Real delta, Real lambda)
{
	//assign value to v1, v2,...
    int k = 0;
    Real v1 = *(f + k - 2);
    Real v2 = *(f + k - 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k + 1); 
    Real v5 = *(f + k + 2);
    Real v6 = *(f + k + 3);

    //smoothness indicator
    Real epsilon = 1.0e-20;
    Real s0 = std::pow((v4-v3), 2.0);
    Real s1 = std::pow((v3-v2), 2.0); 
    Real s2 = (13.0*std::pow(v3-2.0*v4+v5, 2.0) + 3.0*std::pow(3.0*v3-4.0*v4+v5, 2.0))/12.0;
    Real s3 = (13.0*std::pow(v1-2.0*v2+v3, 2.0) + 3.0*std::pow(v1-4.0*v2+3.0*v3, 2.0))/12.0;
    Real t5 = (13.0*std::pow(v5-4.0*v4+6.0*v3-4.0*v2+v1, 2.0) +3.0*std::pow(v5-2.0*v4+2.0*v2-v1, 2.0))/12.0;

    Real e0 = (v4*v4-4.0*v3*v4+2.0*v2*v4+4.0*v3*v3-4.0*v2*v3+v2*v2)/45.0;
    Real e1 = e0;
    Real e2 = 0.0;
    Real e3 = 0.0;

    Real a0 = 0.4*(1.0 + lambda*t5/(lambda*s0 + e0 + epsilon));
    Real a1 = 0.2*(1.0 + lambda*t5/(lambda*s1 + e1 + epsilon));
    Real a2 = 0.3*(1.0 + lambda*t5/(lambda*s2 + e2 + epsilon));
    Real a3 = 0.1*(1.0 + lambda*t5/(lambda*s3 + e3 + epsilon));

    // Real a0 = 0.4*(1.0 + t5/(s0 + epsilon));
    // Real a1 = 0.2*(1.0 + t5/(s1 + epsilon));
    // Real a2 = 0.3*(1.0 + t5/(s2 + epsilon));
    // Real a3 = 0.1*(1.0 + t5/(s3 + epsilon));  
    
    Real tw1= 1.0 / (a0 + a1 + a2 + a3);
    Real w0 = a0*tw1;
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;

    //return weighted average
    return  (w3*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(2.0*v3 + 5.0*v4 - v5)
          + w1*(-3.0*v2 + 9.0*v3)
          + w0*(3.0*v3 + 3.0*v4))/6.0;
}

Real Weno5L2_M(Real *f, Real delta, Real lambda)
{
	//assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 2);
    Real v2 = *(f + k + 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k - 1); 
    Real v5 = *(f + k - 2);
    Real v6 = *(f + k - 3);

    //smoothness indicator
    Real epsilon = 1.0e-20;
    Real s0 = std::pow((v4-v3), 2.0);
    Real s1 = std::pow((v3-v2), 2.0); 
    Real s2 = (13.0*std::pow(v3-2.0*v4+v5, 2.0) + 3.0*std::pow(3.0*v3-4.0*v4+v5, 2.0))/12.0;
    Real s3 = (13.0*std::pow(v1-2.0*v2+v3, 2.0) + 3.0*std::pow(v1-4.0*v2+3.0*v3, 2.0))/12.0;
    Real t5 = (13.0*std::pow(v5-4.0*v4+6.0*v3-4.0*v2+v1, 2.0) +3.0*std::pow(v5-2.0*v4+2.0*v2-v1, 2.0))/12.0;

    Real e0 = (v4*v4-4.0*v3*v4+2.0*v2*v4+4.0*v3*v3-4.0*v2*v3+v2*v2)/45.0;
    Real e1 = e0;
    Real e2 = 0.0;
    Real e3 = 0.0;

    Real a0 = 0.4*(1.0 + lambda*t5/(lambda*s0 + e0 + epsilon));
    Real a1 = 0.2*(1.0 + lambda*t5/(lambda*s1 + e1 + epsilon));
    Real a2 = 0.3*(1.0 + lambda*t5/(lambda*s2 + e2 + epsilon));
    Real a3 = 0.1*(1.0 + lambda*t5/(lambda*s3 + e3 + epsilon));
    
    Real tw1= 1.0 / (a0 + a1 + a2 + a3);
    Real w0 = a0*tw1;
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;

    //return weighted average
    return  (w3*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(2.0*v3 + 5.0*v4 - v5)
          + w1*(-3.0*v2 + 9.0*v3)
          + w0*(3.0*v3 + 3.0*v4))/6.0;
}

Real WENOCU6_P(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 0;
    Real v1 = *(f + k - 2);
    Real v2 = *(f + k - 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k + 1); 
    Real v5 = *(f + k + 2);
    Real v6 = *(f + k + 3);

    //smoothness indicator
    Real epsilon = 1.e-8*delta*delta;
    Real s11 = v1 - 2.0*v2 + v3;
    Real s12 = v1 - 4.0*v2 + 3.0*v3;
    Real s1 = 13.0*s11*s11 + 3.0*s12*s12;
    Real s21 = v2 - 2.0*v3 + v4;
    Real s22 = v2 - v4;
    Real s2 = 13.0*s21*s21 + 3.0*s22*s22;
    Real s31 = v3 - 2.0*v4 + v5;
    Real s32 = 3.0*v3 - 4.0*v4 + v5;
    Real s3 = 13.0*s31*s31 + 3.0*s32*s32;
    Real tau61 = (259.0*v6 - 1895.0*v5 + 6670.0*v4 - 2590.0*v3 - 2785.0*v2 + 341.0*v1)/5760.0;
    Real tau62 = - (v5 - 12.0*v4 + 22.0*v3 - 12.0*v2 + v1)/16.0;
    Real tau63 = - (7.0*v6 - 47.0*v5 + 94.0*v4 - 70.0*v3 + 11.0*v2 + 5.0*v1)/144.0;
    Real tau64 = (v5 - 4.0*v4 + 6.0*v3 - 4.0*v2 + v1)/24.0;
    Real tau65 = - (- v6 + 5.0*v5 - 10.0*v4 + 10.0*v3 - 5.0*v2 + v1)/120.0;
    Real a1a1 = 1.0, a2a2 = 13.0/3.0, a1a3 = 0.5, a3a3 = 3129.0/80.0, a2a4 = 21.0/5.0;
    Real a1a5 = 1.0/8.0, a4a4 = 87617.0/140.0, a3a5 = 14127.0/224.0, a5a5 = 252337135.0/16128.0;
    Real s6 = (tau61*tau61*a1a1 + tau62*tau62*a2a2 + tau61*tau63*a1a3 + tau63*tau63*a3a3 + tau62*tau64*a2a4
                  + tau61*tau65*a1a5 + tau64*tau64*a4a4 + tau63*tau65*a3a5 + tau65*tau65*a5a5)*12.0;

    //weights
    Real s55 = (s1 + s3 + 4.0*s2)/6.0;
    Real s5 = fabs(s6 - s55);
    Real r0 = 20.0;
    Real r1 = r0 + s5/(s1 + epsilon); 
    Real r2 = r0 + s5/(s2 + epsilon);
    Real r3 = r0 + s5/(s3 + epsilon);
    Real r4 = r0 + s5/(s6 + epsilon);
    Real a1 = 0.05*r1;
    Real a2 = 0.45*r2;
    Real a3 = 0.45*r3;
    Real a4 = 0.05*r4;
    Real tw1= 1.0 / (a1 + a2 + a3 + a4);
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;
    Real w4 = a4*tw1;

    //return weighted average
    return  (w1*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(-v2 + 5.0*v3 + 2.0*v4) + w3*(2.0*v3 + 5.0*v4 - v5)
          + w4*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}
//this is WENOCU6
Real WENOCU6_M(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 2);
    Real v2 = *(f + k + 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k - 1); 
    Real v5 = *(f + k - 2);
    Real v6 = *(f + k - 3);

    Real epsilon = 1.e-8*delta*delta;
    Real s11 = v1 - 2.0*v2 + v3;
    Real s12 = v1 - 4.0*v2 + 3.0*v3;
    Real s1 = 13.0*s11*s11 + 3.0*s12*s12;
    Real s21 = v2 - 2.0*v3 + v4;
    Real s22 = v2 - v4;
    Real s2 = 13.0*s21*s21 + 3.0*s22*s22;
    Real s31 = v3 - 2.0*v4 + v5;
    Real s32 = 3.0*v3 - 4.0*v4 + v5;
    Real s3 = 13.0*s31*s31 + 3.0*s32*s32;
    Real tau61 = (259.0*v6 - 1895.0*v5 + 6670.0*v4 - 2590.0*v3 - 2785.0*v2 + 341.0*v1)/5760.0;
    Real tau62 = - (v5 - 12.0*v4 + 22.0*v3 - 12.0*v2 + v1)/16.0;
    Real tau63 = - (7.0*v6 - 47.0*v5 + 94.0*v4 - 70.0*v3 + 11.0*v2 + 5.0*v1)/144.0;
    Real tau64 = (v5 - 4.0*v4 + 6.0*v3 - 4.0*v2 + v1)/24.0;
    Real tau65 = - (- v6 + 5.0*v5 - 10.0*v4 + 10.0*v3 - 5.0*v2 + v1)/120.0;
    Real a1a1 = 1.0, a2a2 = 13.0/3.0, a1a3 = 0.5, a3a3 = 3129.0/80.0, a2a4 = 21.0/5.0;
    Real a1a5 = 1.0/8.0, a4a4 = 87617.0/140.0, a3a5 = 14127.0/224.0, a5a5 = 252337135.0/16128.0;
    Real s6 = (tau61*tau61*a1a1 + tau62*tau62*a2a2 + tau61*tau63*a1a3 + tau63*tau63*a3a3 + tau62*tau64*a2a4
                  + tau61*tau65*a1a5 + tau64*tau64*a4a4 + tau63*tau65*a3a5 + tau65*tau65*a5a5)*12.0;

    //weights
    Real s55 = (s1 + s3 + 4.0*s2)/6.0;
    Real s5 = fabs(s6 - s55);
    Real r0 = 20.0;
    Real r1 = r0 + s5/(s1 + epsilon); 
    Real r2 = r0 + s5/(s2 + epsilon);
    Real r3 = r0 + s5/(s3 + epsilon);
    Real r4 = r0 + s5/(s6 + epsilon);
    Real a1 = 0.05*r1;
    Real a2 = 0.45*r2;
    Real a3 = 0.45*r3;
    Real a4 = 0.05*r4;
    Real tw1= 1.0 / (a1 + a2 + a3 + a4);
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;
    Real w4 = a4*tw1;

    //return weighted average
    return  (w1*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(-v2 + 5.0*v3 + 2.0*v4) + w3*(2.0*v3 + 5.0*v4 - v5)
          + w4*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}

Real WENOCU6M1_P(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 0;
    Real v1 = *(f + k - 2);
    Real v2 = *(f + k - 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k + 1); 
    Real v5 = *(f + k + 2);
    Real v6 = *(f + k + 3);

    //smoothness indicator
    Real epsilon = 1.e-8*delta*delta;
    Real s11 = v1 - 2.0*v2 + v3;
    Real s12 = v1 - 4.0*v2 + 3.0*v3;
    Real s1 = 13.0*s11*s11 + 3.0*s12*s12;
    Real s21 = v2 - 2.0*v3 + v4;
    Real s22 = v2 - v4;
    Real s2 = 13.0*s21*s21 + 3.0*s22*s22;
    Real s31 = v3 - 2.0*v4 + v5;
    Real s32 = 3.0*v3 - 4.0*v4 + v5;
    Real s3 = 13.0*s31*s31 + 3.0*s32*s32;
    Real tau61 = (259.0*v6 - 1895.0*v5 + 6670.0*v4 - 2590.0*v3 - 2785.0*v2 + 341.0*v1)/5760.0;
    Real tau62 = - (v5 - 12.0*v4 + 22.0*v3 - 12.0*v2 + v1)/16.0;
    Real tau63 = - (7.0*v6 - 47.0*v5 + 94.0*v4 - 70.0*v3 + 11.0*v2 + 5.0*v1)/144.0;
    Real tau64 = (v5 - 4.0*v4 + 6.0*v3 - 4.0*v2 + v1)/24.0;
    Real tau65 = - (- v6 + 5.0*v5 - 10.0*v4 + 10.0*v3 - 5.0*v2 + v1)/120.0;
    Real a1a1 = 1.0, a2a2 = 13.0/3.0, a1a3 = 0.5, a3a3 = 3129.0/80.0, a2a4 = 21.0/5.0;
    Real a1a5 = 1.0/8.0, a4a4 = 87617.0/140.0, a3a5 = 14127.0/224.0, a5a5 = 252337135.0/16128.0;
    Real s6 = (tau61*tau61*a1a1 + tau62*tau62*a2a2 + tau61*tau63*a1a3 + tau63*tau63*a3a3 + tau62*tau64*a2a4
                  + tau61*tau65*a1a5 + tau64*tau64*a4a4 + tau63*tau65*a3a5 + tau65*tau65*a5a5)*12.0;

    //weights
    Real s55 = (s1 + s3 + 4.0*s2)/6.0;
    Real s5 = fabs(s6 - s55);
    Real r0 = 1.0e3;
    Real r1 = r0 + s5/(s1 + epsilon); 
    Real r2 = r0 + s5/(s2 + epsilon);
    Real r3 = r0 + s5/(s3 + epsilon);
    Real r4 = r0 + s5/(s6 + epsilon);
    Real a1 = 0.05*r1*r1*r1*r1;
    Real a2 = 0.45*r2*r2*r2*r2;
    Real a3 = 0.45*r3*r3*r3*r3;
    Real a4 = 0.05*r4*r4*r4*r4;
    Real tw1= 1.0 / (a1 + a2 + a3 + a4);
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;
    Real w4 = a4*tw1;

    //return weighted average
    return  (w1*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(-v2 + 5.0*v3 + 2.0*v4) + w3*(2.0*v3 + 5.0*v4 - v5)
          + w4*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}
//this is WENOCU6_M
Real WENOCU6M1_M(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 2);
    Real v2 = *(f + k + 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k - 1); 
    Real v5 = *(f + k - 2);
    Real v6 = *(f + k - 3);

    Real epsilon = 1.e-8*delta*delta;
    Real s11 = v1 - 2.0*v2 + v3;
    Real s12 = v1 - 4.0*v2 + 3.0*v3;
    Real s1 = 13.0*s11*s11 + 3.0*s12*s12;
    Real s21 = v2 - 2.0*v3 + v4;
    Real s22 = v2 - v4;
    Real s2 = 13.0*s21*s21 + 3.0*s22*s22;
    Real s31 = v3 - 2.0*v4 + v5;
    Real s32 = 3.0*v3 - 4.0*v4 + v5;
    Real s3 = 13.0*s31*s31 + 3.0*s32*s32;
    Real tau61 = (259.0*v6 - 1895.0*v5 + 6670.0*v4 - 2590.0*v3 - 2785.0*v2 + 341.0*v1)/5760.0;
    Real tau62 = - (v5 - 12.0*v4 + 22.0*v3 - 12.0*v2 + v1)/16.0;
    Real tau63 = - (7.0*v6 - 47.0*v5 + 94.0*v4 - 70.0*v3 + 11.0*v2 + 5.0*v1)/144.0;
    Real tau64 = (v5 - 4.0*v4 + 6.0*v3 - 4.0*v2 + v1)/24.0;
    Real tau65 = - (- v6 + 5.0*v5 - 10.0*v4 + 10.0*v3 - 5.0*v2 + v1)/120.0;
    Real a1a1 = 1.0, a2a2 = 13.0/3.0, a1a3 = 0.5, a3a3 = 3129.0/80.0, a2a4 = 21.0/5.0;
    Real a1a5 = 1.0/8.0, a4a4 = 87617.0/140.0, a3a5 = 14127.0/224.0, a5a5 = 252337135.0/16128.0;
    Real s6 = (tau61*tau61*a1a1 + tau62*tau62*a2a2 + tau61*tau63*a1a3 + tau63*tau63*a3a3 + tau62*tau64*a2a4
                  + tau61*tau65*a1a5 + tau64*tau64*a4a4 + tau63*tau65*a3a5 + tau65*tau65*a5a5)*12.0;

    //weights
    Real s55 = (s1 + s3 + 4.0*s2)/6.0;
    Real s5 = fabs(s6 - s55);
    Real r0 = 1.0e3;
    Real r1 = r0 + s5/(s1 + epsilon); 
    Real r2 = r0 + s5/(s2 + epsilon);
    Real r3 = r0 + s5/(s3 + epsilon);
    Real r4 = r0 + s5/(s6 + epsilon);
    Real a1 = 0.05*r1*r1*r1*r1;
    Real a2 = 0.45*r2*r2*r2*r2;
    Real a3 = 0.45*r3*r3*r3*r3;
    Real a4 = 0.05*r4*r4*r4*r4;
    Real tw1= 1.0 / (a1 + a2 + a3 + a4);
    Real w1 = a1*tw1;
    Real w2 = a2*tw1;
    Real w3 = a3*tw1;
    Real w4 = a4*tw1;

    //return weighted average
    return  (w1*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w2*(-v2 + 5.0*v3 + 2.0*v4) + w3*(2.0*v3 + 5.0*v4 - v5)
          + w4*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}

Real TENO5_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real b1, b2, b3;
	Real a1, a2, a3, w1, w2, w3;
    Real Variation1,Variation2,Variation3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	Real s1 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) + 3.0/12.0*(v2 - v4)*(v2 - v4); 
	Real s2 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) + 3.0/12.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	Real s3 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) + 3.0/12.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);

    Real tau5 = std::abs(s3 - s2);

	a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40),6.0);
	a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40),6.0);
	a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40),6.0);

	b1 = a1/(a1 + a2 + a3);
	b2 = a2/(a1 + a2 + a3);
	b3 = a3/(a1 + a2 + a3);
    
	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0*v2 + 5.0*v3 + 2.0*v4;
	Variation2 = 2.0*v3 + 5.0*v4 - 1.0*v5;
	Variation3 = 2.0*v1 - 7.0*v2 + 11.0*v3;

	a1 = 0.600 * b1;
	a2 = 0.300 * b2;
	a3 = 0.100 * b3;

	w1 = a1/(a1 + a2 + a3);
	w2 = a2/(a1 + a2 + a3);
	w3 = a3/(a1 + a2 + a3);    

	return	1.0/6.0*(w1*Variation1  + w2*Variation2 + w3*Variation3);
}

Real TENO5_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5;
	Real b1, b2, b3;
	Real a1, a2, a3, w1, w2, w3;
    Real Variation1,Variation2,Variation3;

    //assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);

	Real s1 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) + 3.0/12.0*(v2 - v4)*(v2 - v4); 
	Real s2 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) + 3.0/12.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	Real s3 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) + 3.0/12.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);

    Real tau5 = std::abs(s3 - s2);

	a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40),6.0);
	a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40),6.0);
	a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40),6.0);

	b1 = a1/(a1 + a2 + a3);
	b2 = a2/(a1 + a2 + a3);
	b3 = a3/(a1 + a2 + a3);
    
	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0*v2 + 5.0*v3 + 2.0*v4;
	Variation2 = 2.0*v3 + 5.0*v4 - 1.0*v5;
	Variation3 = 2.0*v1 - 7.0*v2 + 11.0*v3;

	a1 = 0.600 * b1;
	a2 = 0.300 * b2;
	a3 = 0.100 * b3;

	w1 = a1/(a1 + a2 + a3);
	w2 = a2/(a1 + a2 + a3);
	w3 = a3/(a1 + a2 + a3);    

	return	1.0/6.0*(w1*Variation1  + w2*Variation2 + w3*Variation3);
}

Real TENO6_OPT_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6;
	Real b1, b2, b3, b4, b5;
	Real a1, a2, a3, a4, a5, w1, w2, w3, w4, w5;
    Real Variation1,Variation2,Variation3,Variation4,Variation5;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);
    v6 = *(f + k + 3);

	Real s1 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) + 3.0/12.0*(v2 - v4)*(v2 - v4); 
	Real s2 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) + 3.0/12.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	Real s3 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) + 3.0/12.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	Real s4 = 1./240.*std::fabs((2107.0*v3*v3 - 9402.0*v3*v4 + 11003.0*v4*v4 + 7042.0*v3*v5 - 17246.0*v4*v5 + 7043.0*v5*v5 - 1854.0*v3*v6 + 4642.0*v4*v6 - 3882.0*v5*v6 + 547.0*v6*v6));  
    
    Real s64 =  1.0/12.0*std::fabs(139633.0*v6*v6 - 1429976.0*v5*v6 + 3824847.0*v5*v5 + 2863984.0*v4*v6 - 15880404.0*v4*v5 + 17195652.0*v4*v4 - 2792660.0*v3*v6
              - 35817664.0*v3*v4 + 19510972.0*v3*v3 + 1325006.0*v2*v6 - 7727988.0*v2*v5 + 17905032.0*v2*v4 - 20427884.0*v2*v3 + 5653317.0*v2*v2
              - 245620.0*v1*v6 + 1458762.0*v1*v5 - 3462252.0*v1*v4 + 4086352.0*v1*v3 - 2380800.0*v1*v2 + 271779.0*v1*v1 + 15929912.0*v3*v5) / 10080.0;

	Real tau6 = std::abs(s64-(s3+s2+4.0*s1)/6.0);

	a1 = 1./4.*std::pow(1.0 + tau6/(s1+1.0e-40),6.0);
	a2 = 1./4.*std::pow(1.0 + tau6/(s2+1.0e-40),6.0);
	a3 = 1./4.*std::pow(1.0 + tau6/(s3+1.0e-40),6.0);
	a4 = 1./4.*std::pow(1.0 + tau6/(s4+1.0e-40),6.0);

	b1 = a1/(a1 + a2 + a3 + a4);
	b2 = a2/(a1 + a2 + a3 + a4);
	b3 = a3/(a1 + a2 + a3 + a4);
	b4 = a4/(a1 + a2 + a3 + a4);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;
	b4 = b4 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0/6.0*v2 + 5.0/6.0*v3 + 2.0/6.0*v4 - v3;
	Variation2 = 2./6.*v3 + 5./6.*v4 - 1./6.*v5 - v3;
	Variation3 = 2./6.*v1 - 7./6.*v2 + 11./6.*v3 - v3;
	Variation4 = 3./12.*v3 + 13./12.*v4 - 5./12.*v5 + 1./12.*v6 - v3;

	a1 = 0.462 * b1;
	a2 = 0.300 * b2;
	a3 = 0.054 * b3; 
	a4 = 0.184 * b4;

	w1 = a1/(a1 + a2 + a3 + a4);
	w2 = a2/(a1 + a2 + a3 + a4);
	w3 = a3/(a1 + a2 + a3 + a4);
	w4 = a4/(a1 + a2 + a3 + a4);
	 
	return	v3 + w1*Variation1
		  	   + w2*Variation2
			   + w3*Variation3
			   + w4*Variation4;
}
Real TENO6_OPT_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6;
	Real b1, b2, b3, b4;
	Real a1, a2, a3, a4, w1, w2, w3, w4;
    Real Variation1,Variation2,Variation3,Variation4;

	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);
	v6 = *(f + k - 3);

	Real s1 = 13.0/12.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) + 3.0/12.0*(v2 - v4)*(v2 - v4); 
	Real s2 = 13.0/12.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) + 3.0/12.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	Real s3 = 13.0/12.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) + 3.0/12.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	Real s4 = 1./240.*std::fabs((2107.0*v3*v3 - 9402.0*v3*v4 + 11003.0*v4*v4 + 7042.0*v3*v5 - 17246.0*v4*v5 + 7043.0*v5*v5 - 1854.0*v3*v6 + 4642.0*v4*v6 - 3882.0*v5*v6 + 547.0*v6*v6));  
    
    Real s64 =  1.0/12.0*std::fabs(139633.0*v6*v6 - 1429976.0*v5*v6 + 3824847.0*v5*v5 + 2863984.0*v4*v6 - 15880404.0*v4*v5 + 17195652.0*v4*v4 - 2792660.0*v3*v6
              - 35817664.0*v3*v4 + 19510972.0*v3*v3 + 1325006.0*v2*v6 - 7727988.0*v2*v5 + 17905032.0*v2*v4 - 20427884.0*v2*v3 + 5653317.0*v2*v2
              - 245620.0*v1*v6 + 1458762.0*v1*v5 - 3462252.0*v1*v4 + 4086352.0*v1*v3 - 2380800.0*v1*v2 + 271779.0*v1*v1 + 15929912.0*v3*v5) / 10080.0;

	Real tau6 = std::abs(s64-(s3+s2+4.0*s1)/6.0);

	a1 = 1./4.*std::pow(1.0 + tau6/(s1+1.0e-40),6.0);
	a2 = 1./4.*std::pow(1.0 + tau6/(s2+1.0e-40),6.0);
	a3 = 1./4.*std::pow(1.0 + tau6/(s3+1.0e-40),6.0);
	a4 = 1./4.*std::pow(1.0 + tau6/(s4+1.0e-40),6.0);

	b1 = a1/(a1 + a2 + a3 + a4);
	b2 = a2/(a1 + a2 + a3 + a4);
	b3 = a3/(a1 + a2 + a3 + a4);
	b4 = a4/(a1 + a2 + a3 + a4);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;
	b4 = b4 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0/6.0*v2 + 5.0/6.0*v3 + 2.0/6.0*v4 - v3;
	Variation2 = 2./6.*v3 + 5./6.*v4 - 1./6.*v5 - v3;
	Variation3 = 2./6.*v1 - 7./6.*v2 + 11./6.*v3 - v3;
	Variation4 = 3./12.*v3 + 13./12.*v4 - 5./12.*v5 + 1./12.*v6 - v3;

	a1 = 0.462 * b1;
	a2 = 0.300 * b2;
	a3 = 0.054 * b3; 
	a4 = 0.184 * b4;

	w1 = a1/(a1 + a2 + a3 + a4);
	w2 = a2/(a1 + a2 + a3 + a4);
	w3 = a3/(a1 + a2 + a3 + a4);
	w4 = a4/(a1 + a2 + a3 + a4);
	 
	return	v3 + w1*Variation1
		  	   + w2*Variation2
			   + w3*Variation3
			   + w4*Variation4;
}

Real weno5Z_P(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6;
	Real s1, s2, s3;
	Real a1, a2, a3, w1, w2, w3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);
	v6 = *(f + k + 3);

	//smoothness indicator
	Real epsilon = 1.0e-6;
	s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
	   + 3.0*(v2 - v4)*(v2 - v4);
	s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

	//weights
	Real s5 = fabs(s1 - s3);
	a1 = 0.1*(1.0 + s5/(s1+epsilon));
	a2 = 0.6*(1.0 + s5/(s2+epsilon));
	a3 = 0.3*(1.0 + s5/(s3+epsilon));

	w1 = a1/(a1 + a2 + a3);
	w2 = a2/(a1 + a2 + a3);
	w3 = a3/(a1 + a2 + a3);

	//return weighted average
	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
}
Real weno5Z_M(Real *f, Real delta)
{
	int k;
	Real v1, v2, v3, v4, v5, v6;
	Real s1, s2,s3;
	Real a1, a2,a3, w1, w2, w3;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);
	v6 = *(f + k - 3);

	//smoothness indicator
	Real epsilon = 1.0e-6;
	s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
	s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
	   + 3.0*(v2 - v4)*(v2 - v4);
	s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

	//weights
	Real s5 = fabs(s1 - s3);
	a1 = 0.1*(1.0 + s5/(s1+epsilon));
	a2 = 0.6*(1.0 + s5/(s2+epsilon));
	a3 = 0.3*(1.0 + s5/(s3+epsilon));

	w1 = a1/(a1 + a2 + a3);
	w2 = a2/(a1 + a2 + a3);
	w3 = a3/(a1 + a2 + a3);

	//return weighted average
	return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
		  + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
		  + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;
}

Real WENOCU6M2_P(Real *f, Real delta)
{
    Real epsilon = 1.0e-8;
    int k = 0;
    Real v1 = *(f + k - 2); // i-2
    Real v2 = *(f + k - 1); // i-1
    Real v3 = *(f + k);	// i
    Real v4 = *(f + k + 1); // i+1
    Real v5 = *(f + k + 2);	// i+2
    Real v6 = *(f + k + 3);	// i+3

    Real epsdelta2=epsilon*delta*delta; // epsilon*delta^2

    //smoothness indicator
    Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
       + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3) + epsdelta2;
       // beta_1 + epsilon*delta^2
    Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
       + 3.0*(v2 - v4)*(v2 - v4) + epsdelta2;
       // beta_2 + epsilon*delta^2
    Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
       + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5) + epsdelta2;
       // beta_3 + epsilon*delta^2
    Real s64 =  fabs(139633.0*v6*v6 - 1429976.0*v5*v6 + 2863984.0*v4*v6 - 2792660.0*v3*v6
               + 1325006.0*v2*v6 - 245620.0*v1*v6 + 3824847.0*v5*v5  - 15880404.0*v4*v5 + 15929912.0*v3*v5
               - 7727988.0*v2*v5 + 1458762.0*v1*v5 + 17195652.0*v4*v4  - 35817664.0*v3*v4
               + 17905032.0*v2*v4 - 3462252.0*v1*v4 + 19510972.0*v3*v3  - 20427884.0*v2*v3
               + 4086352.0*v1*v3 + 5653317.0*v2*v2  - 2380800.0*v1*v2 + 271779.0*v1*v1) / 10080.0 + epsdelta2;

    //weights
    Real beta_ave = (s1 + s3 + 4.0*s2 - 6.0*epsdelta2)/6.0;
    Real tau_6 = s64- beta_ave -epsdelta2;
    Real chidelta2 = 1.0/epsilon*delta*delta;

// 	Real s5 = fabs(s64 - s56) + epsilon; // tau_6 + epsilon
    Real c_q = 1000.0; // C on page 7242
// 	Real q = 4.0;
    Real a0 = 0.05*pow((c_q + tau_6/s1 * (beta_ave+chidelta2)/(s1-epsdelta2+chidelta2)),4.0); // alpha_0
    Real a1 = 0.45*pow((c_q + tau_6/s2* (beta_ave+chidelta2)/(s2-epsdelta2+chidelta2)),4.0); // alpha_1
    Real a2 = 0.45*pow((c_q + tau_6/s3* (beta_ave+chidelta2)/(s3-epsdelta2+chidelta2)),4.0); // alpha_2
    Real a3 = 0.05*pow((c_q + tau_6/s64* (beta_ave+chidelta2)/(s64-epsdelta2+chidelta2)),4.0); //alpha_3
    Real tw1= 1.0 / (a0 + a1 + a2 + a3); // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
    Real w0 = a0*tw1;// omega_0
    Real w1 = a1*tw1;// omega_1
    Real w2 = a2*tw1;//omega_2
    Real w3 = a3*tw1;//omega_3
    //return weighted average
    return  (w0*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w1*(-v2 + 5.0*v3 + 2.0*v4) + w2*(2.0*v3 + 5.0*v4 - v5)
          + w3*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}
Real WENOCU6M2_M(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 2);
    Real v2 = *(f + k + 1);
    Real v3 = *(f + k);
    Real v4 = *(f + k - 1);
    Real v5 = *(f + k - 2);
    Real v6 = *(f + k - 3);

Real epsilon = 1.0e-8;

    Real epsdelta2=epsilon*delta*delta; // epsilon*delta^2

    //smoothness indicator

    // BIG QUESTION: there is always a " + 3.0"
    // beta_0 + epsilon*delta^2
    Real s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3)
       + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3) + epsdelta2;
       // beta_1 + epsilon*delta^2
    Real s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4)
       + 3.0*(v2 - v4)*(v2 - v4) + epsdelta2;
       // beta_2 + epsilon*delta^2
    Real s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5)
       + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5) + epsdelta2;
       // beta_3 + epsilon*delta^2
    Real s64 =  fabs(139633.0*v6*v6 - 1429976.0*v5*v6 + 2863984.0*v4*v6 - 2792660.0*v3*v6
               + 1325006.0*v2*v6 - 245620.0*v1*v6 + 3824847.0*v5*v5  - 15880404.0*v4*v5 + 15929912.0*v3*v5
               - 7727988.0*v2*v5 + 1458762.0*v1*v5 + 17195652.0*v4*v4  - 35817664.0*v3*v4
               + 17905032.0*v2*v4 - 3462252.0*v1*v4 + 19510972.0*v3*v3  - 20427884.0*v2*v3
               + 4086352.0*v1*v3 + 5653317.0*v2*v2  - 2380800.0*v1*v2 + 271779.0*v1*v1) / 10080.0 + epsdelta2;

    //weights

    Real beta_ave = (s1 + s3 + 4.0*s2 - 6.0*epsdelta2)/6.0;
    Real tau_6 = s64- beta_ave -epsdelta2;
    Real chidelta2 = 1.0/epsilon*delta*delta;


// 	Real s5 = fabs(s64 - s56) + epsilon; // tau_6 + epsilon
    Real c_q = 1000.0; // C on page 7242
// 	Real q = 4.0;
    Real a0 = 0.05*pow((c_q + tau_6/s1 * (beta_ave+chidelta2)/(s1-epsdelta2+chidelta2)),4.0); // alpha_0
    Real a1 = 0.45*pow((c_q + tau_6/s2 * (beta_ave+chidelta2)/(s2-epsdelta2+chidelta2)),4.0); // alpha_1
    Real a2 = 0.45*pow((c_q + tau_6/s3 * (beta_ave+chidelta2)/(s3-epsdelta2+chidelta2)),4.0); // alpha_2
    Real a3 = 0.05*pow((c_q + tau_6/s64 * (beta_ave+chidelta2)/(s64-epsdelta2+chidelta2)),4.0); //alpha_3
    Real tw1= 1.0 / (a0 + a1 + a2 + a3); // 1/ (alpha_0 +alpha_1 + alpha_2 +alpha_3)
    Real w0 = a0*tw1;// omega_0
    Real w1 = a1*tw1;// omega_1
    Real w2 = a2*tw1;//omega_2
    Real w3 = a3*tw1;//omega_3
    //return weighted average
    return  (w0*(2.0*v1 - 7.0*v2 + 11.0*v3)
          + w1*(-v2 + 5.0*v3 + 2.0*v4) + w2*(2.0*v3 + 5.0*v4 - v5)
          + w3*(11.0*v4 - 7.0*v5 + 2.0*v6))/6.0;
}
//-----------------------------------------------------------------------------------------
//		the 7th WENO Scheme
//-----------------------------------------------------------------------------------------
Real weno7_P(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 0;
    Real v1 = *(f + k - 3);
    Real v2 = *(f + k - 2);
    Real v3 = *(f + k - 1);
    Real v4 = *(f + k);
    Real v5 = *(f + k + 1);
    Real v6 = *(f + k + 2);
    Real v7 = *(f + k + 3);

    Real ep = 1.0e-7;
    Real C0 = 1.0 / 35.0;
    Real C1 = 12.0 / 35.0;
    Real C2 = 18.0 / 35.0;
    Real C3 = 4.0 / 35.0;
    //1  阶导数
    Real S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
    Real S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
    Real S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
    Real S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
    // 2 阶导数
    Real S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
    Real S21 = v3 - 2.0 * v4 + v5;
    Real S22 = v4 - 2.0 * v5 + v6;
    Real S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
    // 3 阶导数
    Real S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
    Real S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
    Real S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
    Real S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
    // 光滑度量因子
    Real S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
    Real S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
    Real S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
    Real S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
    // // 新的光滑度量因子, Xinliang Li
    // Real S0 = S10 * S10 + S20 * S20;
    // Real S1 = S11 * S11 + S21 * S21;
    // Real S2 = S12 * S12 + S22 * S22;
    // Real S3 = S13 * S13 + S23 * S23;
    // Alpha weights
    Real a0 = C0 / ((ep + S0) * (ep + S0));
    Real a1 = C1 / ((ep + S1) * (ep + S1));
    Real a2 = C2 / ((ep + S2) * (ep + S2));
    Real a3 = C3 / ((ep + S3) * (ep + S3));
    // Non-linear weigths
    Real W0 = a0 / (a0 + a1 + a2 + a3);
    Real W1 = a1 / (a0 + a1 + a2 + a3);
    Real W2 = a2 / (a0 + a1 + a2 + a3);
    Real W3 = a3 / (a0 + a1 + a2 + a3);
    // 4阶差分格式的通量
    Real q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
    Real q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
    Real q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
    Real q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
    // 由4个4阶差分格式组合成1个7阶差分格式
    return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}
Real weno7_M(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 3);
    Real v2 = *(f + k + 2);
    Real v3 = *(f + k + 1);
    Real v4 = *(f + k);
    Real v5 = *(f + k - 1);
    Real v6 = *(f + k - 2);
    Real v7 = *(f + k - 3);

    Real ep = 1.0e-7;
    Real C0 = 1.0 / 35.0;
    Real C1 = 12.0 / 35.0;
    Real C2 = 18.0 / 35.0;
    Real C3 = 4.0 / 35.0;
    //1  阶导数
    Real S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
    Real S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
    Real S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
    Real S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
    // 2 阶导数
    Real S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
    Real S21 = v3 - 2.0 * v4 + v5;
    Real S22 = v4 - 2.0 * v5 + v6;
    Real S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
    // 3 阶导数
    Real S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
    Real S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
    Real S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
    Real S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
    // 光滑度量因子
    Real S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
    Real S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
    Real S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
    Real S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
    // // 新的光滑度量因子, Xinliang Li
    // Real S0 = S10 * S10 + S20 * S20;
    // Real S1 = S11 * S11 + S21 * S21;
    // Real S2 = S12 * S12 + S22 * S22;
    // Real S3 = S13 * S13 + S23 * S23;    
    // Alpha weights
    Real a0 = C0 / ((ep + S0) * (ep + S0));
    Real a1 = C1 / ((ep + S1) * (ep + S1));
    Real a2 = C2 / ((ep + S2) * (ep + S2));
    Real a3 = C3 / ((ep + S3) * (ep + S3));
    // Non-linear weigths
    Real W0 = a0 / (a0 + a1 + a2 + a3);
    Real W1 = a1 / (a0 + a1 + a2 + a3);
    Real W2 = a2 / (a0 + a1 + a2 + a3);
    Real W3 = a3 / (a0 + a1 + a2 + a3);
    // 4阶差分格式的通量
    Real q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
    Real q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
    Real q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
    Real q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
    // 由4个4阶差分格式组合成1个7阶差分格式
    return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}
Real weno7Z_P(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 0;
    Real v1 = *(f + k - 3);
    Real v2 = *(f + k - 2);
    Real v3 = *(f + k - 1);
    Real v4 = *(f + k);
    Real v5 = *(f + k + 1);
    Real v6 = *(f + k + 2);
    Real v7 = *(f + k + 3);

    Real ep = 1.0e-7;
    Real C0 = 1.0 / 35.0;
    Real C1 = 12.0 / 35.0;
    Real C2 = 18.0 / 35.0;
    Real C3 = 4.0 / 35.0;
    //1  阶导数
    Real S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
    Real S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
    Real S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
    Real S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
    // 2 阶导数
    Real S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
    Real S21 = v3 - 2.0 * v4 + v5;
    Real S22 = v4 - 2.0 * v5 + v6;
    Real S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
    // 3 阶导数
    Real S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
    Real S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
    Real S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
    Real S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
    // 光滑度量因子
    Real S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
    Real S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
    Real S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
    Real S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
    // Alpha weights
    Real tau7 = std::abs(S0 - S3);
    Real a0 = C0 * (1.0 + tau7 / (S0 + 1.0e-40));
    Real a1 = C1 * (1.0 + tau7 / (S1 + 1.0e-40));
    Real a2 = C2 * (1.0 + tau7 / (S2 + 1.0e-40));
    Real a3 = C3 * (1.0 + tau7 / (S3 + 1.0e-40));
    // Non-linear weigths
    Real W0 = a0 / (a0 + a1 + a2 + a3);
    Real W1 = a1 / (a0 + a1 + a2 + a3);
    Real W2 = a2 / (a0 + a1 + a2 + a3);
    Real W3 = a3 / (a0 + a1 + a2 + a3);
    // 4阶差分格式的通量
    Real q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
    Real q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
    Real q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
    Real q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
    // 由4个4阶差分格式组合成1个7阶差分格式
    return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}
Real weno7Z_M(Real *f, Real delta)
{
    //assign value to v1, v2,...
    int k = 1;
    Real v1 = *(f + k + 3);
    Real v2 = *(f + k + 2);
    Real v3 = *(f + k + 1);
    Real v4 = *(f + k);
    Real v5 = *(f + k - 1);
    Real v6 = *(f + k - 2);
    Real v7 = *(f + k - 3);

    Real ep = 1.0e-7;
    Real C0 = 1.0 / 35.0;
    Real C1 = 12.0 / 35.0;
    Real C2 = 18.0 / 35.0;
    Real C3 = 4.0 / 35.0;
    //1  阶导数
    Real S10 = -2.0 / 6.0 * v1 + 9.0 / 6.0 * v2 - 18.0 / 6.0 * v3 + 11.0 / 6.0 * v4;
    Real S11 = 1.0 / 6.0 * v2 - 6.0 / 6.0 * v3 + 3.0 / 6.0 * v4 + 2.0 / 6.0 * v5;
    Real S12 = -2.0 / 6.0 * v3 - 3.0 / 6.0 * v4 + 6.0 / 6.0 * v5 - 1.0 / 6.0 * v6;
    Real S13 = -11.0 / 6.0 * v4 + 18.0 / 6.0 * v5 - 9.0 / 6.0 * v6 + 2.0 / 6.0 * v7;
    // 2 阶导数
    Real S20 = -v1 + 4.0 * v2 - 5.0 * v3 + 2.0 * v4;
    Real S21 = v3 - 2.0 * v4 + v5;
    Real S22 = v4 - 2.0 * v5 + v6;
    Real S23 = 2.0 * v4 - 5.0 * v5 + 4.0 * v6 - 1.0 * v7;
    // 3 阶导数
    Real S30 = -v1 + 3.0 * v2 - 3.0 * v3 + v4;
    Real S31 = -v2 + 3.0 * v3 - 3.0 * v4 + v5;
    Real S32 = -v3 + 3.0 * v4 - 3.0 * v5 + v6;
    Real S33 = -v4 + 3.0 * v5 - 3.0 * v6 + v7;
    // 光滑度量因子
    Real S0 = S10 * S10 + 13.0 / 12.0 * S20 * S20 + 1043.0 / 960.0 * S30 * S30 + 1.0 / 12.0 * S10 * S30;
    Real S1 = S11 * S11 + 13.0 / 12.0 * S21 * S21 + 1043.0 / 960.0 * S31 * S31 + 1.0 / 12.0 * S11 * S31;
    Real S2 = S12 * S12 + 13.0 / 12.0 * S22 * S22 + 1043.0 / 960.0 * S32 * S32 + 1.0 / 12.0 * S12 * S32;
    Real S3 = S13 * S13 + 13.0 / 12.0 * S23 * S23 + 1043.0 / 960.0 * S33 * S33 + 1.0 / 12.0 * S13 * S33;
    // Alpha weights
    Real tau7 = std::abs(S0 - S3);
    Real a0 = C0 * (1.0 + tau7 / (S0 + 1.0e-40));
    Real a1 = C1 * (1.0 + tau7 / (S1 + 1.0e-40));
    Real a2 = C2 * (1.0 + tau7 / (S2 + 1.0e-40));
    Real a3 = C3 * (1.0 + tau7 / (S3 + 1.0e-40));    
    // Non-linear weigths
    Real W0 = a0 / (a0 + a1 + a2 + a3);
    Real W1 = a1 / (a0 + a1 + a2 + a3);
    Real W2 = a2 / (a0 + a1 + a2 + a3);
    Real W3 = a3 / (a0 + a1 + a2 + a3);
    // 4阶差分格式的通量
    Real q0 = -3.0 / 12.0 * v1 + 13.0 / 12.0 * v2 - 23.0 / 12.0 * v3 + 25.0 / 12.0 * v4;
    Real q1 = 1.0 / 12.0 * v2 - 5.0 / 12.0 * v3 + 13.0 / 12.0 * v4 + 3.0 / 12.0 * v5;
    Real q2 = -1.0 / 12.0 * v3 + 7.0 / 12.0 * v4 + 7.0 / 12.0 * v5 - 1.0 / 12.0 * v6;
    Real q3 = 3.0 / 12.0 * v4 + 13.0 / 12.0 * v5 - 5.0 / 12.0 * v6 + 1.0 / 12.0 * v7;
    // 由4个4阶差分格式组合成1个7阶差分格式
    return W0 * q0 + W1 * q1 + W2 * q2 + W3 * q3;
}