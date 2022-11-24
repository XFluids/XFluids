#pragma once

#include <math.h>
#include "global_class.h"

Real minmod(Real r);
Real van_Leer(Real r);
Real van_Albada(Real r);
void MUSCL(Real p[4], Real &LL, Real &RR, int flag);
Real KroneckerDelta(const int i, const int j);
Real RoeAverage(const Real ul, const Real ur, const Real D, const Real D1);
// schemes
Real upwind_P(Real *f, Real delta);					Real upwind_M(Real *f, Real delta);
// 5th-upwind
Real linear_5th_P(Real *f, Real delta);
Real linear_5th_M(Real *f, Real delta);
Real linear_2th(Real *f, Real delta);
Real linear_4th(Real *f, Real delta);
Real linear_6th(Real *f, Real delta);
Real linear_3rd_P(Real *f, Real delta);
Real linear_3rd_M(Real *f, Real delta);
Real du_upwind5(Real *f, Real delta);
Real f2_upwind5(Real *f, Real delta);
// original weno
Real weno_P(Real *f, Real delta);
Real weno_M(Real *f, Real delta);
extern SYCL_EXTERNAL Real weno5old_P(Real *f, Real delta);
extern SYCL_EXTERNAL Real weno5old_M(Real *f, Real delta);
Real weno7_P(Real *f, Real delta);							Real weno7_M(Real *f, Real delta);
// 6th-order weno
Real WENOCU6_P(Real *f, Real delta);				Real WENOCU6_M(Real *f, Real delta);
Real WENOCU6M1_P(Real *f, Real delta);		Real WENOCU6M1_M(Real *f, Real delta);
Real WENOCU6M2_P(Real *f, Real delta);			Real WENOCU6M2_M(Real *f, Real delta);
Real TENO5_P(Real *f, Real delta);						Real TENO5_M(Real *f, Real delta);
Real TENO6_OPT_P(Real *f, Real delta);			Real TENO6_OPT_M(Real *f, Real delta);
// wenoZ
Real weno5Z_P(Real *f, Real delta);						Real weno5Z_M(Real *f, Real delta);
Real weno7Z_P(Real *f, Real delta);							Real weno7Z_M(Real *f, Real delta);
// WENO-AO
Real WENOAO53_P(Real *f, Real delta);						Real WENOAO53_M(Real *f, Real delta);
Real WENOAO73_P(Real *f, Real delta);						Real WENOAO73_M(Real *f, Real delta);
Real WENOAO753_P(Real *f, Real delta);					Real WENOAO753_M(Real *f, Real delta);

Real Weno5L2_P(Real *f, Real delta, Real lambda);
Real Weno5L2_M(Real *f, Real delta, Real lambda);