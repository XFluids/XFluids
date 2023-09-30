#pragma once

#include "Utils_schemes.hpp"
#include "LowOrder_schemes.hpp"
#include "UpWind_schemes.hpp"
#include "WENO5s_schemes.hpp"
#include "WENO6s_schemes.hpp"
#include "WENO7s_schemes.hpp"
#include "TENOs_schemes.hpp"
#include "WENOAOs_schemes.hpp"

// =======================================================
//    Rename Reconstruction schemes
#define MARCO_WENO5 weno5old_GPU(&pp[3], &mm[3])
#define MARCO_WENOCU6 WENOCU6_GPU(&pp[3], &mm[3], dl)
#if SCHEME_ORDER == 5
#define WENO_GPU MARCO_WENO5
#elif SCHEME_ORDER == 6
#define WENO_GPU MARCO_WENOCU6
#endif

// =======================================================
//    Artificial_type in Flux Reconstruction
#if 1 == Artificial_type // ROE
#define Roe_type _DF(1.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(0.0)
#elif 2 == Artificial_type // LLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(1.0)
#define GLF_type _DF(0.0)
#elif 3 == Artificial_type // GLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(1.0)
#endif

// //-----------------------------------------------------------------------------------------
// //		the Schemes List
// //-----------------------------------------------------------------------------------------
// // // low order
// inline real_t minmod(real_t r);
// inline real_t van_Leer(real_t r);
// inline real_t van_Albada(real_t r);
// inline void MUSCL(real_t p[4], real_t &LL, real_t &RR, int flag);
// inline real_t KroneckerDelta(const int i, const int j);

// // // upwind schemes
// inline real_t upwind_P(real_t *f, real_t delta);
// inline real_t upwind_M(real_t *f, real_t delta);
// // // high order-upwind
// inline real_t linear_3rd_P(real_t *f, real_t delta);
// inline real_t linear_3rd_M(real_t *f, real_t delta);
// inline real_t linear_5th_P(real_t *f, real_t delta);
// inline real_t linear_5th_M(real_t *f, real_t delta);
// inline real_t linear_2th(real_t *f, real_t delta);
// inline real_t linear_4th(real_t *f, real_t delta);
// inline real_t linear_6th(real_t *f, real_t delta);
// inline real_t du_upwind5(real_t *f, real_t delta);
// inline real_t f2_upwind5(real_t *f, real_t delta);

// // // 5-th WENOs
// inline real_t weno5old_GPU(real_t *f, real_t *m);
// // inline real_t weno5old_P(real_t *f, real_t delta);
// // inline real_t weno5old_M(real_t *f, real_t delta);
// inline real_t weno5_P(real_t *f, real_t delta);
// inline real_t weno5_M(real_t *f, real_t delta);
// inline real_t weno5Z_P(real_t *f, real_t delta);
// inline real_t weno5Z_M(real_t *f, real_t delta);
// inline real_t Weno5L2_P(real_t *f, real_t delta, real_t lambda);
// inline real_t Weno5L2_M(real_t *f, real_t delta, real_t lambda);

// // // 6th-order WENOs
// inline real_t WENOCU6_GPU(real_t *f, real_t *m, real_t delta);
// // inline real_t WENOCU6_P(real_t *f, real_t delta);
// // inline real_t WENOCU6_M(real_t *f, real_t delta);
// inline real_t WENOCU6M1_P(real_t *f, real_t delta);
// inline real_t WENOCU6M1_M(real_t *f, real_t delta);
// inline real_t WENOCU6M2_P(real_t *f, real_t delta);
// inline real_t WENOCU6M2_M(real_t *f, real_t delta);

// // // 7th-order WENOs
// inline real_t weno7_P(real_t *f, real_t delta);
// inline real_t weno7_M(real_t *f, real_t delta);
// inline real_t weno7Z_P(real_t *f, real_t delta);
// inline real_t weno7Z_M(real_t *f, real_t delta);

// // // TENOs
// inline real_t TENO5_P(real_t *f, real_t delta);
// inline real_t TENO5_M(real_t *f, real_t delta);
// inline real_t TENO6_OPT_P(real_t *f, real_t delta);
// inline real_t TENO6_OPT_M(real_t *f, real_t delta);

// // // WENO-AOs
// inline real_t WENOAO53_P(real_t *f, real_t delta);
// inline real_t WENOAO53_M(real_t *f, real_t delta);
// inline real_t WENOAO73_P(real_t *f, real_t delta);
// inline real_t WENOAO73_M(real_t *f, real_t delta);
// inline real_t WENOAO753_P(real_t *f, real_t delta);
// inline real_t WENOAO753_M(real_t *f, real_t delta);
