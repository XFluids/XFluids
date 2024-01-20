#pragma once

// =======================================================
//    prepare for getting viscous flux new
#define MARCO_PREVISCFLUX()                                                                                                                           \
	real_t F_wall_v[Emax], f_x, f_y, f_z, u_hlf, v_hlf, w_hlf;                                                                                        \
	real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) * _sxtn; /*mue at wall*/ \
	real_t lamada = -_DF(2.0) * _OT * mue;

#ifdef COP
#define MARCO_VIS_COP_IN_DIFFU1()                                                                                                                                                              \
	Yil_wall[l] = sycl::min(sycl::max((_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr, -Yil_limiter[l]), Yil_limiter[l]); /* temperature gradient at wall*/ \
	Yi_wall[l] = sycl::min(sycl::max((_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn, _DF(1.0E-20)), _DF(1.0));                                                    \
	Dim_Yil = sycl::min(sycl::max(Dim_wall[l] * Yil_wall[l], -Diffu_limiter[l]), Diffu_limiter[l]);                                                                                            \
	CorrectTerm += Dim_Yil;
/**
 * NOTE: CorrectTerm for diffusion to Average the error from the last species to
 *  all species according to the mass fraction
 */
#else
#define MARCO_VIS_COP_IN_DIFFU1() Yil_wall[l] = _DF(0.0);
#endif // COP

// for error out of Vis
#if ESTIM_OUT

#define MARCO_Err_Dffu()        \
	ErDimw[g_id] = Dim_wall[l]; \
	Erhiw[g_id] = hi_wall[l];   \
	ErYiw[g_id] = Yi_wall[l];   \
	ErYilw[g_id] = Yil_wall[l];

#define MARCO_Err_VisFw() \
	ErvisFw[n + Emax * id] = F_wall_v[n];

#else

#define MARCO_Err_Dffu() ;
#define MARCO_Err_VisFw() ;

#endif // end ESTIM_OUT

#if Visc_Diffu
#define MARCO_VIS_Diffu()                                                                                                                         \
	real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn, CorrectTerm = _DF(0.0);                            \
	real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES], Dim_Yil = _DF(1.0E-20);                      \
	for (int l = 0; l < NUM_SPECIES; l++)                                                                                                         \
	{                                                                                                                                             \
		int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1; \
		hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;                                                 \
		Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;                        \
		MARCO_VIS_COP_IN_DIFFU1();                                                                                                                \
		/* visc flux for heat of diffusion */                                                                                                     \
		F_wall_v[4] += rho_wall * hi_wall[l] * Dim_Yil;                                                                                           \
		MARCO_Err_Dffu();                                                                                                                         \
	}                                                                                                                                             \
	/* visc flux for cop equations*/                                                                                                              \
	CorrectTerm *= rho_wall;                                                                                                                      \
	for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                                                                        \
	{                                                                                                                                             \
		int p_temp = p - 5;                                                                                                                       \
		F_wall_v[p] = rho_wall * Dim_Yil - Yi_wall[p_temp] * CorrectTerm; /*CorrectTerm = 0.0 while not added in loop*/                           \
	}
#else // Visc_Diffu
#define MARCO_VIS_Diffu()                                \
	for (int p = 5; p < Emax; p++) /* Avoid NAN error */ \
		F_wall_v[p] = _DF(0.0);
#endif // Visc_Diffu

#if Visc_Heat
#define MARCO_VIS_HEAT() /* thermal conductivity at wall*/                                                                                                   \
	real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; \
	kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr; /* temperature gradient at wall*/                                          \
	F_wall_v[4] += kk;
#else // else Visc_Heat
#define MARCO_VIS_HEAT() ;
#endif // end Visc_Heat

// =======================================================
//    get viscous flux
#define MARCO_VISCFLUX()                                   \
	F_wall_v[0] = _DF(0.0);                                \
	F_wall_v[1] = f_x;                                     \
	F_wall_v[2] = f_y;                                     \
	F_wall_v[3] = f_z;                                     \
	F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf; \
	/* Fourier thermal conductivity*/                      \
	MARCO_VIS_HEAT();                                      \
	/* energy fiffusion depends on mass diffusion*/        \
	MARCO_VIS_Diffu();                                     \
	/* add viscous flux to fluxwall*/                      \
	for (size_t n = 0; n < Emax; n++)                      \
	{                                                      \
		Flux_wall[n + Emax * id] -= F_wall_v[n];           \
		MARCO_Err_VisFw();                                 \
	}

// =======================================================
//    prepare for getting viscous flux new
// #ifdef COP
// const bool _COP = true;
// #define MARCO_VIS_COP_IN_DIFFU1()                                                                                                       \
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;                                           \
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/ \
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];

// #define MARCO_VIS_COP_IN_DIFFU2()                                                                     \
//     CorrectTerm *= rho_wall;                                                                          \
//     for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                            \
//     {                                                                                                 \
//         int p_temp = p - 5;                                                                           \
//         F_wall_v[p] = rho_wall * Dim_wall[p_temp] * Yil_wall[p_temp] - Yi_wall[p_temp] * CorrectTerm; \
//     }

// #else
// const bool _COP = false;
// #define MARCO_VIS_COP_IN_DIFFU1() Yil_wall[l] = _DF(0.0);
// #define MARCO_VIS_COP_IN_DIFFU2() ;
// #endif

// #if Visc_Heat
// const bool _Heat = true;

// #define MARCO_VIS_HEAT()                                                                                                                                                                       \
//     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/ \
//     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/ \
//     F_wall_v[4] += kk;

// #define MARCO_VIS_HEAT_IN_DIFFU() \
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];

// // for (int l = 0; l < NUM_SPECIES; l++)                                 \
//     // {                                                                     \
//     //     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l]; \
//     // }
// #else
// const bool _Heat = false;
// #define MARCO_VIS_HEAT() ;
// #define MARCO_VIS_HEAT_IN_DIFFU() ;
// #endif

// #if Visc_Diffu
// const bool _Diffu = true;

// #define MARCO_VIS_Diffu()                                                                                                                         \
//     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;                                                    \
//     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES], CorrectTerm = _DF(0.0);                      \
//     for (int l = 0; l < NUM_SPECIES; l++)                                                                                                         \
//     {                                                                                                                                             \
//         int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1; \
//         hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;                                                 \
//         Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;                        \
//         MARCO_VIS_COP_IN_DIFFU1();                                                                                                                \
//         /* visc flux for heat of diffusion */                                                                                                     \
//         MARCO_VIS_HEAT_IN_DIFFU();                                                                                                                \
//     }                                                                                                                                             \
//     /* visc flux for cop equations*/                                                                                                              \
//     MARCO_VIS_COP_IN_DIFFU2();

// #else
// const bool _Diffu = false;
// #define MARCO_VIS_Diffu() ;
// #endif

// F_wall_v[0] = _DF(0.0);
// F_wall_v[1] = f_x;
// F_wall_v[2] = f_y;
// F_wall_v[3] = f_z;
// F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;
// int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;                /* Fourier thermal conductivity*/
// real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/
// kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/
// F_wall_v[4] += kk;                                                                                                                                       /* energy fiffusion depends on mass diffusion*/
// real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;
// real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES], CorrectTerm = _DF(0.0);
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;
//     Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];
// } /* visc flux for cop equations*/
// CorrectTerm *= rho_wall;
// for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/
// {
//     F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;
// }
// for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/
// {
//     Flux_wall[n + Emax * id] -= F_wall_v[n];
// }

// F_wall_v[0] = _DF(0.0);
// F_wall_v[1] = f_x;
// F_wall_v[2] = f_y;
// F_wall_v[3] = f_z;
// F_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;
// int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id, g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;                /* Fourier thermal conductivity*/
// real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) * _sxtn; /* thermal conductivity at wall*/
// kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) * _dl * _twfr;                                                                            /* temperature gradient at wall*/
// F_wall_v[4] += kk;
// real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) * _sxtn;
// real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     hi_wall[l] = (_DF(9.0) * (hi[g_id_p1] + hi[g_id]) - (hi[g_id_p2] + hi[g_id_m1])) * _sxtn;
//     Dim_wall[l] = (_DF(9.0) * (Dkm_aver[g_id_p1] + Dkm_aver[g_id]) - (Dkm_aver[g_id_p2] + Dkm_aver[g_id_m1])) * _sxtn;
//     Yi_wall[l] = (_DF(9.0) * (Yi[g_id_p1] + Yi[g_id]) - (Yi[g_id_p2] + Yi[g_id_m1])) * _sxtn;
//     Yil_wall[l] = (_DF(27.0) * (Yi[g_id_p1] - Yi[g_id]) - (Yi[g_id_p2] - Yi[g_id_m1])) * _dl * _twfr; /* temperature gradient at wall*/
// } /* visc flux for heat of diffusion */
// for (int l = 0; l < NUM_SPECIES; l++)
// {
//     F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];
// }
// real_t for (int l = 0; l < NUM_SPECIES; l++)
// {
//     CorrectTerm += Dim_wall[l] * Yil_wall[l];
// }
// CorrectTerm *= rho_wall;
// for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/
// {
//     F_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yil_wall[p - 5] - Yi_wall[p - 5] * CorrectTerm;
// }
// for (size_t n = 0; n < Emax; n++) /* add viscous flux to fluxwall*/
// {
//     Flux_wall[n + Emax * id] -= F_wall_v[n];
// }
