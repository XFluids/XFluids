#pragma once

// =======================================================
//    prepare for getting viscous flux new
#define MARCO_PREVISCFLUX()                                                                 \
	real_t F_wall_v[Emax], f_x, f_y, f_z, u_hlf, v_hlf, w_hlf;                              \
	real_t mue = ((viscosity_aver[id_p1] + viscosity_aver[id])) * _DF(0.5); /*mue at wall*/ \
	real_t lamada = -_DF(2.0) * _OT * mue;

#ifdef COP
#define MARCO_VIS_COP_IN_DIFFU1()                                                     \
	Yil_wall[l] = ((Yi[g_id_p1] - Yi[g_id])) * _dl; /* temperature gradient at wall*/ \
	Yi_wall[l] = ((Yi[g_id_p1] + Yi[g_id])) * _DF(0.5);                               \
	CorrectTerm += Dim_wall[l] * Yil_wall[l];
/**
 * NOTE: CorrectTerm for diffusion to Average the error from the last species to
 *  all species according to the mass fraction
 */
#else
#define MARCO_VIS_COP_IN_DIFFU1() Yil_wall[l] = _DF(0.0);
#endif // COP

// for error out of Vis
#if ESTIM_NAN

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

#endif // end ESTIM_NAN

#ifdef Visc_Diffu
#define MARCO_VIS_Diffu()                                                                             \
	real_t rho_wall = ((rho[id_p1] + rho[id])) * _DF(0.5), CorrectTerm = _DF(0.0);                    \
	real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yil_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];  \
	for (int l = 0; l < NUM_SPECIES; l++)                                                             \
	{                                                                                                 \
		int g_id_p1 = l + NUM_SPECIES * id_p1, g_id = l + NUM_SPECIES * id;                           \
		/*int g_id_p2 = l + NUM_SPECIES * id_p2, g_id_m1 = l + NUM_SPECIES * id_m1;*/                 \
		hi_wall[l] = ((hi[g_id_p1] + hi[g_id])) * _DF(0.5);                                           \
		Dim_wall[l] = ((Dkm_aver[g_id_p1] + Dkm_aver[g_id])) * _DF(0.5);                              \
		MARCO_VIS_COP_IN_DIFFU1();                                                                    \
		/* visc flux for heat of diffusion */                                                         \
		F_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yil_wall[l];                             \
		MARCO_Err_Dffu();                                                                             \
	}                                                                                                 \
	/* visc flux for cop equations*/                                                                  \
	CorrectTerm *= rho_wall;                                                                          \
	for (int p = 5; p < Emax; p++) /* ADD Correction Term in X-direction*/                            \
	{                                                                                                 \
		int p_temp = p - 5; /*CorrectTerm = 0.0 while not added in loop*/                             \
		F_wall_v[p] = rho_wall * Dim_wall[p_temp] * Yil_wall[p_temp] - Yi_wall[p_temp] * CorrectTerm; \
	}
#else // Visc_Diffu
#define MARCO_VIS_Diffu()                                \
	for (int p = 5; p < Emax; p++) /* Avoid NAN error */ \
		F_wall_v[p] = _DF(0.0);
#endif // Visc_Diffu

#ifdef Visc_Heat
#define MARCO_VIS_HEAT() /* thermal conductivity at wall*/                             \
	real_t kk = ((thermal_conduct_aver[id_p1] + thermal_conduct_aver[id])) * _DF(0.5); \
	kk *= ((T[id_p1] - T[id])) * _dl; /* temperature gradient at wall*/                \
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
