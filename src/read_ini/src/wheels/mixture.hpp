#pragma once

#include "../../../include/global_setup.h"

// ================================================================================
// // // for mixture property calculation
// ================================================================================
/**
 *@brief calculate R for every cell
 */
real_t get_MixtureR(real_t *species_chara, const real_t *yi)
{
	real_t R = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		R += yi[n] * Ru / (species_chara[n * SPCH_Sz + Wi]);
	}
	return R;
}

/**
 * @brief calculate W of the mixture at given point
 */
real_t get_MixtureW(Thermal thermal, const real_t *yi)
{
	real_t _W = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
	{
		_W += yi[ii] / (thermal.species_chara[ii * SPCH_Sz + Wi]); // Wi
	}
	// printf("W=%lf \n", 1.0 / _W);
	return _DF(1.0) / _W;
}

// ================================================================================
// // // for viscosity calculation
// ================================================================================
/**
 * @brief get quadratic interpolation coefficient for y=ax*x+b*x+c
 * @para x1,x2,x3 :interpolation node;
 * @para f1,f2,f2 :interpolated function value at corresponding nodes
 */
void GetQuadraticInterCoeff(real_t x1, real_t x2, real_t x3, real_t f1, real_t f2, real_t f3, real_t *aa)
{
	aa[2] = ((f1 - f2) / (x1 - x2) - (f2 - f3) / (x2 - x3)) / (x1 - x3);
	aa[1] = (f1 - f2) / (x1 - x2) - aa[2] * (x1 + x2);
	aa[0] = f1 - aa[1] * x1 - aa[2] * x1 * x1;
}

/**
 * @brief get Omega1 interpolated
 * @para T_star reduced temperature of species jk;
 */
real_t Omega1_interpolated(const real_t Tstar)
{ // 公式拟合扩散中的第一类碰撞积分
	real_t A = _DF(1.06036), B = _DF(-0.1561), C = _DF(0.19300), D = _DF(-0.47635), E = _DF(1.03587), F = _DF(-1.52996), G = _DF(1.76474), H = _DF(-3.89411);
	real_t Omega = A * std::pow(Tstar, B) + C * std::exp(D * Tstar) + E * std::exp(F * Tstar) + G * std::exp(H * Tstar);
	return Omega;
}

/**
 * @brief get Omega2 interpolated
 * @para T_star reduced temperature;
 */
real_t Omega2_interpolated(const real_t Tstar)
{ // 公式拟合粘性中的第二类碰撞积分
	real_t A = _DF(1.16145), B = _DF(-0.14874), C = _DF(0.52487), D = _DF(-0.7732), E = _DF(2.16178), F = _DF(-2.43787);
	real_t Omega = A * std::pow(Tstar, B) + C * std::exp(D * Tstar) + E * std::exp(F * Tstar);
	return Omega;
}

/**
 * @brief get Zrot at temperature T; equation 5-34
 * @para x epsilon_kB/T
 */
real_t ZrotFunc(real_t x)
{
	real_t zrot = _DF(1.0) + std::sqrt(pi * x) * pi / _DF(2.0) + x * (_DF(0.25) * pi * pi + _DF(2.0)) + std::pow(pi * x, _DF(1.5));
	return zrot;
}

/**
 * @brief solve overdetermined linear equations 超定方程组求解
 * @para AA 2-D array for coefficients matrix;
 * @para b 1-D colum vector for RHS;
 * @para mm the number of rows(fitting nodes);
 * @para xx 1-D the solution vector(the coefficient of fitting polynomial);
 */
void Solve_Overdeter_equations(real_t AA[][order_polynominal_fitted], real_t *b, int mm, real_t *xx)
{
	int nn = order_polynominal_fitted;
	real_t CC[nn][nn], dd[nn]; // the coefficient matrix and RHS column vector of the corresponding normal equatons
	for (int i = 0; i < nn; i++)
	{
		dd[i] = 0.0;
		for (int j = 0; j < nn; j++)
			CC[i][j] = 0.0;
	}
	for (int i = 0; i < nn; i++)
	{
		for (int q = 0; q < mm; q++)
			dd[i] = dd[i] + AA[q][i] * b[q];
		for (int j = 0; j < nn; j++)
			for (int k = 0; k < mm; k++)
				CC[i][j] = CC[i][j] + AA[k][i] * AA[k][j];
	}
	// Cholesky decoposition of the coefficient Matrix CC
	CC[0][0] = std::sqrt(CC[0][0]);
	for (int p = 1; p < nn; p++)
		CC[p][0] = CC[p][0] / CC[0][0];
	for (int k = 1; k < nn; k++)
	{
		for (int m = 0; m < k; m++)
			CC[k][k] = CC[k][k] - CC[k][m] * CC[k][m];
		CC[k][k] = std::sqrt(CC[k][k]);
		for (int i = k + 1; i < nn; i++)
		{
			for (int m = 0; m < k; m++)
				CC[i][k] = CC[i][k] - CC[i][m] * CC[k][m];
			CC[i][k] = CC[i][k] / CC[k][k];
		}
	}
	// solve the equations
	// slove the lower triangle equations
	xx[0] = dd[0] / CC[0][0];
	for (int j = 1; j < nn; j++)
	{
		for (int q = 0; q < j; q++)
			dd[j] = dd[j] - CC[j][q] * xx[q];
		xx[j] = dd[j] / CC[j][j];
	}
	// solve the upper triangle equations:xx[nn] is the unknown RHS,b[nn] is for solution column vector
	dd[nn - 1] = xx[nn - 1] / CC[nn - 1][nn - 1];
	for (int i = nn - 2; i >= 0; i--)
	{
		for (int p = nn - 1; p > i; p--)
			xx[i] = xx[i] - CC[p][i] * dd[p]; // CC[i][p]=CC[p][i]
		dd[i] = xx[i] / CC[i][i];
	}
	for (int m = 0; m < nn; m++)
		xx[m] = dd[m];
}
