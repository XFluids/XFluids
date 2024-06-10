#include <cmath>
#include "XThermal.h"
#include "kinetics1.hpp"
#include "cantera_interface.h"

static void check_flag(void *flagvalue, const char *funcname, int opt, std::string msg)
{
	int *errflag, if_error = 0;

	/* Check if SUNDIALS function returned NULL pointer - no memory allocated */
	if (opt == 0 && flagvalue == NULL)
	{
		fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
				funcname);
		if_error = 1;
	}

	/* Check if flag < 0 */
	else if (opt == 1)
	{
		errflag = (int *)flagvalue;
		if (*errflag < 0)
		{
			fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
					funcname, *errflag);
			if_error = 1;
		}
	}

	/* Check if function returned NULL pointer - no memory allocated */
	else if (opt == 2 && flagvalue == NULL)
	{
		fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
				funcname);
		if_error = 1;
	}

	if (if_error)
		std::cout << msg << std::endl, abort();
}

static void checkError(long flag, const std::string &ctMethod,
					   const std::string &cvodesMethod)
{
	if (flag == CV_SUCCESS)
		return;
	else if (flag == CV_MEM_NULL)
		throw Cantera::CanteraError("CanteraInterface::" + ctMethod,
									"CVODES integrator is not initialized");
	else
	{
		const char *flagname = CVodeGetReturnFlagName(flag);
		throw Cantera::CanteraError("CanteraInterface::" + ctMethod,
									"{} returned error code {} ({})\n",
									cvodesMethod, flag, flagname);
	}
}

static int cvodes_rhs(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
	CanteraInterface *f = (CanteraInterface *)user_data;
	f->eval(t, (real_t *)NV_DATA_S(y), (real_t *)NV_DATA_S(ydot));

	return 0;
}

ZdCrtl CanteraInterface::zl = ZdCrtl();

CanteraInterface::CanteraInterface(Thermal *Tm, Reaction *Rn, const size_t NSpecies, ZdCrtl Zl)
{
	kinetics1(zl, 0, 0);

	// Ini Thermal and Reaction objects.
	tm = Tm, rn = Rn, m_p = Zl.P, m_tmp = Zl.T;
	// Both tempreature and mass are invoked in cvode solving processing.
	m_nspecies = NSpecies, m_neq = NSpecies + 2, m_nv = m_neq;
	// Ini mole fraction vector.
	m_ym.resize(NSpecies), m_xm.resize(NSpecies);

	// Ini states
	m_y = N_VNew_Serial(m_neq);
	check_flag((void *)m_y, "N_VNew_Serial", 0, "create vector y in VODE faliure! "), N_VConst(0.0, m_y);
	ydot = N_VNew_Serial(m_neq), m_t = 0, m_tInteg = m_t;
	check_flag((void *)ydot, "N_VNew_Serial", 0, "create vector ydot in VODE faliure! "), N_VConst(0.0, ydot);
	real_t *yi = tm->species_ratio_in, *m_yi = (real_t *)NV_DATA_S(m_y);
	for (size_t i = 0; i < m_nspecies; i++)
		m_yi[i + 2] = yi[i];
	m_vol = 1, m_mass = m_p / (m_tmp * CopR(tm->_Wi, yi)) * m_vol, m_yi[0] = m_mass, m_yi[1] = m_tmp;

	updatestates((real_t *)NV_DATA_S(m_y));

	// Processing CVode solver
	CVodeSolver();
}

void CanteraInterface::CVodeSolver()
{
	IniCVode(); // Ini
	std::cout << "time (s),Temperature (K),Density (kg/m3),Pressure (Pa)" << std::endl;
	for (size_t i = 0; i <= zl.nsteps; i++)
	{
		IntegrateCVode(i * zl.dt);

		std::cout << m_t << ", ";
		// for (size_t j = 1; j < m_neq; j++)
		// 	std::cout << NV_Ith_S(m_y, j) << ", ";
		std::cout << NV_Ith_S(m_y, 1) << ", ";
		std::cout << NV_Ith_S(m_y, 0) / m_vol << ", ";
		std::cout << m_p << ", ";
		for (size_t j = 0; j < m_xm.size() - 1; j++)
			std::cout << m_xm[j] << ", ";
		std::cout << m_xm[m_xm.size() - 1];
		std::cout << std::endl;
	}
	updatestates((real_t *)NV_DATA_S(m_y));
}

void CanteraInterface::IniCVode()
{
	// Create CVode solver
	cvode_mem = CVodeCreate(CV_BDF);
	check_flag((void *)cvode_mem, "CVodeCreate", 0, "create solver memory in VODE faliure! ");

	// Ini cvode solver
	checkError(CVodeInit(cvode_mem, cvodes_rhs, m_t, m_y), "IniCVode", "CVodeInit");

	// Ini tolerance
	m_reltol = 1E-9;
	m_abstol = N_VNew_Serial(m_neq);
	check_flag((void *)m_abstol, "N_VNew_Serial", 0, "create vector abstol in VODE faliure! "), N_VConst(1.0E-15, m_abstol);
	checkError(CVodeSVtolerances(cvode_mem, m_reltol, m_abstol), "IniCVode", "CVodeSVtolerances");

	// Ini UserData, UserDate type data is called by cvodes_rhs
	CVodeSetUserData(cvode_mem, this);

	// Ini dense matrix used by cvodes solvers based on dense matrix, such as SUNDenseLinearSolver
	SUNMatDestroy(m_linsol_matrix);
	m_linsol_matrix = SUNDenseMatrix((sunindextype)m_neq, (sunindextype)m_neq);
	m_linsol = SUNDenseLinearSolver(m_y, m_linsol_matrix);
	CVDlsSetLinearSolver(cvode_mem, m_linsol, m_linsol_matrix);

	// Set Max steps
	CVodeSetMaxNumSteps(cvode_mem, m_maxsteps);
}

void CanteraInterface::IntegrateCVode(real_t tout)
{
	if (tout == m_t)
		return;
	else if (tout < m_t)
		throw Cantera::CanteraError("CVodesIntegrator::integrate",
									"Cannot integrate backwards in time.\n"
									"Requested time {} < current time {}",
									tout, m_t);
	// std::cout << "CanteraInterface::IntegrateCVode, Cannot integrate backwards in time.\n  Requested time {"
	// 		  << tout << "} < current time {" << m_t << "}.",
	// 	abort();
	int nsteps = 0;
	while (m_tInteg < tout)
	{
		if (nsteps >= m_maxsteps)
			throw Cantera::CanteraError("CVodesIntegrator::integrate",
										"Maximum number of timesteps ({}) taken without reaching output "
										"time ({}).\nCurrent integrator time: {}",
										nsteps, tout, m_tInteg);
		// std::cout << "CanteraInterface::IntegrateCVode, Maximum number of timesteps ({" << nsteps
		// 		  << "}) taken without reaching output time ({" << tout
		// 		  << "}).\nCurrent integrator time: {" << m_tInteg << "}.\n",
		// 	abort();

		/**
		 * @brief execute one cvode step
		 *   The CV_NORMAL option causes the solver to take internal steps until it has reached or just passed tout.
		 * The solver then interpolates in order to return an approximate value of y(tout).
		 *   The CV_ONE_STEP option tells the solver to take just one internal step and then return the solution at the point reached by that step.
		 */
		checkError(CVode(cvode_mem, tout, m_y, &m_tInteg, CV_ONE_STEP), "IntegrateCVode", "CVode");
		// if (flag != CV_SUCCESS)
		// std::cout << "CanteraInterface::IntegrateCVode, CVodes error encountered. Error code: {" << flag
		// 		  << "Components with largest weighted error estimates:\n",
		// 	abort();
		nsteps++;
	}
	checkError(CVodeGetDky(cvode_mem, tout, 0, m_y), "IntegrateCVode", "CVodeGetDky");
	m_t = tout;
}

real_t CanteraInterface::setTemperature(real_t const tem)
{
	if (tem > 0)
		m_tmp = tem;
	else
		throw Cantera::CanteraError("Phase::setTemperature",
									"temperature must be positive. T = {}", m_tmp);

	return m_tmp;
}

real_t CanteraInterface::setDensity(real_t const den)
{
	if (den > 0.0)
		m_dens = den;
	else
		throw Cantera::CanteraError("Phase::setDensity",
									"density must be positive. density = {}", den);

	return m_dens;
}

void CanteraInterface::setState_TP(real_t const tem, real_t const pre)
{
	setTemperature(tem);
	setDensity(pre * m_mmw / (m_tmp * Ru));
}

void CanteraInterface::updatestates(real_t *y)
{
	Cantera::checkFinite("y", (double *)y, m_nv);
	// The components of y are [0] the total mass, [1] the temperature,
	// [2...K+2) are the mass fractions of each species, and [K+2...] are the
	// coverages of surface species on each wall.
	m_mass = y[0];
	const real_t *y1 = static_cast<const real_t *>(y + 2);
	std::transform(y + 2, y + m_neq, tm->_Wi, m_ym.begin(), std::multiplies<real_t>());
	m_mmw = 1.0 / std::accumulate(m_ym.begin(), m_ym.end(), 0.0);
	Cantera::scale(m_ym.begin(), m_ym.end(), m_xm.begin(), m_mmw); // mole fraction

	setState_TP(y[1], m_p);
	m_vol = m_mass / m_dens;
	m_enthalpy = CopEnthalpy(*tm, y1, m_tmp);
	m_inEnergy = m_enthalpy - CopR(tm->_Wi, y1) * m_tmp;
}

void CanteraInterface::eval(real_t t, real_t *y, real_t *ydot)
{
	m_t = t;
	updatestates(y);

	ydot[0] = 0, ydot[1] = 0;
	real_t *yi = y + 2, *yidot = ydot + 2;

	evalcpWrapper(tm, rn, y + 1, ydot + 1, m_dens, m_p);

	// evalcoreWrapper(tm, rn, yi, yidot, m_dens, m_p, m_tmp);

	// // get Kf Kb
	// const int _NR = NUM_REA, _NS = NUM_SPECIES;
	// real_t Kf[_NR], _Kck[_NR];
	// real_t logStandConc = std::log(m_p / (universal_gas_const * m_tmp));
	// for (size_t m = 0; m < _NR; m++)
	// {
	// 	real_t DeltaGibbs = _DF(0.0);
	// 	real_t A = rn->Rargus[m * 6 + 0], B = rn->Rargus[m * 6 + 1], E = rn->Rargus[m * 6 + 2];
	// 	Kf[m] = A * std::exp(B * log(m_tmp) - E * _DF(4.184) / Ru / m_tmp);
	// 	int *Nu_dm_ = rn->Nu_d_ + m * _NS, m_dn = _DF(0.0);
	// 	for (size_t n = 0; n < _NS; n++)
	// 	{
	// 		DeltaGibbs += Nu_dm_[n] * (std::log(m_p / _DF(101325.0)) - Gibson(*tm, m_tmp, n));
	// 		m_dn += Nu_dm_[n];
	// 	}
	// 	_Kck[m] = std::min(exp(DeltaGibbs - m_dn * logStandConc), _DF(1.0E+40));
	// }
	// // get yidot(production date of species), namely ydot[2] to ydot[end]
	// real_t m_cm[_NS];
	// // get mole concentrations from mass fraction
	// for (size_t n = 0; n < _NS; n++)
	// 	m_cm[n] = yi[n] * tm->_Wi[n] * m_dens * _DF(1.0E-3);

	// for (int react_id = 0; react_id < _NR; react_id++)
	// {
	// 	// third-body collision effect
	// 	real_t tb = _DF(0.0);
	// 	if (1 == rn->third_ind[react_id])
	// 	{
	// 		for (int it = 0; it < _NS; it++)
	// 			tb += rn->React_ThirdCoef[react_id * _NS + it] * m_cm[it];
	// 	}
	// 	else
	// 		tb = _DF(1.0);

	// 	Kf[react_id] *= tb; // forward
	// 	int *nu_f = rn->Nu_f_ + react_id * _NS;
	// 	_Kck[react_id] *= Kf[react_id]; // backward
	// 	int *nu_b = rn->Nu_b_ + react_id * _NS;
	// 	for (int it = 0; it < _NS; it++)					// forward
	// 		Kf[react_id] *= std::pow(m_cm[it], nu_f[it]);	// ropf
	// 	for (int it = 0; it < _NS; it++)					// backward
	// 		_Kck[react_id] *= std::pow(m_cm[it], nu_b[it]); // ropb

	// 	Kf[react_id] -= _Kck[react_id];
	// }

	// // // get omega dot (production rate in the form of mole) units: mole*cm^-s*s^-1
	// for (int n = 0; n < _NS; n++)
	// 	yidot[n] = _DF(.0);
	// for (int react_id = 0; react_id < _NR; react_id++)
	// {
	// 	int *nu_d = rn->Nu_d_ + react_id * _NS;
	// 	for (int n = 0; n < _NS; n++)
	// 		yidot[n] += nu_d[n] * Kf[react_id]; // // get omega dot
	// }

	// // production rate in the form of mass;
	// for (int n = 0; n < NUM_SPECIES; n++)
	// {
	// 	yidot[n] *= tm->Wi[n] / m_dens;
	// 	ydot[1] -= yidot[n] * Enthalpy(*tm, m_tmp, n);
	// } // ydot[1] = mass * c_p * dT/dt while the loop ends, we need get dT/dt
	// // get ydot
	// ydot[1] /= CopHeatCapacity(*tm, y + 2, m_tmp);

	Cantera::checkFinite("ydot", (double *)ydot, m_nv);
}
