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
	// f->evalCp(t, (real_t *)NV_DATA_S(y), (real_t *)NV_DATA_S(ydot)); // NOTE: constant pressure model
	f->evalCv(t, (real_t *)NV_DATA_S(y), (real_t *)NV_DATA_S(ydot)); // NOTE: constant volume model

	return 0;
}

ZdCrtl CanteraInterface::zl = ZdCrtl();

CanteraInterface::CanteraInterface(Thermal *Tm, Reaction *Rn, const size_t NSpecies, ZdCrtl Zl)
{
	CpTfinal = kinetics1Cp(zl, 0, 0);
	CvTfinal = kinetics1Cv(zl, 0, 0);

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

	updatestatesCp((real_t *)NV_DATA_S(m_y));

	// Processing CVode solver
	CvTfinalXFCVode = CVodeSolver();

	// ReIni Thermal and Reaction objects.
	tm = Tm, rn = Rn, m_p = Zl.P, m_tmp = Zl.T;
	m_t = 0, m_tInteg = m_t;
	for (size_t i = 0; i < m_nspecies; i++)
		m_yi[i + 2] = yi[i];
	m_vol = 1, m_mass = m_p / (m_tmp * CopR(tm->_Wi, yi)) * m_vol, m_yi[0] = m_mass, m_yi[1] = m_tmp;

	updatestatesCp((real_t *)NV_DATA_S(m_y));

	// Processing ChemQ2 solver
	CvTfinalXFQ2 = ChemQ2Solver(); // if (ReactSources && ODETest_json)

	if (std::abs(CvTfinalXFCVode / CvTfinal - 1.0) > 0.001)
	{
		std::cout << "XFluids-CVode 0D-" << Type << " solving failed !" << std::endl;
		// abort();
	}
}

float CanteraInterface::CVodeSolver()
{
	IniCVode(); // Ini
	std::cout << "\nzone t='XFluids-CVode-" << Type << "'" << std::endl;
	for (size_t i = 0; i <= zl.nsteps; i++)
	{
		IntegrateCVode(i * zl.dt);

		std::cout << m_t << " ";
		// for (size_t j = 1; j < m_neq; j++)
		// 	std::cout << NV_Ith_S(m_y, j) << " ";
		std::cout << NV_Ith_S(m_y, 1) << " ";
		std::cout << NV_Ith_S(m_y, 0) / m_vol << " ";
		std::cout << m_p << " ";
		for (size_t j = 0; j < m_xm.size() - 1; j++)
			std::cout << m_xm[j] << " ";
		std::cout << m_xm[m_xm.size() - 1];
		std::cout << std::endl;
	}

	return NV_Ith_S(m_y, 1);
}

// NOTE: set to constant volume model now, model types in Chemq2 defined by updatestatesCp/Cv function and evalcp/cv in Chemq2Wrapper
float CanteraInterface::ChemQ2Solver()
{
	real_t *y = (real_t *)NV_DATA_S(m_y);
	std::cout << "\nzone t='XFluids-ChemQ2" << Type << "'" << std::endl;
	for (size_t i = 0; i <= zl.nsteps; i++)
	{
		m_t = i * zl.dt;
		updatestatesCv(y);

		std::cout << m_t << " ";
		// for (size_t j = 1; j < m_neq; j++)
		// 	std::cout << NV_Ith_S(m_y, j) << " ";
		std::cout << *(y + 1) << " ";
		std::cout << *(y + 0) / m_vol << " ";
		std::cout << m_p << " ";
		for (size_t j = 0; j < m_xm.size() - 1; j++)
			std::cout << m_xm[j] << " ";
		std::cout << m_xm[m_xm.size() - 1];
		std::cout << std::endl;

		// // NOTE: In this version, temperature is not as a solution element of vector y, and the test is corresponding to the CVode
		// Chemq2WrapperCv0(tm, rn, y + 1, zl.dt, m_dens, m_p);
		// NOTE: In this version, temperature is solved as a solution element of vector y like CVode does
		Chemq2WrapperCv1(tm, rn, y + 1, zl.dt, m_dens, m_p);
	}

	return *(y + 1);
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

real_t CanteraInterface::setPressure(real_t const pre)
{
	if (pre > 0)
		m_p = pre;
	else
		throw Cantera::CanteraError("Phase::setTemperature",
									"temperature must be positive. T = {}", pre);

	return pre;
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

void CanteraInterface::setState_TD(real_t const tem, real_t const dens)
{
	setTemperature(tem);
	setPressure(dens / m_mmw * (m_tmp * Ru));
}

void CanteraInterface::updatestatesCp(real_t *y)
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

void CanteraInterface::updatestatesCv(real_t *y)
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

	setState_TD(y[1], m_dens);
	m_enthalpy = CopEnthalpy(*tm, y1, m_tmp);
	m_inEnergy = m_enthalpy - CopR(tm->_Wi, y1) * m_tmp;
}

void CanteraInterface::evalCp(real_t t, real_t *y, real_t *ydot)
{
	m_t = t;
	updatestatesCp(y);

	ydot[0] = 0, ydot[1] = 0;
	real_t *yi = y + 2, *yidot = ydot + 2;

	// const int _NR = NUM_REA, _NS = NUM_SPECIES;
	// real_t q[_NR], p[_NR];
	// ydot[1] = evalcpWrapper(tm, rn, yi, q, p, m_dens, m_tmp, m_p);
	// for (size_t i = 0; i < _NR; i++)
	// 	yidot[i] = q[i] - p[i];

	// evalcpWrapper(tm, rn, y + 1, ydot + 1, m_dens, m_p);
	evalcpWrapper(tm, rn, y + 1, ydot + 1, m_dens, m_p);

	Cantera::checkFinite("ydot", (double *)ydot, m_nv);
}

void CanteraInterface::evalCv(real_t t, real_t *y, real_t *ydot)
{
	m_t = t;
	updatestatesCv(y);

	ydot[0] = 0, ydot[1] = 0;
	real_t *yi = y + 2, *yidot = ydot + 2;

	// const int _NR = NUM_REA, _NS = NUM_SPECIES;
	// real_t q[_NR], p[_NR];
	// ydot[1] = evalcpWrapper(tm, rn, yi, q, p, m_dens, m_tmp, m_p);
	// for (size_t i = 0; i < _NR; i++)
	// 	yidot[i] = q[i] - p[i];

	// evalcpWrapper(tm, rn, y + 1, ydot + 1, m_dens, m_p);
	evalcvWrapper(tm, rn, y + 1, ydot + 1, m_dens, m_p);

	Cantera::checkFinite("ydot", (double *)ydot, m_nv);
}
