// #pragma once

// #include "include/global_setup.h"
// // #include "cvodes/cvodes.h"
// // #include "cvodes/cvodes_direct.h"
// // #include "nvector/nvector_serial.h"
// // #include "sunmatrix/sunmatrix_dense.h"
// // #include "sunlinsol/sunlinsol_dense.h"

// class CanteraInterface
// {
// private:
// 	std::string Type = "ConstVolume";
// 	// number of species and solving parameters(=number of species+tempreature+mass).
// 	sunindextype m_nspecies, m_neq, m_nv;
// 	// mass, volume, density, tempreature, pressure, mean molecular weight, enthalpy, internal Energy of the mixture (kg kmol-1).
// 	real_t m_mass, m_vol, m_dens, m_tmp, m_p, m_mmw, m_enthalpy, m_inEnergy;
// 	// mole fraction, mole concentration of each species
// 	std::vector<real_t> m_ym, m_xm;

// 	void *cvode_mem = nullptr;
// 	// m_y[k] = mass fraction of species k whose summation = 1.
// 	// m_ym[k] = mole fraction of species k divided by the mean molecular.
// 	// ydot[k] =f(y,t) used in cvode_rhs user-defined function.
// 	N_Vector m_y, ydot;
// 	// for CVodeSVtolerances scalar(m_reltol) and vector(m_abstol) tolerances.
// 	// realtype m_reltol;
// 	N_Vector m_abstol;
// 	// current time and current time of integration.
// 	realtype m_t, m_tInteg;
// 	SUNMatrix m_linsol_matrix = nullptr;
// 	SUNLinearSolver m_linsol = nullptr;

// 	float CpTfinal, CvTfinal, CpTfinalXFCVode, CvTfinalXFCVode, CpTfinalXFQ2, CvTfinalXFQ2;

// public:
// 	Thermal *tm;
// 	Reaction *rn;
// 	static ZdCrtl zl;
// 	int m_maxsteps = 2000;

// 	CanteraInterface(){};
// 	~CanteraInterface(){};
// 	CanteraInterface(Thermal *Tm, Reaction *Rn, const size_t NSpecies, ZdCrtl Zl = zl);

// 	float CVodeSolver();
// 	void IniCVode();
// 	void IntegrateCVode(real_t tout);
// 	void evalCp(real_t t, real_t *y, real_t *ydot);
// 	void evalCv(real_t t, real_t *y, real_t *ydot);

// 	float ChemQ2Solver();

// 	void updatestatesCp(real_t *y);
// 	void updatestatesCv(real_t *y);
// 	void setState_TP(real_t const tem, real_t const pre);
// 	void setState_TD(real_t const tem, real_t const dens);
// 	real_t setDensity(real_t const den);
// 	real_t setPressure(real_t const pre);
// 	real_t setTemperature(real_t const tem);
// };
