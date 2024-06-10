#include "../setupini.h"
#include "wheels/mixture.hpp"
#include "../../solver_Ini/Mixing_device.h"
#include "../../solver_Reconstruction/viscosity/Visc_device.h"

// =======================================================
// =======================================================
bool Setup::Mach_Shock()
{ // called after Setup::get_Yi();, only support X dir shock
    real_t Ma_1 = ini.Ma * ini.Ma;
    if (Ma_1 < 1.0)
        return false;

    real_t p2 = ini.blast_pressure_out; // P2
    real_t T2 = ini.blast_T_out;        // T2

    real_t R, Gamma_m2;

    // #ifdef DEBUG
    //     R = 1399.0 * 0.33;
    //     Gamma_m2 = 1.33;
    // #else
    R = get_MixtureR(h_thermal.species_chara, h_thermal.species_ratio_out);
    Gamma_m2 = get_CopGamma(h_thermal, h_thermal.species_ratio_out, T2);
    // #endif                                        // end DEBUG
    real_t c2 = std::sqrt(Gamma_m2 * R * T2); // sound speed downstream the shock
    ini.blast_c_out = c2, ini.blast_gamma_out = Gamma_m2, ini.tau_H = _DF(2.0) * ini.xa / (ini.Ma * ini.blast_c_out);
    ini.blast_density_out = p2 / R / T2; // rho2
    real_t rho2 = ini.blast_density_out; // rho2

    // {
    // // Positive shock wave theroy
    // real_t Ma_2 = (_DF(1.0) + _DF(0.5) * (Gamma_m2 - _DF(1.0)) * Ma_1) / (Gamma_m2 * Ma_1 - _DF(0.5) * (Gamma_m2 - _DF(1.0)));
    // ini.blast_u_out = std::sqrt(Ma_2) * c2;                                                                                  // u2
    // ini.blast_density_in = ini.blast_density_out / ((Gamma_m2 + _DF(1.0)) * Ma_1 / (_DF(2.0) + (Gamma_m2 - _DF(1.0)) * Ma_1)); // same gas component as the downstream shock
    // ini.blast_pressure_in = p2 / (_DF(1.0) + _DF(2.0) * Gamma_m2 / (Gamma_m2 + _DF(1.0)) * (Ma_1 - _DF(1.0)));                 // positive shock wave relationship equation of pressure
    // ini.blast_u_in = ini.blast_u_out * ini.blast_density_out / ini.blast_density_in;
    // }

    ini.blast_v_in = ini.blast_v_out;
    ini.blast_w_in = ini.blast_w_out;
    ini.cop_pressure_in = ini.blast_pressure_out;
    ini.cop_T_in = ini.blast_T_out;
    real_t R_cop = get_MixtureR(h_thermal.species_chara, h_thermal.species_ratio_in);
    ini.cop_density_in = ini.cop_pressure_in / R_cop / ini.cop_T_in;

    // //  SBI Shock wave mach theroy
    real_t Ma, e2, u2, E2, Si, T1, e1, E1, p1, rho1, u1; // *1: 激波上游; *2: 激波下游;

#ifdef DEBUG
#define MARCO_Coph(T) 1860.67 * T + 1990000
    Ma = 2.83;
#else
    Ma = ini.Ma;
#define MARCO_Coph(T) get_Coph(h_thermal, h_thermal.species_ratio_out, T)
#endif // end DEBUG

    if (!Mach_Modified)
    {
        if (myRank == 0)
        {
            std::cout << " --> Iter post-shock states by Mach number: " << Ma << std::endl;
        }
        /*ini upstream and downstream*/ //
        e2 = MARCO_Coph(T2) - R * T2;
        u2 = ini.blast_u_out;
        E2 = e2 + _DF(0.5) * (u2 * u2 + ini.blast_v_out * ini.blast_v_out + ini.blast_w_out * ini.blast_w_out);

        Si = Ma * c2;
        // give prediction value
        p1 = Ma * p2;
        rho1 = Ma * ini.blast_density_out;
        T1 = T2;

        real_t residual, threshold = 1.0e-6; // residual: 实际误差; threshold: 误差控制
        int iter = 0;
        do
        {
            if (iter != 0)
            {
                real_t delta_rho = _DF(1.0e-6) * rho1;
                rho1 += delta_rho;
                u1 = rho2 * (u2 - Si) / rho1 + Si;
                p1 = rho2 * (u2 - Si) * u2 + p2 - rho1 * (u1 - Si) * u1;

                T1 = p1 / rho1 / R;
                e1 = MARCO_Coph(T1) - p1 / rho1;
                E1 = e1 + _DF(0.5) * (u1 * u1 + ini.blast_v_in * ini.blast_v_in + ini.blast_w_in * ini.blast_w_in);

                real_t residual_new = rho2 * (u2 - Si) * E2 - rho1 * (u1 - Si) * E1 + p2 * u2 - p1 * u1;
                real_t dfdrho = (residual_new - residual) / delta_rho;
                rho1 -= delta_rho;
                rho1 = rho1 - residual / dfdrho;
            }
            if (iter > 1000)
            {
                if (myRank == 0)
                {
                    std::cout << "   Mach number Iteration failed: Over 1000 steps has been done." << std::endl;
                }
                exit(EXIT_FAILURE);
            }

            u1 = rho2 * (u2 - Si) / rho1 + Si;
            p1 = rho2 * (u2 - Si) * u2 + p2 - rho1 * (u1 - Si) * u1;

            T1 = p1 / rho1 / R;
            e1 = MARCO_Coph(T1) - p1 / rho1;
            E1 = e1 + _DF(0.5) * (u1 * u1 + ini.blast_v_in * ini.blast_v_in + ini.blast_w_in * ini.blast_w_in);

            residual = rho2 * (u2 - Si) * E2 - rho1 * (u1 - Si) * E1 + p2 * u2 - p1 * u1;
            iter++;
#ifdef DEBUG
            if (myRank == 0)
            {
                std::cout << "   The " << iter << "th iterations, residual : " << residual << std::endl;
            }
#endif // end DEBUG
        } while (fabs(residual) > threshold);
    }

    // Ref0: https://doi.org/10.1016/j.combustflame.2022.112085 theroy
    if (Mach_Modified)
    {
        if (myRank == 0)
        {
            std::cout << "\n--> Modified the shock's status by Ref0:https://doi.org/10.1016/j.combustflame.2022.112085" << std::endl;
        }

        rho1 = rho2 * (Gamma_m2 + _DF(1.0)) * Ma_1 / (_DF(2.0) + (Gamma_m2 - _DF(1.0)) * Ma_1);
        p1 = p2 * (_DF(1.0) + _DF(2.0) * Gamma_m2 * (Ma_1 - _DF(1.0)) / (Gamma_m2 + _DF(1.0)));
        u1 = ini.Ma * c2 * (_DF(1.0) - rho2 / rho1);
    }

    ini.blast_density_in = rho1;
    ini.blast_pressure_in = p1;
    ini.blast_T_in = ini.blast_pressure_in / R / ini.blast_density_in; // downstream : upstream states of the shock p2/rho2 : p1/rho1=T2 : T1
    ini.blast_u_in = u1;

    return true;
}

// =======================================================
// =======================================================
/**
 * @brief get coefficients of polynominal fitted
 */
void Setup::GetFitCoefficient()
{
    ReadOmega_table(); // read Omega_table here for fitting
    real_t *Dkj_matrix = NULL, *fitted_coefficients_visc = NULL, *fitted_coefficients_therm = NULL;
    Dkj_matrix = middle::MallocHost<real_t>(Dkj_matrix, NUM_SPECIES * NUM_SPECIES * order_polynominal_fitted, q);
    h_thermal.Dkj_matrix = middle::MallocHost2D<real_t>(Dkj_matrix, NUM_SPECIES * NUM_SPECIES, order_polynominal_fitted, q);
    fitted_coefficients_visc = middle::MallocHost<real_t>(fitted_coefficients_visc, NUM_SPECIES * order_polynominal_fitted, q);
    h_thermal.fitted_coefficients_visc = middle::MallocHost2D<real_t>(fitted_coefficients_visc, NUM_SPECIES, order_polynominal_fitted, q);
    fitted_coefficients_therm = middle::MallocHost<real_t>(fitted_coefficients_therm, NUM_SPECIES * order_polynominal_fitted, q);
    h_thermal.fitted_coefficients_therm = middle::MallocHost2D<real_t>(fitted_coefficients_therm, NUM_SPECIES, order_polynominal_fitted, q);

    for (int k = 0; k < NUM_SPECIES; k++)
    { // Allocate Mem
        // h_thermal.fitted_coefficients_visc[k] = middle::MallocHost<real_t>(h_thermal.fitted_coefficients_visc[k], order_polynominal_fitted, q);
        // h_thermal.fitted_coefficients_therm[k] = middle::MallocHost<real_t>(h_thermal.fitted_coefficients_therm[k], order_polynominal_fitted, q);

        real_t *specie_k = &(h_thermal.species_chara[k * SPCH_Sz]);
        Fitting(Tnode, specie_k, specie_k, h_thermal.fitted_coefficients_visc[k], 0);  // Visc
        Fitting(Tnode, specie_k, specie_k, h_thermal.fitted_coefficients_therm[k], 1); // diffu
        for (int j = 0; j < NUM_SPECIES; j++)
        { // Allocate Mem
            // h_thermal.Dkj_matrix[k * NUM_SPECIES + j] = middle::MallocHost<real_t>(h_thermal.Dkj_matrix[k * NUM_SPECIES + j], order_polynominal_fitted, q);

            real_t *specie_j = &(h_thermal.species_chara[j * SPCH_Sz]);
            if (k <= j)                                                                           // upper triangle
                Fitting(Tnode, specie_k, specie_j, h_thermal.Dkj_matrix[k * NUM_SPECIES + j], 2); // Dim
            else
            { // lower triangle==>copy
                for (int n = 0; n < order_polynominal_fitted; n++)
                    h_thermal.Dkj_matrix[k * NUM_SPECIES + j][n] = h_thermal.Dkj_matrix[j * NUM_SPECIES + k][n];
            }
        }
    }

    real_t *d_Dkj_matrix, *d_fitted_coefficients_visc, *d_fitted_coefficients_therm;
    d_Dkj_matrix = middle::MallocDevice<real_t>(d_Dkj_matrix, NUM_SPECIES * NUM_SPECIES * order_polynominal_fitted, q);
    d_fitted_coefficients_visc = middle::MallocDevice<real_t>(d_fitted_coefficients_visc, NUM_SPECIES * order_polynominal_fitted, q);
    d_fitted_coefficients_therm = middle::MallocDevice<real_t>(d_fitted_coefficients_therm, NUM_SPECIES * order_polynominal_fitted, q);
    middle::MemCpy<real_t>(d_Dkj_matrix, Dkj_matrix, NUM_SPECIES * NUM_SPECIES * order_polynominal_fitted, q);
    middle::MemCpy<real_t>(d_fitted_coefficients_visc, fitted_coefficients_visc, NUM_SPECIES * order_polynominal_fitted, q);
    middle::MemCpy<real_t>(d_fitted_coefficients_therm, fitted_coefficients_therm, NUM_SPECIES * order_polynominal_fitted, q);
    d_thermal.Dkj_matrix = middle::MallocDevice2D<real_t>(d_Dkj_matrix, NUM_SPECIES * NUM_SPECIES, order_polynominal_fitted, q);
    d_thermal.fitted_coefficients_visc = middle::MallocDevice2D<real_t>(d_fitted_coefficients_visc, NUM_SPECIES, order_polynominal_fitted, q);
    d_thermal.fitted_coefficients_therm = middle::MallocDevice2D<real_t>(d_fitted_coefficients_therm, NUM_SPECIES, order_polynominal_fitted, q);

    // Test
    if (ViscosityTest_json)
        VisCoeffsAccuracyTest(ViscosityTestRange[0], ViscosityTestRange[1]);
}

/**
 * @brief get accurate three kind of viscosity coefficients
 * @param Tmin beginning temperature point of the coefficient-Temperature plot
 * @param Tmax Ending temperature point of the coefficient-Temperature plot
 * @note  /delta T is devided by space discrete step
 */
void Setup::VisCoeffsAccuracyTest(real_t Tmin, real_t Tmax)
{
    size_t reso = std::max(200, BlSz.X_inner);
    // // out Thermal
    std::string file_name = OutputDir + "/viscosity-thermal.dat";
    std::ofstream theo(file_name);
    theo << "variables= Temperature(K)";
    for (size_t k = 0; k < species_name.size(); k++)
    {
        theo << "," << species_name[k] << "_Cp_NASA";
        theo << "," << species_name[k] << "_Cp_JANAF";
    }
    for (size_t k = 0; k < species_name.size(); k++)
    {
        theo << "," << species_name[k] << "_hi_NASA";
        theo << "," << species_name[k] << "_hi_JANAF";
    }
    for (size_t k = 0; k < species_name.size(); k++)
    {
        theo << "," << species_name[k] << "_S_NASA";
        theo << "," << species_name[k] << "_S_JANAF";
    }
    for (size_t i = 1; i <= reso; i++)
    {
        real_t Tpoint = Tmin + (i / real_t(reso)) * (Tmax - Tmin);
        theo << Tpoint << " "; // Visc
        for (size_t k = 0; k < species_name.size(); k++)
        {
            theo << HeatCapacity_NASA(h_thermal.Hia_NASA, Tpoint, h_thermal.Ri[k], k) << " ";   // Cp
            theo << HeatCapacity_JANAF(h_thermal.Hia_JANAF, Tpoint, h_thermal.Ri[k], k) << " "; // Cp
        }
        for (size_t k = 0; k < species_name.size(); k++)
        {
            theo << get_Enthalpy_NASA(h_thermal.Hia_NASA, h_thermal.Hib_NASA, Tpoint, h_thermal.Ri[k], k) << " ";    // hi
            theo << get_Enthalpy_JANAF(h_thermal.Hia_JANAF, h_thermal.Hib_JANAF, Tpoint, h_thermal.Ri[k], k) << " "; // hi
        }
        for (size_t k = 0; k < species_name.size(); k++)
        {
            theo << get_Entropy_NASA(h_thermal.Hia_NASA, h_thermal.Hib_NASA, Tpoint, h_thermal.Ri[k], k) << " ";    // S
            theo << get_Entropy_JANAF(h_thermal.Hia_JANAF, h_thermal.Hib_JANAF, Tpoint, h_thermal.Ri[k], k) << " "; // S
        }
        theo << "\n";
    }
    theo.close();

    // // visc coefficients
    file_name = OutputDir + "/viscosity-test";
#if Thermo
    file_name += "-(NASA9).dat";
#else
    file_name += "-(JANAF).dat";
#endif
    std::ofstream out(file_name);
    out << "variables= Temperature(K)";
    for (size_t k = 0; k < species_name.size(); k++)
    {
        out << ",visc_" << species_name[k];
        out << ",furier_" << species_name[k];
        for (size_t j = 0; j <= k; j++)
            out << ",Dkj_" << species_name[k] << "-" << species_name[j];
    }
    // zone name
    out << "\nzone t='Accurate-solution'\n";
    for (size_t i = 1; i <= reso; i++)
    {
        real_t Tpoint = Tmin + (i / real_t(reso)) * (Tmax - Tmin);
        out << Tpoint << " "; // Visc
        for (size_t k = 0; k < species_name.size(); k++)
        {
            real_t *specie_k = &(h_thermal.species_chara[k * SPCH_Sz]);
            out << viscosity(specie_k, Tpoint) << " ";                   // Visc
            out << thermal_conductivities(specie_k, Tpoint, 1.0) << " "; // diffu
            for (size_t j = 0; j <= k; j++)
            {
                real_t *specie_j = &(h_thermal.species_chara[j * SPCH_Sz]);
                out << Dkj(specie_k, specie_j, Tpoint, 1.0) << " "; // Dkj
            }
        }
        out << "\n";
    }

    out << "\nzone t='Fitting-solution'\n";
    for (size_t i = 1; i <= reso; i++)
    {
        real_t **Dkj = h_thermal.Dkj_matrix;
        real_t **fcv = h_thermal.fitted_coefficients_visc;
        real_t **fct = h_thermal.fitted_coefficients_therm;

        real_t Tpoint = Tmin + (i / real_t(reso)) * (Tmax - Tmin);
        out << Tpoint << " "; // Visc
        for (size_t k = 0; k < species_name.size(); k++)
        {
            real_t *specie_k = &(h_thermal.species_chara[k * SPCH_Sz]);
            out << Viscosity(fcv[int(specie_k[SID])], Tpoint) << " ";            // Visc
            out << Thermal_conductivity(fct[int(specie_k[SID])], Tpoint) << " "; // diffu
            for (size_t j = 0; j <= k; j++)
            {
                real_t *specie_j = &(h_thermal.species_chara[j * SPCH_Sz]);
                out << GetDkj(specie_k, specie_j, Dkj, Tpoint, 1.0) << " "; // Dkj
            }
        }
        out << "\n";
    }
    out.close();
}

/**
 * @brief fitting procedure for transport coefficients
 * @para specie_k,the fitting is for specie_k
 * @para specie_j,if fitting is for binarry diffusion coefficient,specie_j is another specie; otherwise, it is set as the same with specie_k
 * @para aa the coefficients of the polynominal;
 * @para indicator fitting for viscosity(0),thermal conductivities(1) and binary diffusion coefficients(2)
 */
void Setup::Fitting(std::vector<real_t> TT, real_t *specie_k, real_t *specie_j, real_t *aa, int indicator)
{
    int mm = TT.size();
    real_t b[mm], AA[mm][order_polynominal_fitted];
    for (int ii = 0; ii < mm; ii++)
    {
        switch (indicator)
        {
        case 0:
            b[ii] = std::log(viscosity(specie_k, TT[ii])); // get RHS of the overdetermined equations
            break;
        case 1:
            b[ii] = std::log(thermal_conductivities(specie_k, TT[ii], 1.0));
            break;
        case 2:
            b[ii] = std::log(Dkj(specie_k, specie_j, TT[ii], 1.0));
            break;
        }
        // b[ii] = std::log(viscosity(specie_k, TT[ii])); // get RHS column vector of the overdetermined systems of linear equations
        for (int jj = 0; jj < order_polynominal_fitted; jj++)
            AA[ii][jj] = std::pow(std::log(TT[ii]), jj);
    }
    Solve_Overdeter_equations(AA, b, mm, aa);
}

// =======================================================
// =======================================================
/**
 * @brief read collision integral table from "collision_integral.dat"
 */
void Setup::ReadOmega_table()
{
    std::string fpath = WorkDir + std::string(RPath) + "/collision_integral.dat";
    std::ifstream fin(fpath);
    for (int n = 0; n < 8; n++)
        fin >> delta_star[n]; // reduced dipole moment;
    for (int i = 0; i < 37; i++)
    {
        fin >> T_star[i]; // reduced temperature;
        for (int j = 0; j < 8; j++)
            fin >> Omega_table[1][i][j]; // collision integral for binary diffusion coefficient;
    }
    for (int p = 0; p < 37; p++)
        for (int q = 0; q < 8; q++)
            fin >> Omega_table[0][p][q]; // collision integral for viscosity and thermal conductivity;
    fin.close();
}

/**
 * @brief get Omega interpolated
 * @para T_star reduced temperature;
 * @para delta reduced dipole moment;
 * @para index:0(1):look up table 0(1);
 */
real_t Setup::Omega_interpolated(real_t Tstar, real_t deltastar, int index)
{
    int ti1, ti2, ti3;
    if (Tstar > T_star[0] && Tstar < T_star[36])
    {
        int ii = 1;
        {
            while (Tstar > T_star[ii])
                ii = ii + 1;
        }
        ti1 = ii - 1;
        ti2 = ii;
        ti3 = ii + 1;
    }
    else if (Tstar <= T_star[0])
    {
        ti1 = 0;
        ti2 = 1;
        ti3 = 2;
    }
    else if (Tstar >= T_star[36])
    {
        ti1 = 34;
        ti2 = 35;
        ti3 = 36;
    }
    int tj1, tj2, tj3;
    if (deltastar > delta_star[0] && deltastar < delta_star[7])
    {
        int jj = 1;
        {
            while (deltastar > delta_star[jj])
                jj = jj + 1;
        }
        tj1 = jj - 1;
        tj2 = jj;
        tj3 = jj + 1;
    }
    else if (deltastar <= delta_star[0])
    {
        tj1 = 0;
        tj2 = 1;
        tj3 = 2;
    }
    else if (deltastar >= delta_star[7])
    {
        tj1 = 5;
        tj2 = 6;
        tj3 = 7;
    }
    real_t aa[3];

    GetQuadraticInterCoeff(T_star[ti1], T_star[ti2], T_star[ti3],
                           Omega_table[index][ti1][tj1], Omega_table[index][ti2][tj1], Omega_table[index][ti3][tj1], aa);
    real_t temp1 = aa[0] + aa[1] * Tstar + aa[2] * Tstar * Tstar;
    GetQuadraticInterCoeff(T_star[ti1], T_star[ti2], T_star[ti3],
                           Omega_table[index][ti1][tj2], Omega_table[index][ti2][tj2], Omega_table[index][ti3][tj2], aa);
    real_t temp2 = aa[0] + aa[1] * Tstar + aa[2] * Tstar * Tstar;
    GetQuadraticInterCoeff(T_star[ti1], T_star[ti2], T_star[ti3],
                           Omega_table[index][ti1][tj3], Omega_table[index][ti2][tj3], Omega_table[index][ti3][tj3], aa);
    real_t temp3 = aa[0] + aa[1] * Tstar + aa[2] * Tstar * Tstar;

    GetQuadraticInterCoeff(delta_star[tj1], delta_star[tj2], delta_star[tj3], temp1, temp2, temp3, aa);

    return aa[0] + aa[1] * deltastar + aa[2] * deltastar * deltastar;
}

/**
 * @brief get molecular viscosity at T temperature for specie
 * @para T temperature
 * @para specie
 */
real_t Setup::viscosity(real_t *specie, const real_t T)
{
    real_t Tstar = T / specie[epsilon_kB];
    real_t deltastar = 0.5 * specie[mue] * specie[mue] / specie[epsilon_kB] / kB / (std::pow(specie[d], 3)) * 1.0e-12;        // equation 5-2
    real_t Omega2 = Omega_interpolated(Tstar, deltastar, 0);                                                                  // real_t Omega2 =  Omega2_interpolated(Tstar);
    real_t visc = 5 * 1.0e16 * std::sqrt(pi * (specie[Wi] * 1e3) / NA * kB * T) / (16 * pi * specie[d] * specie[d] * Omega2); // equation 5-1,unit: g/(cm.s)
    return visc = 0.1 * visc;                                                                                                 // unit: Pa.s=kg/(m.s)
}

/**
 * @brief get thermal conductivities at T temperature
 * @para T temperature
 * @para PP
 * unit:SI
   p:pa=kg/(m.s2) T:K  visc: pa.s=kg/(m.s) thermal conductivity:W/(m.K)
 */
real_t Setup::thermal_conductivities(real_t *specie, const real_t T, const real_t PP)
{
    real_t Cv_trans = 1.5 * universal_gas_const, Cv_rot, Cv_vib;
    int id = int(specie[SID]);
    real_t Cpi = HeatCapacity(h_thermal.Hia, T, h_thermal.Ri[id], id);
    real_t Cv = Cpi * specie[Wi] * 1.0e3 - universal_gas_const; // unit:J/(kmol.K)
    switch (int(specie[geo]))
    {
    case 0:
    {
        Cv_rot = 0.0;
        Cv_vib = 0.0;
        break;
    }
    case 1:
    {
        Cv_rot = 1.0 * universal_gas_const; // unit:J/(kmol*K)
        Cv_vib = Cv - 2.5 * universal_gas_const;
        break;
    }
    case 2:
    {
        Cv_rot = 1.5 * universal_gas_const;
        Cv_vib = Cv - 3.0 * universal_gas_const;
        break;
    }
    }
    real_t rho = PP * specie[Wi] / T / universal_gas_const;          // unit:g/cm3 equation5-32
    real_t Dkk = Dkj(specie, specie, T, PP);                         // unit:cm*cm/s
    real_t visc = viscosity(specie, T);                              // unit: Pa.s=kg/(m.s)
    real_t f_trans, f_rot, f_vib = rho * Dkk / (visc * 10.0);        // unit:1

    real_t Zrot = specie[Zrot_298] * ZrotFunc(specie[epsilon_kB] / 298.0) / ZrotFunc(specie[epsilon_kB] / T); // unit:1
    real_t Aa = 2.5 - f_vib, Bb = Zrot + 2.0 * (5.0 * Cv_rot / 3.0 / (universal_gas_const) + f_vib) / pi;

    f_trans = 2.5 * (1.0 - 2.0 * Cv_rot * Aa / pi / Cv_trans / Bb);
    f_rot = f_vib * (1.0 + 2.0 * Aa / pi / Bb);
    real_t temp = visc * (f_trans * Cv_trans + f_rot * Cv_rot + f_vib * Cv_vib) / specie[Wi] * _DF(1.0e-3); // unit:W/(m.K)
    return temp;
}

/**
 * @brief get binary(specie j&specie k) diffusion coefficient at T temperature per pressure
 * @para T temperature
 * @para specie
 * unit: 1 pa=1 kg/(m.s2)=10 g/(cm.s2)
   [Wi]=kg/mol;   [T]=K;  [PP]=pa;   [Djk]=cm2/s
 */
real_t Setup::Dkj(real_t *specie_k, real_t *specie_j, const real_t T, const real_t PP) // PP:pressure,unit:Pa
{
    real_t epsilon_jk_kB, d_jk, mue_jk_sqr;
    // either both nonpolar or both polar
    if ((specie_j[mue] > 0 && specie_k[mue] > 0) || (specie_j[mue] == 0 && specie_k[mue] == 0))
    {
        epsilon_jk_kB = std::sqrt(specie_j[epsilon_kB] * specie_k[epsilon_kB]); // unit:K,equation5-6
        d_jk = (specie_j[d] + specie_k[d]) / 2.0;
        mue_jk_sqr = specie_j[mue] * specie_k[mue];
    }
    // polar molecule interacting with a nonpolar molecule
    else
    {
        real_t epsilon_n_kB, epsilon_p_kB, alpha_n, mue_p, d_n, d_p; // equation 5-9~~5-14
        if (specie_k[mue] > 0 && specie_j[mue] == 0)
        {
            epsilon_n_kB = specie_j[epsilon_kB];
            epsilon_p_kB = specie_k[epsilon_kB];
            alpha_n = specie_j[alpha];
            d_n = specie_j[d];
            d_p = specie_k[d];
            mue_p = specie_k[mue];
        }
        if (specie_j[mue] > 0 && specie_k[mue] == 0)
        {
            epsilon_n_kB = specie_k[epsilon_kB];
            epsilon_p_kB = specie_j[epsilon_kB];
            alpha_n = specie_k[alpha];
            d_n = specie_k[d];
            d_p = specie_j[d];
            mue_p = specie_j[mue];
        }
        real_t alpha_n_star = alpha_n / std::pow(d_n, _DF(3.0));                                                   // equation5-13
        real_t mue_p_star = mue_p / std::pow(epsilon_p_kB * kB, _DF(0.5)) / std::pow(d_p, _DF(1.5)) * _DF(1.0e-6); // equation5-14
        real_t ksi = _DF(1.0) + _DF(0.25) * alpha_n_star * mue_p_star * std::sqrt(epsilon_p_kB / epsilon_n_kB);    // equation5-12

        epsilon_jk_kB = ksi * ksi * std::sqrt(epsilon_n_kB * epsilon_p_kB); // equation5-9
        d_jk = std::pow(ksi, -_DF(1.0) / _DF(6.0)) * (specie_j[d] + specie_k[d]) / _DF(2.0);
        mue_jk_sqr = _DF(0.0);
    }
    real_t T_jk_star = T / epsilon_jk_kB;                                                                                                               // equation5-15
    real_t delta_jk_star = _DF(0.5) * mue_jk_sqr / d_jk / d_jk / d_jk / epsilon_jk_kB / kB * _DF(1.0e-12);                                              // equation5-16
    real_t W_jk = specie_k[Wi] * specie_j[Wi] / (specie_k[Wi] + specie_j[Wi]) / NA * _DF(1.0e3);                                                        // unit,g;equation5-5
    real_t Omega1 = Omega_interpolated(T_jk_star, delta_jk_star, 1);                                                                                    // real_t Omega1 = Omega1_interpolated(T_jk_star);
    real_t PPP = PP * _DF(10.0);                                                                                                                        // pa==>g/(cm.s2)
    real_t Dkj = _DF(3.0) * std::sqrt(_DF(2.0) * pi * std::pow(T * kB, _DF(3.0)) / W_jk) / (_DF(16.0) * PPP * pi * d_jk * d_jk * Omega1) * _DF(1.0e16); // equation5-4 //unit:cm*cm/s
    return Dkj;
}
