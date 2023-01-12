/****************************************
 * 目标是全部使用setup.ini文件控制流体参数，初始化状态，MPI设置，输出设置等尽可能多的设置
 */
#include <iomanip>
#include "global_setup_function.hpp"

Setup::Setup(ConfigMap &configMap, middle::device_t &Q) : q(Q)
{
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    if (myRank == 0)
#endif
    {
        std::cout << "<---------------------------------------------------> \n";
    }
    ReadIni(configMap);

    /*begin runtime read , fluid && compoent characteristics set*/
    ReadSpecies(); // 化学反应的组分数太多不能直接放进.ini 文件，等以后实现在ini中读取数组
#ifdef COP_CHEME
    ReadReactions();
#endif // end COP_CHEME
    /*end runtime read*/

#ifdef Visc // read && caculate coffes for visicity
    GetFitCoefficient();
#endif

    init(); // Ini
#ifdef USE_MPI
    mpiTrans = new MpiTrans(BlSz, Boundarys);
    if (0 == mpiTrans->myRank)
        print();
    mpiTrans->communicator->synchronize();
#else
    print();
#endif // end USE_MPI

    std::cout << "Selected Device: " << middle::DevInfo(q);
#ifdef USE_MPI
    std::cout << "  of rank: " << mpiTrans->myRank;
#endif

    std::cout << std::endl;
    //<< q.get_device().get_info<sycl::info::device::name>() << ", version = "<< q.get_device().get_info<sycl::info::device::version>() << "\n";

#ifdef USE_MPI
    mpiTrans->communicator->synchronize();
#endif // end USE_MPI

    CpyToGPU();
} // Setup::Setup end
// =======================================================
// =======================================================
void Setup::ReadSpecies()
{                                                                                                          // compoent in or out bubble differs
    h_thermal.species_ratio_in = middle::MallocHost<real_t>(h_thermal.species_ratio_in, NUM_SPECIES, q);   // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * sizeof(real_t), q));  // real_t species_ratio[NUM_SPECIES]
    h_thermal.species_ratio_out = middle::MallocHost<real_t>(h_thermal.species_ratio_out, NUM_SPECIES, q); // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * sizeof(real_t), q)); // real_t species_ratio[NUM_SPECIES]

    std::string path = std::string(RFile) + "/species_list.dat";
    std::fstream fins(path);
    for (int n = 0; n < NUM_SPECIES; n++)
        fins >> species_name[n]; // name of the species
#ifdef COP
    for (int n = 0; n < NUM_SPECIES; n++) // molar ratio
        fins >> h_thermal.species_ratio_out[n];
    for (int n = 0; n < NUM_SPECIES; n++)      // species_ratio in bubble if exsit
        fins >> h_thermal.species_ratio_in[n]; // molar ratio
    fins.close();
#endif // end COP
    ReadThermal();
}
// =======================================================
// =======================================================
void Setup::ReadThermal()
{
    h_thermal.species_chara = middle::MallocHost<real_t>(h_thermal.species_chara, NUM_SPECIES * SPCH_Sz, q); // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * SPCH_Sz * sizeof(real_t), q)); // new real_t[NUM_SPECIES * SPCH_Sz];
    h_thermal.Ri = middle::MallocHost<real_t>(h_thermal.Ri, NUM_SPECIES, q);                                 // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * sizeof(real_t), q));                      // new real_t[NUM_SPECIES];
    h_thermal.Wi = middle::MallocHost<real_t>(h_thermal.Wi, NUM_SPECIES, q);                                 // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * sizeof(real_t), q));                      // new real_t[NUM_SPECIES];
    h_thermal._Wi = middle::MallocHost<real_t>(h_thermal._Wi, NUM_SPECIES, q);                               // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * sizeof(real_t), q));                      // new real_t[NUM_SPECIES];
    h_thermal.Hia = middle::MallocHost<real_t>(h_thermal.Hia, NUM_SPECIES * 7 * 3, q);                       // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * 7 * 3 * sizeof(real_t), q));             // Hia = new real_t[NUM_SPECIES * 7 * 3];
    h_thermal.Hib = middle::MallocHost<real_t>(h_thermal.Hib, NUM_SPECIES * 2 * 3, q);                       // static_cast<real_t *>(sycl::malloc_host(NUM_SPECIES * 2 * 3 * sizeof(real_t), q));             // Hib = new real_t[NUM_SPECIES * 2 * 3];

    char Key_word[128];
#if Thermo
    std::string apath = std::string(RPath) + "/thermal_dynamics.dat";
#else
    std::string apath = std::string(RPath) + "/thermal_dynamics_janaf.dat";
#endif
    std::fstream finc(apath);
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        // check the name of the species "n"
        std::string my_string = "*" + species_name[n];
        char *species_name_n = new char[my_string.size() + 1];
        std::strcpy(species_name_n, my_string.c_str());
        // reset file point location
        finc.seekg(0);
        while (!finc.eof())
        {
            finc >> Key_word;
            if (!std::strcmp(Key_word, "*END"))
                break;
            if (!std::strcmp(Key_word, species_name_n))
            {
#if Thermo
                // low temperature parameters, 200K<T<1000K
                for (int m = 0; m < 7; m++)
                {
                    finc >> h_thermal.Hia[n * 7 * 3 + m * 3 + 0]; // a1-a7
                    // std::cout << Hia[n * 7 * 3 + m * 3 + 0] << std::endl;
                }
                for (int m = 0; m < 2; m++)
                {
                    finc >> h_thermal.Hib[n * 2 * 3 + m * 3 + 0]; // b1,b2
                    // std::cout << Hib[n * 2 * 3 + m * 3 + 0] << std::endl;
                }
                // high temperature parameters, 1000K<T<6000K
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia[n * 7 * 3 + m * 3 + 1]; // a1-a7
                for (int m = 0; m < 2; m++)
                    finc >> h_thermal.Hib[n * 2 * 3 + m * 3 + 1]; // b1,b2
                // high temperature parameters, 6000K<T<15000K
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia[n * 7 * 3 + m * 3 + 2]; // a1-a7
                for (int m = 0; m < 2; m++)
                    finc >> h_thermal.Hib[n * 2 * 3 + m * 3 + 2]; // b1,b2
#else
                // high temperature parameters
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia[n * 7 * 3 + m * 3 + 0]; // a1-a7
                // low temperature parameters
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia[n * 7 * 3 + m * 3 + 1]; // a1-a7
#endif
                finc >> h_thermal.species_chara[n * SPCH_Sz + Wi]; // species[n].Wi; // molar mass, unit: g/mol
                h_thermal.species_chara[n * SPCH_Sz + 6] *= 1e-3;  // kg/mol
                h_thermal.Wi[n] = h_thermal.species_chara[n * SPCH_Sz + Wi];
                h_thermal._Wi[n] = _DF(1.0) / h_thermal.Wi[n];
                h_thermal.Ri[n] = Ru / h_thermal.Wi[n];
                h_thermal.species_chara[n * SPCH_Sz + 7] = 0;   // Miu[i];
                h_thermal.species_chara[n * SPCH_Sz + SID] = n; // SID;
                break;
            }
        }
    }
    finc.close();

    std::string spath = std::string(RPath) + "/transport_data.dat";
    std::fstream fint(spath);
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        // check the name of the species "n"
        std::string my_string = species_name[i];
        char *species_name_n = new char[my_string.size()];
        strcpy(species_name_n, my_string.c_str());
        // reset file point location
        fint.seekg(0);
        while (!fint.eof())
        {
            fint >> Key_word;
            if (!strcmp(Key_word, "*END"))
                break;
            if (!strcmp(Key_word, species_name_n))
            {
                fint >> h_thermal.species_chara[i * SPCH_Sz + 0]; // int geo;//0:monoatom,1:nonpolar(linear) molecule,2:polar molecule//极性
                fint >> h_thermal.species_chara[i * SPCH_Sz + 1]; // double epsilon_kB;//epsilon: Lennard-Jones potential well depth;unit:K//势井深度
                fint >> h_thermal.species_chara[i * SPCH_Sz + 2]; // d;//Lennard-Jones collision diameter, unit: angstroms,10e-10m//碰撞直径in 4-3;
                fint >> h_thermal.species_chara[i * SPCH_Sz + 3]; // mue;//dipole moment,unit:Debye(m);//偶极距
                fint >> h_thermal.species_chara[i * SPCH_Sz + 4]; // alpha;//polarizability;unit:cubic angstrom//极化率
                fint >> h_thermal.species_chara[i * SPCH_Sz + 5]; // Zrot_298;//rotational relaxation collision Zrot at 298K;
                break;
            }
        }
    }
    fint.close();

    h_thermal.xi_in = middle::MallocHost<real_t>(h_thermal.xi_in, NUM_SPECIES, q);
    h_thermal.xi_out = middle::MallocHost<real_t>(h_thermal.xi_out, NUM_SPECIES, q);
    middle::MemCpy<real_t>(h_thermal.xi_in, h_thermal.species_ratio_in, NUM_SPECIES, q);
    middle::MemCpy<real_t>(h_thermal.xi_out, h_thermal.species_ratio_out, NUM_SPECIES, q);

    // transfer mole fraction to mess fraction
    get_Yi(h_thermal.species_ratio_out);
    get_Yi(h_thermal.species_ratio_in);
    mach_shock = Mach_Shock();
}
// =======================================================
// =======================================================
void Setup::get_Yi(real_t *yi)
{ // yi是体积分数，Yi=rhos/rho是质量分数
    real_t W_mix = _DF(0.0);
    for (size_t i = 0; i < NUM_SPECIES; i++)
        W_mix += yi[i] * h_thermal.Wi[i];
    for (size_t n = 0; n < NUM_SPECIES; n++) // Ri=Ru/Wi
        yi[n] = yi[n] * h_thermal.Wi[n] / W_mix;
}
// =======================================================
// =======================================================
bool Setup::Mach_Shock()
{ // called after Setup::get_Yi();, only support X dir shock
    real_t Ma_1 = ini.Ma * ini.Ma;
    if (Ma_1 < 1.0)
    {
#ifdef USE_MPI
        if (myRank == 0)
#endif
        {
            std::cout << "   Mach number < 1, shock is not initialized by it." << std::endl;
        }
        return false;
    } //

    real_t p2 = ini.blast_pressure_out; // P2
    real_t T2 = ini.blast_T_out;        // T2

    real_t R, Gamma_m2;

    // #ifdef DEBUG
    //     R = 1399.0 * 0.33;
    //     Gamma_m2 = 1.33;
    // #else
    R = get_MixtureR(h_thermal.species_chara, h_thermal.species_ratio_out);
    Gamma_m2 = get_CopGamma(h_thermal.species_ratio_out, T2);
    // #endif                                        // end DEBUG
    real_t c2 = std::sqrt(Gamma_m2 * R * T2); // sound speed downstream the shock
    ini.blast_density_out = p2 / R / T2;      // rho2
    real_t rho2 = ini.blast_density_out;      // rho2

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
#define MARCO_Coph(T) get_Coph(h_thermal.species_ratio_out, T)
#endif // end DEBUG

    if (!Mach_Modified)
    {
#ifdef USE_MPI
        if (myRank == 0)
#endif
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

                double residual_new = rho2 * (u2 - Si) * E2 - rho1 * (u1 - Si) * E1 + p2 * u2 - p1 * u1;
                double dfdrho = (residual_new - residual) / delta_rho;
                rho1 -= delta_rho;
                rho1 = rho1 - residual / dfdrho;
            }
            if (iter > 1000)
            {
#ifdef USE_MPI
                if (myRank == 0)
#endif
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
#ifdef USE_MPI
            if (myRank == 0)
#endif
            {
                std::cout << "   The " << iter << "th iterations, residual : " << residual << std::endl;
            }
#endif // end DEBUG
        } while (fabs(residual) > threshold);
    }

    // Ref0: https://doi.org/10.1016/j.combustflame.2022.112085 theroy
    if (Mach_Modified)
    {
#ifdef USE_MPI
        if (myRank == 0)
#endif
        {
            std::cout << "--> Modified the shock's status by Ref0:https://doi.org/10.1016/j.combustflame.2022.112085" << std::endl;
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
#ifdef COP_CHEME
void Setup::ReadReactions()
{
    h_react.Nu_f_ = middle::MallocHost<int>(h_react.Nu_f_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    h_react.Nu_b_ = middle::MallocHost<int>(h_react.Nu_b_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    h_react.Nu_d_ = middle::MallocHost<int>(h_react.Nu_d_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    h_react.React_ThirdCoef = middle::MallocHost<real_t>(h_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q); // static_cast<real_t *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(real_t), q));
    h_react.Rargus = middle::MallocHost<real_t>(h_react.Rargus, NUM_REA * 6, q);                             // static_cast<real_t *>(sycl::malloc_host(NUM_REA * 6 * sizeof(real_t), q));
    h_react.react_type = middle::MallocHost<int>(h_react.react_type, NUM_REA * 2, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * 2 * sizeof(int), q));
    h_react.third_ind = middle::MallocHost<int>(h_react.third_ind, NUM_REA, q);                              // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));

    char Key_word[128];
    std::string rpath = std::string(RFile) + "/reaction_list.dat";
    std::ifstream fint(rpath);
    {
        fint.seekg(0);
        while (!fint.eof())
        {
            fint >> Key_word;
            if (!std::strcmp(Key_word, "*forward_reaction"))
            { // stoichiometric coefficients of each reactants in all forward reactions
                for (int i = 0; i < NUM_REA; i++)
                    for (int j = 0; j < NUM_SPECIES; j++)
                        fint >> h_react.Nu_f_[i * NUM_SPECIES + j];
            }
            else if (!std::strcmp(Key_word, "*backward_reaction"))
            { // stoichiometric coefficients of each reactants in all backward reactions
                // Nu_b_ is also the coefficients of products of previous forward reactions
                for (int i = 0; i < NUM_REA; i++)
                    for (int j = 0; j < NUM_SPECIES; j++)
                    {
                        fint >> h_react.Nu_b_[i * NUM_SPECIES + j]; // the net stoichiometric coefficients
                        h_react.Nu_d_[i * NUM_SPECIES + j] = h_react.Nu_b_[i * NUM_SPECIES + j] - h_react.Nu_f_[i * NUM_SPECIES + j];
                    }
            }
            else if (!std::strcmp(Key_word, "*third_body"))
            { // the third body coefficients
                for (int i = 0; i < NUM_REA; i++)
                    for (int j = 0; j < NUM_SPECIES; j++)
                        fint >> h_react.React_ThirdCoef[i * NUM_SPECIES + j];
            }
            else if (!std::strcmp(Key_word, "*Arrhenius"))
            { // reaction rate constant parameters, A, B, E for Arrhenius law // unit: cm^3/mole/sec/kcal
                for (int i = 0; i < NUM_REA; i++)
                    fint >> h_react.Rargus[i * 6 + 0] >> h_react.Rargus[i * 6 + 1] >> h_react.Rargus[i * 6 + 2];
            } //-----------------*backwardArrhenius------------------//
            else if (!std::strcmp(Key_word, "*A"))
            {
                BackArre = true;
                // reaction rate constant parameters, A, B, E for Arrhenius law // unit: cm^3/mole/sec/kcal
                for (int i = 0; i < NUM_REA; i++)
                    fint >> h_react.Rargus[i * 6 + 3] >> h_react.Rargus[i * 6 + 4] >> h_react.Rargus[i * 6 + 5];
                break;
            } //-----------------*backwardArrhenius------------------//
        }
    }
    fint.close();
    IniSpeciesReactions();
}
// =======================================================
// =======================================================
void Setup::IniSpeciesReactions()
{
    for (size_t j = 0; j < NUM_SPECIES; j++)
    {
        reaction_list[j].clear();
        for (size_t i = 0; i < NUM_REA; i++)
        {
            if (h_react.Nu_f_[i * NUM_SPECIES + j] > 0 || h_react.Nu_b_[i * NUM_SPECIES + j] > 0)
            {
                reaction_list[j].push_back(i);
            }
        }
    }
    for (size_t i = 0; i < NUM_REA; i++)
    {
        reactant_list[i].clear();
        product_list[i].clear();
        species_list[i].clear();
        real_t sum = _DF(0.0);
        for (size_t j = 0; j < NUM_SPECIES; j++)
        {
            if (h_react.Nu_f_[i * NUM_SPECIES + j] > 0)
                reactant_list[i].push_back(j);
            if (h_react.Nu_b_[i * NUM_SPECIES + j] > 0)
                product_list[i].push_back(j);
            if (h_react.Nu_f_[i * NUM_SPECIES + j] > 0 || h_react.Nu_b_[i * NUM_SPECIES + j] > 0)
                species_list[i].push_back(j);
            // third body indicator
            sum += h_react.React_ThirdCoef[i * NUM_SPECIES + j];
        }
        h_react.third_ind[i] = (sum > _DF(0.0)) ? 1 : 0;
        ReactionType(0, i, h_react.Nu_f_, h_react.Nu_b_);
        ReactionType(1, i, h_react.Nu_b_, h_react.Nu_f_);
    }
}
// =======================================================
// =======================================================
void Setup::ReactionType(int flag, int i, int *Nuf, int *Nub)
{
    std::vector<int> forward_list, backward_list;
    if (flag == 0)
    {
        forward_list = reactant_list[i];
        backward_list = product_list[i];
    }
    else
    {
        forward_list = product_list[i];
        backward_list = reactant_list[i];
    }
    // reaction type
    h_react.react_type[i * 2 + flag] = 0;
    int Od_Rec = 0, Od_Pro = 0, Num_Repeat = 0; // the order of the reaction
    // loop all species in reaction "i"
    for (int l = 0; l < forward_list.size(); l++)
    {
        int species_id = forward_list[l];
        Od_Rec += Nuf[i * NUM_SPECIES + species_id];
    }
    for (int l = 0; l < backward_list.size(); l++)
    {
        int specie_id = backward_list[l];
        Od_Pro += Nub[i * NUM_SPECIES + specie_id];
    }
    for (int l = 0; l < forward_list.size(); l++)
    {
        int specie_id = forward_list[l];
        for (int l1 = 0; l1 < backward_list.size(); l1++)
        {
            int specie2_id = backward_list[l1];
            if (specie_id == specie2_id)
            {
                Num_Repeat++;
                break;
            }
        }
    }
    switch (Od_Rec)
    {
    case 0: // 0th-order
        h_react.react_type[i * 2 + flag] = 1;
        break;
    case 1: // 1st-order
        if (Od_Pro == 0)
            h_react.react_type[i * 2 + flag] = 2;
        else if (Od_Pro == 1)
            h_react.react_type[i * 2 + flag] = 3;
        else if (Od_Pro == 2)
        {
            h_react.react_type[i * 2 + flag] = Num_Repeat == 0 ? 4 : 5;
            if (Nub[i * NUM_SPECIES + backward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 12;
        }
        break;
    case 2: // 2nd-order
        if (Od_Pro == 0)
            h_react.react_type[i * 2 + flag] = 6;
        else if (Od_Pro == 1)
        {
            h_react.react_type[i * 2 + flag] = Num_Repeat == 0 ? 7 : 9;
            if (Nuf[i * NUM_SPECIES + forward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 11;
        }
        else if (Od_Pro == 2)
        {
            h_react.react_type[i * 2 + flag] = Num_Repeat == 0 ? 8 : 10;
            if (Nuf[i * NUM_SPECIES + forward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 13;
            if (Nub[i * NUM_SPECIES + backward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 14;
        }
        else if (Od_Pro == 3)
        {
            if (Nub[i * NUM_SPECIES + backward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 16;
        }
    case 3: // 3rd-order
        if (Od_Pro == 2)
        {
            if (Nuf[i * NUM_SPECIES + forward_list[0]] == 2 || Nuf[i * NUM_SPECIES + backward_list[0]] == 2)
                h_react.react_type[i * 2 + flag] = 15;
        }
        break;
    }
    if (h_react.react_type[i * 2 + flag] == 0)
#ifdef USE_MPI
        if (myRank == 0)
#endif
        {
            std::cout << "reaction type error for i = " << i << "\n";
        }
}
#endif // end COP_CHEME
// =======================================================
// =======================================================
#ifdef Visc
/**
 * @brief read collision integral table from "collision_integral.dat"
 */
void Setup::ReadOmega_table()
{
    std::string fpath = std::string(RPath) + "/collision_integral.dat";
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
// =======================================================
// =======================================================
/**
 * @brief get coefficients of polynominal fitted
 */
void Setup::GetFitCoefficient()
{
    ReadOmega_table(); // read Omega_table here for fitting
    for (int k = 0; k < NUM_SPECIES; k++)
    {                                                                                                                                             // Allocate Mem
        h_thermal.fitted_coefficients_visc[k] = middle::MallocHost<real_t>(h_thermal.fitted_coefficients_visc[k], order_polynominal_fitted, q);   // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));
        h_thermal.fitted_coefficients_therm[k] = middle::MallocHost<real_t>(h_thermal.fitted_coefficients_therm[k], order_polynominal_fitted, q); // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));

        real_t *specie_k = &(h_thermal.species_chara[k * SPCH_Sz]);
        Fitting(specie_k, specie_k, h_thermal.fitted_coefficients_visc[k], 0);  // Visc
        Fitting(specie_k, specie_k, h_thermal.fitted_coefficients_therm[k], 1); // diffu
        for (int j = 0; j < NUM_SPECIES; j++)
        {                                                                                                                                                   // Allocate Mem
            h_thermal.Dkj_matrix[k * NUM_SPECIES + j] = middle::MallocHost<real_t>(h_thermal.Dkj_matrix[k * NUM_SPECIES + j], order_polynominal_fitted, q); // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));

            real_t *specie_j = &(h_thermal.species_chara[j * SPCH_Sz]);
            if (k <= j)                                                                    // upper triangle
                Fitting(specie_k, specie_j, h_thermal.Dkj_matrix[k * NUM_SPECIES + j], 2); // Dim
            else
            { // lower triangle==>copy
                for (int n = 0; n < order_polynominal_fitted; n++)
                    h_thermal.Dkj_matrix[k * NUM_SPECIES + j][n] = h_thermal.Dkj_matrix[j * NUM_SPECIES + k][n];
            }
        }
    }
}

/**
 * @brief fitting procedure for transport coefficients
 * @para specie_k,the fitting is for specie_k
 * @para specie_j,if fitting is for binarry diffusion coefficient,specie_j is another specie; otherwise, it is set as the same with specie_k
 * @para aa the coefficients of the polynominal;
 * @para indicator fitting for viscosity(0),thermal conductivities(1) and binary diffusion coefficients(2)
 */
void Setup::Fitting(real_t *specie_k, real_t *specie_j, real_t *aa, int indicator)
{
    int mm = 12;
    real_t b[mm], AA[mm][order_polynominal_fitted], TT[] = {273.15, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0}; //{100, 200, 298.15, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000}; // 0 oC= 273.15K
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
    real_t Cv_trans = 1.5 * universal_gas_const * 1.0e3, Cv_rot, Cv_vib;
    int id = int(specie[SID]);
    real_t Cpi = HeatCapacity(h_thermal.Hia, T, h_thermal.Ri[id], id);
    real_t Cv = Cpi * specie[Wi] * 1.0e3 - universal_gas_const * 1.0e3; // unit:J/(kmol.K)
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
        Cv_rot = 1.0 * universal_gas_const * 1.0e3; // unit:J/(kmol*K)
        Cv_vib = Cv - 2.5 * universal_gas_const * 1.0e3;
        break;
    }
    case 2:
    {
        Cv_rot = 1.5 * universal_gas_const * 1.0e3;
        Cv_vib = Cv - 3.0 * universal_gas_const * 1.0e3;
        break;
    }
    }
    real_t rho = PP * specie[Wi] / T / universal_gas_const * 1.0e-3; // unit:g/cm3 equation5-32
    real_t Dkk = Dkj(specie, specie, T, PP);                         // unit:cm*cm/s
    real_t visc = viscosity(specie, T);                              // unit: Pa.s=kg/(m.s)
    real_t f_trans, f_rot, f_vib = rho * Dkk / (visc * 10.0);        // unit:1

    real_t Zrot = specie[Zrot_298] * ZrotFunc(specie[epsilon_kB] / 298.0) / ZrotFunc(specie[epsilon_kB] / T); // unit:1
    real_t Aa = 2.5 - f_vib, Bb = Zrot + 2.0 * (5.0 * Cv_rot / 3.0 / (universal_gas_const * 1.0e3) + f_vib) / pi;

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
        real_t alpha_n_star = alpha_n / std::pow(d_n, _DF(3));                                                     // equation5-13
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
#endif // Visc
// =======================================================
// =======================================================
/**
 * @brief update heat capacity
 * @param T temperature Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
 * @return real_t, unit: J/(kg.K)
 */
real_t Setup::HeatCapacity(real_t *Hia, const real_t T0, const real_t Ri, const int n)
{
    // real_t T = T0; // sycl::max<real_t>(T0, _DF(200.0));
    // real_t Cpi = _DF(0.0), _T = _DF(1.0) / T;
    // #if Thermo
    //     if (T >= (_DF(1000.0)) && T < (_DF(6000.0)))
    //         Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 1] * _T + Hia[n * 7 * 3 + 1 * 3 + 1]) * _T + Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] + (Hia[n * 7 * 3 + 5 * 3 + 1] + Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T);
    //     else if (T < (_DF(1000.0)))
    //         Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 0] * _T + Hia[n * 7 * 3 + 1 * 3 + 0]) * _T + Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] + (Hia[n * 7 * 3 + 5 * 3 + 0] + Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T);
    //     else if (T >= (_DF(6000.0)) && T < (_DF(15000.0)))
    //     {
    //         Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 2] * _T + Hia[n * 7 * 3 + 1 * 3 + 2]) * _T + Hia[n * 7 * 3 + 2 * 3 + 2] + (Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] + (Hia[n * 7 * 3 + 5 * 3 + 2] + Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T);
    //     }
    //     else
    //     {
    //         printf("T=%lf , Cpi=%lf , T > 15000 K,please check!!!NO Cpi[n] for T>15000 K \n", T, Cpi);
    //     }
    // #else // Cpi[n)/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    //     if (T > (1000.0))
    //         Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + (Hia[n * 7 * 3 + 1 * 3 + 0] + (Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + Hia[n * 7 * 3 + 4 * 3 + 0] * T) * T) * T) * T);
    //     else
    //         Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + (Hia[n * 7 * 3 + 1 * 3 + 1] + (Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + Hia[n * 7 * 3 + 4 * 3 + 1] * T) * T) * T) * T);
    // #endif

#if Thermo
    MARCO_HeatCapacity_NASA();
#else
    MARCO_HeatCapacity_JANAF();
#endif // end Thermo

    return Cpi;
}
// =======================================================
// =======================================================
/**
 * @brief calculate Hi of every compoent at given point	unit:J/kg/K // get_hi
 */
real_t Setup::Enthalpy(const real_t T0, const int n)
{
    real_t *Hia = h_thermal.Hia, *Hib = h_thermal.Hib, Ri = Ru / h_thermal.Wi[n];
    //     real_t hi = _DF(0.0), TT = T0, T = std::max<real_t>(T0, _DF(200.0));
    // #if Thermo
    //     if (T >= _DF(1000.0) && T < _DF(6000.0))
    //         hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 1] / T + Hia[n * 7 * 3 + 1 * 3 + 1] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 1] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 1] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 1]);
    //     else if (T < _DF(1000.0))
    //         hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 0] / T + Hia[n * 7 * 3 + 1 * 3 + 0] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 0] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 0] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 0]);
    //     else if (T >= _DF(6000.0))
    //         hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 2] / T + Hia[n * 7 * 3 + 1 * 3 + 2] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 2] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 2] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 2]);
    // #else
    //     // H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T
    //     if (T > _DF(1000.0))
    //         hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 0] + T * (Hia[n * 7 * 3 + 1 * 3 + 0] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 0] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 0] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 0] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 0]);
    //     else
    //         hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 1] + T * (Hia[n * 7 * 3 + 1 * 3 + 1] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 1] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 1] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 1] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 1]);
    // #endif
    //     if (TT < _DF(200.0)) // take low tempreture into consideration
    //     {                    // get_hi at T>200
    //         real_t Cpi = HeatCapacity(_DF(200.0), n);
    //         hi += Cpi * (TT - _DF(200.0));
    //     }

#if Thermo
    MARCO_Enthalpy_NASA();
#else
    MARCO_Enthalpy_JANAF();
#endif

    return hi;
}
// =======================================================
// =======================================================
/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
real_t Setup::get_Coph(const real_t yi[NUM_SPECIES], const real_t T)
{
    real_t h = _DF(0.0);
    for (size_t i = 0; i < NUM_SPECIES; i++)
    {
        real_t hi = Enthalpy(T, i);
        h += hi * yi[i];
    }
    return h;
}
// =======================================================
// =======================================================
/**
 * @brief calculate Gamma of the mixture at given point
 */
real_t Setup::get_CopGamma(const real_t yi[NUM_SPECIES], const real_t T)
{
    real_t Cp = _DF(0.0);
    for (size_t ii = 0; ii < NUM_SPECIES; ii++)
        Cp += yi[ii] * HeatCapacity(h_thermal.Hia, T, h_thermal.Ri[ii], ii);
    real_t CopW = get_MixtureW(h_thermal, yi);
    real_t _CopGamma = Cp / (Cp - Ru / CopW);
    if (_CopGamma > 1)
    {
        return _CopGamma;
    }
    else
    {
        printf("  Illegal Gamma captured.\n");
        exit(EXIT_FAILURE);
    }
}
// =======================================================
// =======================================================
void Setup::ReadIni(ConfigMap &configMap)
{
    /* initialize RUN parameters */
    OutTimeMethod=configMap.getInteger("run","OutTimeMethod",1);// use settings in .ini file by default
    if(1==OutTimeMethod)
    {                                                                     // settings in .ini file
        nOutTimeStamps = configMap.getInteger("run", "nOutTimeStamps", 0) + 1;
        OutTimeStamps = new real_t[nOutTimeStamps];
        OutTimeStart = configMap.getFloat("run", "OutTimeBeginning", 0.0f);
        OutTimeStamp = configMap.getFloat("run", "OutTimeInterval", 0.0f); // OutTimeStamp
        OutTimeStamps[0] = OutTimeStart;
        for (size_t n = 1; n < nOutTimeStamps; n++)
            OutTimeStamps[n] = OutTimeStamps[0] + real_t(n) * OutTimeStamp;
    }
    else if (0==OutTimeMethod)
    {
        std::string tpath = std::string(RPath) + "/time_stamps.dat";
        std::fstream fint(tpath);
        fint >> nOutTimeStamps; // read number of OutTimeStamps;
        OutTimeStamps = new real_t[nOutTimeStamps];
        for (int n = 0; n < nOutTimeStamps; n++)
            fint >> OutTimeStamps[n];
        fint.close();
    }
    else
    {
#ifdef USE_MPI
        if (myRank == 0)
#endif
        {
            std::cout << "Undefined Output Time Method." << std::endl;
        }
        exit(EXIT_FAILURE);
    }
    StartTime = configMap.getFloat("run", "StartTime", 0.0f); // physical time when the simulation start
    EndTime = configMap.getFloat("run", "EndTime", OutTimeStamps[nOutTimeStamps - 1]);
    OutTimeStamps[nOutTimeStamps - 1] = EndTime;
    NumThread = configMap.getInteger("run", "NumThread", 8);
    OutputDir = std::string(configMap.getString("run", "OutputDir", "./"));
    OutBoundary = configMap.getInteger("run", "OutBoundary", 0);
    OutDIRX = bool(configMap.getInteger("run", "OutDIRX", DIM_X));
    OutDIRY = bool(configMap.getInteger("run", "OutDIRY", DIM_Y));
    OutDIRZ = bool(configMap.getInteger("run", "OutDIRZ", DIM_Z));
    outpos_x = configMap.getInteger("run", "outpos_x", 0);
    outpos_y = configMap.getInteger("run", "outpos_y", 0);
    outpos_z = configMap.getInteger("run", "outpos_z", 0);
    Mach_Modified = configMap.getInteger("ini", "Mach_modified", 1);
    nStepmax = configMap.getInteger("run", "nStepMax", 10);
    nOutput = configMap.getInteger("run", "nOutMax", 0);
    OutInterval = configMap.getInteger("run", "OutInterval", nStepmax);
    POutInterval = configMap.getInteger("run", "PushInterval", 5);
    // for thread allign
    BlSz.BlockSize = configMap.getInteger("run", "DtBlockSize", 4);
    BlSz.dim_block_x = DIM_X ? configMap.getInteger("run", "blockSize_x", BlSz.BlockSize) : 1;
    BlSz.dim_block_y = DIM_Y ? configMap.getInteger("run", "blockSize_y", BlSz.BlockSize) : 1;
    BlSz.dim_block_z = DIM_Z ? configMap.getInteger("run", "blockSize_z", BlSz.BlockSize) : 1;

    /* initialize MPI parameters */
#ifdef USE_MPI
    BlSz.mx = DIM_X ? configMap.getInteger("mpi", "mx", 1) : 1;
    BlSz.my = DIM_Y ? configMap.getInteger("mpi", "my", 1) : 1;
    BlSz.mz = DIM_Z ? configMap.getInteger("mpi", "mz", 1) : 1;
#else                    // no USE_MPI
    BlSz.mx = 1;
    BlSz.my = 1;
    BlSz.mz = 1;
#endif                   // end USE_MPI
    BlSz.myMpiPos_x = 0; // initial rank postion to zero, will be changed in MpiTrans
    BlSz.myMpiPos_y = 0;
    BlSz.myMpiPos_z = 0;
    /* initialize MESH parameters */
    // DOMAIN_length = configMap.getFloat("mesh", "DOMAIN_length", 1.0);
    Domain_length = configMap.getFloat("mesh", "DOMAIN_length", 1.0);
    Domain_width = configMap.getFloat("mesh", "DOMAIN_width", 1.0);
    Domain_height = configMap.getFloat("mesh", "DOMAIN_height", 1.0);
    BlSz.Domain_xmin = configMap.getFloat("mesh", "xmin", 0.0); // 计算域x方向永远是最长边
    BlSz.Domain_ymin = configMap.getFloat("mesh", "ymin", 0.0);
    BlSz.Domain_zmin = configMap.getFloat("mesh", "zmin", 0.0);
    // read block size set from .ini
    BlSz.OutBC = OutBoundary;
    BlSz.X_inner = DIM_X ? configMap.getInteger("mesh", "X_inner", 1) : 1;
    BlSz.Y_inner = DIM_Y ? configMap.getInteger("mesh", "Y_inner", 1) : 1;
    BlSz.Z_inner = DIM_Z ? configMap.getInteger("mesh", "Z_inner", 1) : 1;
    BlSz.Bwidth_X = DIM_X ? configMap.getInteger("mesh", "Bwidth_X", 4) : 0;
    BlSz.Bwidth_Y = DIM_Y ? configMap.getInteger("mesh", "Bwidth_Y", 4) : 0;
    BlSz.Bwidth_Z = DIM_Z ? configMap.getInteger("mesh", "Bwidth_Z", 4) : 0;
    BlSz.CFLnumber = configMap.getFloat("mesh", "CFLnumber", 0.6);

    NUM_BISD = configMap.getInteger("mesh", "NUM_BISD", 1);
    width_xt = configMap.getFloat("mesh", "width_xt", 4.0);
    width_hlf = configMap.getFloat("mesh", "width_hlf", 2.0);
    mx_vlm = configMap.getFloat("mesh", "mx_vlm", 0.5);
    ext_vlm = configMap.getFloat("mesh", "ext_vlm", 0.5);
    BandforLevelset = configMap.getFloat("mesh", "BandforLevelset", 6.0);

    Boundarys[0] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_xmin", Symmetry));
    Boundarys[1] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_xmax", Symmetry));
    Boundarys[2] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_ymin", Symmetry));
    Boundarys[3] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_ymax", Symmetry));
    Boundarys[4] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_zmin", Symmetry));
    Boundarys[5] = static_cast<BConditions>(configMap.getInteger("mesh", "boundary_zmax", Symmetry));

    /* initialize FLUID parameters */
    fname[0] = std::string(configMap.getString("fluid", "fluid1_name", "O2"));
    material_kind[0] = configMap.getInteger("fluid", "fluid1_kind", 0);
    // material properties:1: phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
    material_props[0][0] = configMap.getFloat("fluid", "fluid1_phase_indicator", 0);
    material_props[0][1] = configMap.getFloat("fluid", "fluid1_gamma", 0);
    material_props[0][2] = configMap.getFloat("fluid", "fluid1_A", 0);
    material_props[0][3] = configMap.getFloat("fluid", "fluid1_B", 0);
    material_props[0][4] = configMap.getFloat("fluid", "fluid1_rho0", 0);
    material_props[0][5] = configMap.getFloat("fluid", "fluid1_R0", 0);
    material_props[0][6] = configMap.getFloat("fluid", "fluid1_lambda0", 0);
    material_props[0][7] = configMap.getFloat("fluid", "fluid1_ac", 0);
    // Ini Fluid dynamic states
    ini.Ma = configMap.getFloat("init", "blast_mach", 0);
    ini.blast_type = configMap.getInteger("fluid", "blast_type", 0);
    ini.blast_center_x = configMap.getFloat("init", "blast_center_x", 0);
    ini.blast_center_y = configMap.getFloat("init", "blast_center_y", 0);
    ini.blast_center_z = configMap.getFloat("init", "blast_center_z", 0);
    ini.blast_radius = configMap.getFloat("init", "blast_radius", 0);
    // downstream of blast
    ini.blast_density_out = configMap.getFloat("init", "blast_density_out", 0);
    ini.blast_pressure_out = configMap.getFloat("init", "blast_pressure_out", 0);
    ini.blast_T_out = configMap.getFloat("init", "blast_tempreture_out", 298.15);
    ini.blast_u_out = configMap.getFloat("init", "blast_u_out", 0);
    ini.blast_v_out = configMap.getFloat("init", "blast_v_out", 0);
    ini.blast_w_out = configMap.getFloat("init", "blast_w_out", 0);
    // upstream of blast
    ini.blast_density_in = configMap.getFloat("init", "blast_density_in", 0);
    ini.blast_pressure_in = configMap.getFloat("init", "blast_pressure_in", 0);
    ini.blast_T_in = configMap.getFloat("init", "blast_tempreture_in", 298.15);
    ini.blast_u_in = configMap.getFloat("init", "blast_u_in", 0);
    ini.blast_v_in = configMap.getFloat("init", "blast_v_in", 0);
    ini.blast_w_in = configMap.getFloat("init", "blast_w_in", 0);
    // states inside mixture bubble
#ifdef COP
    ini.cop_type = configMap.getInteger("init", "cop_type", 0);
    ini.cop_center_x = configMap.getFloat("init", "cop_center_x", 0);
    ini.cop_center_y = configMap.getFloat("init", "cop_center_y", 0);
    ini.cop_center_z = configMap.getFloat("init", "cop_center_z", 0);
    ini.cop_density_in = configMap.getFloat("init", "cop_density_in", ini.blast_density_out);
    ini.cop_pressure_in = configMap.getFloat("init", "cop_pressure_in", ini.blast_pressure_out);
    ini.cop_T_in = configMap.getFloat("init", "cop_tempreture_in", ini.blast_T_out);
    ini.cop_y1_in = configMap.getFloat("init", "cop_y1_in", 0);
    ini.cop_y1_out = configMap.getFloat("init", "cop_y1_out", 0);
#endif // end COP
    // bubble size
    real_t Dmin = Domain_length + Domain_width + Domain_height;
#if DIM_X
    Dmin = std::min(Domain_length, Dmin);
#endif
#if DIM_Y
    Dmin = std::min(Domain_width, Dmin);
#endif
#if DIM_Z
    Dmin = std::min(Domain_height, Dmin);
#endif

    ini.xa = configMap.getFloat("init", "bubble_shape_x", 0.4 * Dmin);
    ini.yb = ini.xa / configMap.getFloat("init", "bubble_shape_ratioy", 1.0);
    ini.zc = ini.xa / configMap.getFloat("init", "bubble_shape_ratioz", 1.0);
    ini.yb = configMap.getFloat("init", "bubble_shape_y", ini.yb);
    ini.zc = configMap.getFloat("init", "bubble_shape_z", ini.zc);
    bubble_boundary = configMap.getFloat("init", "bubble_boundary_cells", 2);
    ini.C = ini.xa * configMap.getFloat("init", "bubble_boundary_width", real_t(BlSz.mx * BlSz.X_inner) * bubble_boundary);
}
// =======================================================
// =======================================================
void Setup::init()
{ // set other parameters
#ifndef MIDDLE_SYCL_ENABLED
    BlSz.dim_blk = middle::range_t(BlSz.dim_block_x, BlSz.dim_block_y, BlSz.dim_block_z);
    BlSz.dim_grid = middle::AllocThd(BlSz.X_inner, BlSz.Y_inner, BlSz.Z_inner, BlSz.dim_blk);
#endif // end non def MIDDLE_SYCL_ENABLED

    BlSz.dx = DIM_X ? Domain_length / real_t(BlSz.mx * BlSz.X_inner) : _DF(1.0); //
    BlSz.dy = DIM_Y ? Domain_width / real_t(BlSz.my * BlSz.Y_inner) : _DF(1.0);
    BlSz.dz = DIM_Z ? Domain_height / real_t(BlSz.mz * BlSz.Z_inner) : _DF(1.0);

    BlSz.Domain_xmax = BlSz.Domain_xmin + Domain_length;
    BlSz.Domain_ymax = BlSz.Domain_ymin + Domain_width;
    BlSz.Domain_zmax = BlSz.Domain_zmin + Domain_height;

    // ini.blast_center_x = BlSz.Domain_xmin + ini.blast_center_x * Domain_length;
    // ini.blast_center_y = BlSz.Domain_ymin + ini.blast_center_y * Domain_width;
    // ini.blast_center_z = BlSz.Domain_zmin + ini.blast_center_z * Domain_height;

#ifdef COP
    // ini.cop_center_x = BlSz.Domain_xmin + ini.cop_center_x * Domain_length;
    // ini.cop_center_y = BlSz.Domain_ymin + ini.cop_center_y * Domain_width;
    // ini.cop_center_z = BlSz.Domain_zmin + ini.cop_center_z * Domain_height;
#endif
#if 2 == NumFluid
    ini.bubble_center_x = BlSz.Domain_xmin + ini.bubble_center_x * Domain_length;
    ini.bubble_center_y = BlSz.Domain_ymin + ini.bubble_center_y * Domain_width;
    ini.bubble_center_z = BlSz.Domain_zmin + ini.bubble_center_z * Domain_height;
#endif

    BlSz.dl = BlSz.dx + BlSz.dy + BlSz.dz;
    real_t cdl = BlSz.dl;
    // // x direction
#if DIM_X
    BlSz.Xmax = (BlSz.X_inner + 2 * BlSz.Bwidth_X); // maximum number of total cells in x direction
    BlSz.dl = std::min(BlSz.dl, BlSz.dx);
    cdl = std::max(cdl, BlSz.dx);
#else
    BlSz.Xmax = 1;
#endif
    // // y direction
#if DIM_Y
    BlSz.Ymax = (BlSz.Y_inner + 2 * BlSz.Bwidth_Y); // maximum number of total cells in y direction
    BlSz.dl = std::min(BlSz.dl, BlSz.dy);
    cdl = std::max(cdl, BlSz.dy);
#else
    BlSz.Ymax = 1;
#endif
    // // z direction
#if DIM_Z
    BlSz.Zmax = (BlSz.Z_inner + 2 * BlSz.Bwidth_Z); // maximum number of total cells in z direction
    BlSz.dl = std::min(BlSz.dl, BlSz.dz);
    cdl = std::max(cdl, BlSz.dz);
#else
    BlSz.Zmax = 1;
#endif
    dt = _DF(0.0);
    BlSz.offx = (_DF(0.5) - BlSz.Bwidth_X + BlSz.myMpiPos_x * BlSz.X_inner) * BlSz.dx + BlSz.Domain_xmin;
    BlSz.offy = (_DF(0.5) - BlSz.Bwidth_Y + BlSz.myMpiPos_y * BlSz.Y_inner) * BlSz.dy + BlSz.Domain_ymin;
    BlSz.offz = (_DF(0.5) - BlSz.Bwidth_Z + BlSz.myMpiPos_z * BlSz.Z_inner) * BlSz.dz + BlSz.Domain_zmin;

    BlSz._dx = _DF(1.0) / BlSz.dx;
    BlSz._dy = _DF(1.0) / BlSz.dy;
    BlSz._dz = _DF(1.0) / BlSz.dz;
    BlSz._dl = _DF(1.0) / BlSz.dl;

    // bubble size: two cell boundary
    ini._xa2 = _DF(1.0) / (ini.xa * ini.xa);
    ini._yb2 = _DF(1.0) / (ini.yb * ini.yb);
    ini._zc2 = _DF(1.0) / (ini.zc * ini.zc);
    real_t xa_in = int(ini.xa / BlSz.dx) * BlSz.dx;
    real_t yb_in = int(ini.yb / BlSz.dy) * BlSz.dy;
    real_t zc_in = int(ini.zc / BlSz.dz) * BlSz.dz;
    real_t xa_out = xa_in + bubble_boundary * BlSz.dx;
    real_t yb_out = yb_in + bubble_boundary * BlSz.dy;
    real_t zc_out = zc_in + bubble_boundary * BlSz.dz;
    ini._xa2_in = _DF(1.0) / (xa_in * xa_in);
    ini._yb2_in = _DF(1.0) / (yb_in * yb_in);
    ini._zc2_in = _DF(1.0) / (zc_in * zc_in);
    ini._xa2_out = _DF(1.0) / (xa_out * xa_out);
    ini._yb2_out = _DF(1.0) / (yb_out * yb_out);
    ini._zc2_out = _DF(1.0) / (zc_out * zc_out);
    // DataBytes set
    Block_Inner_Cell_Size = (BlSz.X_inner * BlSz.Y_inner * BlSz.Z_inner);
    Block_Inner_Data_Size = (Block_Inner_Cell_Size * Emax);
    Block_Cell_Size = (BlSz.Xmax * BlSz.Ymax * BlSz.Zmax);
    Block_Data_Size = (Block_Cell_Size * Emax);
    bytes = BlSz.Xmax * BlSz.Ymax * BlSz.Zmax * sizeof(real_t);
    cellbytes = Emax * bytes;
} // Setup::init
// =======================================================
// =======================================================
void Setup::CpyToGPU()
{
#ifdef USE_MPI
    if (0 == mpiTrans->myRank)
#endif // end USE_MPI
    {
        std::cout << "<---------------------------------------------------> \n";
        std::cout << "Setup_ini is copying buffers into Device . ";
    }
    d_thermal.species_chara = middle::MallocDevice<real_t>(d_thermal.species_chara, NUM_SPECIES * SPCH_Sz, q);
    d_thermal.Ri = middle::MallocDevice<real_t>(d_thermal.Ri, NUM_SPECIES, q);
    d_thermal.Wi = middle::MallocDevice<real_t>(d_thermal.Wi, NUM_SPECIES, q);
    d_thermal._Wi = middle::MallocDevice<real_t>(d_thermal._Wi, NUM_SPECIES, q);
    d_thermal.Hia = middle::MallocDevice<real_t>(d_thermal.Hia, NUM_SPECIES * 7 * 3, q);
    d_thermal.Hib = middle::MallocDevice<real_t>(d_thermal.Hib, NUM_SPECIES * 2 * 3, q);
    d_thermal.species_ratio_in = middle::MallocDevice<real_t>(d_thermal.species_ratio_in, NUM_SPECIES, q);
    d_thermal.species_ratio_out = middle::MallocDevice<real_t>(d_thermal.species_ratio_out, NUM_SPECIES, q);
    d_thermal.xi_in = middle::MallocDevice<real_t>(d_thermal.xi_in, NUM_SPECIES, q);
    d_thermal.xi_out = middle::MallocDevice<real_t>(d_thermal.xi_out, NUM_SPECIES, q);

    middle::MemCpy<real_t>(d_thermal.species_chara, h_thermal.species_chara, NUM_SPECIES * SPCH_Sz, q);
    middle::MemCpy<real_t>(d_thermal.Ri, h_thermal.Ri, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal.Wi, h_thermal.Wi, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal._Wi, h_thermal._Wi, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal.Hia, h_thermal.Hia, NUM_SPECIES * 7 * 3, q);
    middle::MemCpy<real_t>(d_thermal.Hib, h_thermal.Hib, NUM_SPECIES * 2 * 3, q);
    middle::MemCpy<real_t>(d_thermal.species_ratio_in, h_thermal.species_ratio_in, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal.species_ratio_out, h_thermal.species_ratio_out, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal.xi_in, h_thermal.xi_in, NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_thermal.xi_out, h_thermal.xi_out, NUM_SPECIES, q);

#ifdef Visc
    for (int k = 0; k < NUM_SPECIES; k++)
    {                                                                                                                                               // Allocate Mem
        d_thermal.fitted_coefficients_visc[k] = middle::MallocDevice<real_t>(d_thermal.fitted_coefficients_visc[k], order_polynominal_fitted, q);   // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));
        d_thermal.fitted_coefficients_therm[k] = middle::MallocDevice<real_t>(d_thermal.fitted_coefficients_therm[k], order_polynominal_fitted, q); // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));
        middle::MemCpy<real_t>(d_thermal.fitted_coefficients_visc[k], h_thermal.fitted_coefficients_visc[k], order_polynominal_fitted, q);
        middle::MemCpy<real_t>(d_thermal.fitted_coefficients_therm[k], h_thermal.fitted_coefficients_therm[k], order_polynominal_fitted, q);
        for (int j = 0; j < NUM_SPECIES; j++)
        {                                                                                                                                                     // Allocate Mem
            d_thermal.Dkj_matrix[k * NUM_SPECIES + j] = middle::MallocDevice<real_t>(d_thermal.Dkj_matrix[k * NUM_SPECIES + j], order_polynominal_fitted, q); // static_cast<real_t *>(sycl::malloc_host(order_polynominal_fitted * sizeof(real_t), q));
            middle::MemCpy<real_t>(d_thermal.Dkj_matrix[k * NUM_SPECIES + j], h_thermal.Dkj_matrix[k * NUM_SPECIES + j], order_polynominal_fitted, q);
        }
    }
#endif // end Visc
#ifdef COP_CHEME
    d_react.Nu_f_ = middle::MallocDevice<int>(d_react.Nu_f_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    d_react.Nu_b_ = middle::MallocDevice<int>(d_react.Nu_b_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    d_react.Nu_d_ = middle::MallocDevice<int>(d_react.Nu_d_, NUM_REA * NUM_SPECIES, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(int), q));
    d_react.react_type = middle::MallocDevice<int>(d_react.react_type, NUM_REA * 2, q);                        // static_cast<int *>(sycl::malloc_host(NUM_REA * 2 * sizeof(int), q));
    d_react.third_ind = middle::MallocDevice<int>(d_react.third_ind, NUM_REA, q);                              // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    d_react.React_ThirdCoef = middle::MallocDevice<real_t>(d_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q); // static_cast<real_t *>(sycl::malloc_host(NUM_REA * NUM_SPECIES * sizeof(real_t), q));
    d_react.Rargus = middle::MallocDevice<real_t>(d_react.Rargus, NUM_REA * 6, q);                             // static_cast<real_t *>(sycl::malloc_host(NUM_REA * 6 * sizeof(real_t), q));

    middle::MemCpy<int>(d_react.Nu_f_, h_react.Nu_f_, NUM_REA * NUM_SPECIES, q);
    middle::MemCpy<int>(d_react.Nu_b_, h_react.Nu_b_, NUM_REA * NUM_SPECIES, q);
    middle::MemCpy<int>(d_react.Nu_d_, h_react.Nu_d_, NUM_REA * NUM_SPECIES, q);
    middle::MemCpy<int>(d_react.react_type, h_react.react_type, NUM_REA * 2, q);
    middle::MemCpy<int>(d_react.third_ind, h_react.third_ind, NUM_REA, q);
    middle::MemCpy<real_t>(d_react.React_ThirdCoef, h_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q);
    middle::MemCpy<real_t>(d_react.Rargus, h_react.Rargus, NUM_REA * 6, q);

    h_react.rns = middle::MallocHost<int>(h_react.rns, NUM_SPECIES, q); // static_cast<int *>(sycl::malloc_host(NUM_SPECIES * sizeof(int), q));
    for (size_t j = 0; j < NUM_SPECIES; j++)
    {
        h_react.rns[j] = reaction_list[j].size();
        h_react.reaction_list[j] = middle::MallocHost<int>(h_react.reaction_list[j], h_react.rns[j], q);                          // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.rns[j], q));
        middle::MemCpy(h_react.reaction_list[j], &(reaction_list[j][0]), sizeof(int) * h_react.rns[j], q, middle::MemCpy_t::HtH); // q.memcpy(h_react.reaction_list[j], &(reaction_list[j][0]), sizeof(int) * h_react.rns[j]);

        d_react.reaction_list[j] = middle::MallocDevice<int>(d_react.reaction_list[j], h_react.rns[j], q); // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.rns[j], q));
        middle::MemCpy<int>(d_react.reaction_list[j], h_react.reaction_list[j], h_react.rns[j], q);        // q.memcpy(h_react.reaction_list[j], &(reaction_list[j][0]), sizeof(int) * h_react.rns[j]);
    }
    h_react.rts = middle::MallocHost<int>(h_react.rts, NUM_REA, q); // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    h_react.pls = middle::MallocHost<int>(h_react.pls, NUM_REA, q); // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    h_react.sls = middle::MallocHost<int>(h_react.sls, NUM_REA, q); // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    for (size_t i = 0; i < NUM_REA; i++)
    {
        h_react.rts[i] = reactant_list[i].size();
        h_react.pls[i] = product_list[i].size();
        h_react.sls[i] = species_list[i].size();
        h_react.reactant_list[i] = middle::MallocHost<int>(h_react.reactant_list[i], h_react.rts[i], q);                          // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.rts[i], q));
        h_react.product_list[i] = middle::MallocHost<int>(h_react.product_list[i], h_react.pls[i], q);                            // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.pls[i], q));
        h_react.species_list[i] = middle::MallocHost<int>(h_react.species_list[i], h_react.sls[i], q);                            // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.sls[i], q));
        middle::MemCpy(h_react.reactant_list[i], &(reactant_list[i][0]), sizeof(int) * h_react.rts[i], q, middle::MemCpy_t::HtH); // q.memcpy(h_react.reactant_list[i], &(reactant_list[i][0]), sizeof(int) * h_react.rts[i]);
        middle::MemCpy(h_react.product_list[i], &(product_list[i][0]), sizeof(int) * h_react.pls[i], q, middle::MemCpy_t::HtH);   // q.memcpy(h_react.product_list[i], &(product_list[i][0]), sizeof(int) * h_react.pls[i]);
        middle::MemCpy(h_react.species_list[i], &(species_list[i][0]), sizeof(int) * h_react.sls[i], q, middle::MemCpy_t::HtH);   // q.memcpy(h_react.species_list[i], &(species_list[i][0]), sizeof(int) * h_react.sls[i]);

        d_react.reactant_list[i] = middle::MallocDevice<int>(d_react.reactant_list[i], h_react.rts[i], q); // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.rts[i], q));
        d_react.product_list[i] = middle::MallocDevice<int>(d_react.product_list[i], h_react.pls[i], q);   // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.pls[i], q));
        d_react.species_list[i] = middle::MallocDevice<int>(d_react.species_list[i], h_react.sls[i], q);   // static_cast<int *>(sycl::malloc_host(sizeof(int) * h_react.sls[i], q));
        middle::MemCpy<int>(d_react.reactant_list[i], h_react.reactant_list[i], h_react.rts[i], q);        // q.memcpy(h_react.reactant_list[i], &(reactant_list[i][0]), sizeof(int) * h_react.rts[i]);
        middle::MemCpy<int>(d_react.product_list[i], h_react.product_list[i], h_react.pls[i], q);          // q.memcpy(h_react.product_list[i], &(product_list[i][0]), sizeof(int) * h_react.pls[i]);
        middle::MemCpy<int>(d_react.species_list[i], h_react.species_list[i], h_react.sls[i], q);          // q.memcpy(h_react.species_list[i], &(species_list[i][0]), sizeof(int) * h_react.sls[i]);
    }
    d_react.rns = middle::MallocDevice<int>(d_react.rns, NUM_SPECIES, q); // static_cast<int *>(sycl::malloc_host(NUM_SPECIES * sizeof(int), q));
    d_react.rts = middle::MallocDevice<int>(d_react.rts, NUM_REA, q);     // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    d_react.pls = middle::MallocDevice<int>(d_react.pls, NUM_REA, q);     // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    d_react.sls = middle::MallocDevice<int>(d_react.sls, NUM_REA, q);     // static_cast<int *>(sycl::malloc_host(NUM_REA * sizeof(int), q));
    middle::MemCpy<int>(d_react.rns, h_react.rns, NUM_SPECIES, q);
    middle::MemCpy<int>(d_react.rts, h_react.rts, NUM_REA, q);
    middle::MemCpy<int>(d_react.pls, h_react.pls, NUM_REA, q);
    middle::MemCpy<int>(d_react.sls, h_react.sls, NUM_REA, q);
#endif // COP_CHEME
#ifdef USE_MPI
    if (0 == mpiTrans->myRank)
#endif // end USE_MPI
    {
        std::cout << " . Done \n";
        std::cout << "<---------------------------------------------------> \n";
    }
}
// =======================================================
// =======================================================
void Setup::print()
{
    if (mach_shock)
    {
        // States initializing
        printf("blast_type: %d and blast_center(x = %.6lf , y = %.6lf , z = %.6lf).\n", ini.blast_type, ini.blast_center_x, ini.blast_center_y, ini.blast_center_z);
        printf(" Use Mach number = %lf to reinitialize fluid states upstream the shock\n", ini.Ma);
        printf("  States of   upstream:     (P = %.6lf, T = %.6lf, rho = %.6lf, u = %.6lf, v = %.6lf, w = %.6lf).\n", ini.blast_pressure_in, ini.blast_T_in, ini.blast_density_in, ini.blast_u_in, ini.blast_v_in, ini.blast_w_in);
        printf("  States of downstream:     (P = %.6lf, T = %.6lf, rho = %.6lf, u = %.6lf, v = %.6lf, w = %.6lf).\n", ini.blast_pressure_out, ini.blast_T_out, ini.blast_density_out, ini.blast_u_out, ini.blast_v_out, ini.blast_w_out);
#ifdef COP
        printf("cop_type:   %d and cop_radius: %lf.\n", ini.cop_type, ini.cop_radius);
        printf("  cop_center: (x = %.6lf, y = %.6lf, z = %.6lf).\n", ini.cop_center_x, ini.cop_center_y, ini.cop_center_z);
        // printf("  States inside cop bubble: (P = %.6lf, T = %.6lf, rho = %.6lf, u = %.6lf, v = %.6lf, w = %.6lf).\n", ini.cop_pressure_in, ini.cop_T_in, ini.cop_density_in, ini.blast_u_out, ini.blast_v_out, ini.blast_w_out);
#endif // end COP
    }
// 后接流体状态输出
#ifdef COP
#if Visc
    for (size_t n = 0; n < NUM_SPECIES; n++)
    {
        printf("compoent[%zd]: %s, characteristics(geo, epsilon_kB, L-J collision diameter, dipole moment, polarizability, Zort_298, molar mass): \n", n, species_name[n].c_str());
        printf("  %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf\n", h_thermal.species_chara[n * SPCH_Sz + 0],
               h_thermal.species_chara[n * SPCH_Sz + 1], h_thermal.species_chara[n * SPCH_Sz + 2],
               h_thermal.species_chara[n * SPCH_Sz + 3], h_thermal.species_chara[n * SPCH_Sz + 4],
               h_thermal.species_chara[n * SPCH_Sz + 5], h_thermal.species_chara[n * SPCH_Sz + 6]);
    }
#endif // Visc
#ifdef COP_CHEME
    printf("%d Reactions been actived                                     \n", NUM_REA);
    for (size_t id = 0; id < NUM_REA; id++)
    {
        if (id + 1 < 10)
            std::cout << "Reaction:" << id + 1 << "  ";
        else
            std::cout << "Reaction:" << id + 1 << " ";
        int numreactant = 0, numproduct = 0;
        // output reactant
        for (int j = 0; j < NUM_SPECIES; ++j)
            numreactant += h_react.Nu_f_[id * NUM_SPECIES + j];

        if (numreactant == 0)
            std::cout << "0	"
                      << "  <-->  ";
        else
        {
            for (int j = 0; j < NUM_SPECIES; j++)
            {
                if (j != NUM_COP)
                {
                    if (h_react.Nu_f_[id * NUM_SPECIES + j] > 0)
                        std::cout << h_react.Nu_f_[id * NUM_SPECIES + j] << " " << species_name[j] << " + ";
                }
                else
                {
                    if (h_react.React_ThirdCoef[id * NUM_SPECIES + j] > 1e-6)
                        std::cout << " M ";
                    std::cout << "  <-->  ";
                }
            }
        }
        // output product
        for (int j = 0; j < NUM_SPECIES; ++j)
            numproduct += h_react.Nu_b_[id * NUM_SPECIES + j];
        if (numproduct == 0)
            std::cout << "0	";
        else
        {
            for (int j = 0; j < NUM_SPECIES; j++)
            {
                if (j != NUM_COP)
                {
                    if (h_react.Nu_b_[id * NUM_SPECIES + j] > 0)
                        std::cout << h_react.Nu_b_[id * NUM_SPECIES + j] << " " << species_name[j] << " + ";
                }
                else
                {
                    if (h_react.React_ThirdCoef[id * NUM_SPECIES + j] > 1e-6)
                        std::cout << " M ";
                }
            }
        }
        std::cout << " with rate: " << h_react.Rargus[id * 6 + 0] << " " << h_react.Rargus[id * 6 + 1] << " " << h_react.Rargus[id * 6 + 2] << std::endl;
        //-----------------*backwardArrhenius------------------//
        if (BackArre)
        {
            std::cout << " with back rate: " << h_react.Rargus[id * 6 + 3] << " " << h_react.Rargus[id * 6 + 4] << " " << h_react.Rargus[id * 6 + 5] << std::endl;
        }
        //-----------------*backwardArrhenius------------------//
    }
    std::cout << "species mass fraction" << std::endl;
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        std::cout << "species[" << n << "]=" << std::setw(5) << species_name[n] << std::setw(10) << h_thermal.species_ratio_in[n] << std::setw(10) << h_thermal.species_ratio_out[n] << std::endl;
    }
#endif // end COP_CHEME
#endif // COP

#if 1 != NumFluid
    printf("Extending width: width_xt                                : %lf\n", width_xt);
    printf("Ghost-fluid update width: width_hlf                      : %lf\n", width_hlf);
    printf("cells' volume less than this vule will be mixed          : %lf\n", mx_vlm);
    printf("cells' volume less than states updated based on mixed    : %lf\n", ext_vlm);
    printf("half-width of level set narrow band                      : %lf\n", BandforLevelset);
    printf("Number of fluids                                         : %d\n", NumFluid);
#endif // end NumFluid
#if 2 == NumFluid
    printf("bubble_type: %d and bubble_radius: %lf.\n", ini.bubble_type, ini.bubbleSz);
    printf("  bubble_center: (x =%.6lf,y =%.6lf,z =%.6lf).\n", ini.bubble_center_x, ini.bubble_center_y, ini.bubble_center_z);
    printf("  States inside multiphase bubble:(P =%.6lf,T =%.6lf,rho =%.6lf,u =%.6lf,v =%.6lf,w =%.6lf,).\n", ini.cop_pressure_in, ini.cop_T_in, ini.cop_density_in, ini.blast_u_out, ini.blast_v_out, ini.blast_w_out);
    for (n = 0; n < NumFluid; n++)
    { // 0: phase_indicator, 1: gamma, 2: A, 3: B, 4: rho0, 5: R_0, 6: lambda_0, 7: a(rtificial)s(peed of)s(ound)
        printf("fluid[%d]: %s, characteristics(Material, Phase_indicator, Gamma, A, B, Rho0, R_0, Lambda_0, artificial speed of sound): \n", n, fname[n].c_str());
        printf("  %d,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f\n", material_kind[n],
               material_props[n][0], material_props[n][1], material_props[n][2], material_props[n][3],
               material_props[n][4], material_props[n][5], material_props[n][6], material_props[n][7]);
    }
#endif

    std::cout << "<---------------------------------------------------> \n";
    printf("Start time: %.6lf and End time: %.6lf                        \n", StartTime, EndTime);
    printf("   XYZ dir Domain size:                  %1.3lf x %1.3lf x %1.3lf\n", Domain_length, Domain_width, Domain_height);
#ifdef USE_MPI // end USE_MPI
    {          // print information about current setup
        std::cout << "MPI rank mesh setup as below: \n";
        std::cout << "   Global resolution of MPI World: " << BlSz.X_inner * BlSz.mx << " x " << BlSz.Y_inner * BlSz.my << " x " << BlSz.Z_inner * BlSz.mz << "\n";
        std::cout << "   Local  resolution of one Rank : " << BlSz.X_inner << " x " << BlSz.Y_inner << " x " << BlSz.Z_inner << "\n";
        std::cout << "   MPI Cartesian topology        : " << BlSz.mx << " x " << BlSz.my << " x " << BlSz.mz << std::endl;
        // std::cout << "<---------------------------------------------------> \n";
    }
#else
    std::cout << "   Resolution of Domain:                 " << BlSz.X_inner << " x " << BlSz.Y_inner << " x " << BlSz.Z_inner << "\n";
#endif // end USE_MPI
    printf("   Difference steps: dx, dy, dz:         %lf, %lf, %lf\n", BlSz.dx, BlSz.dy, BlSz.dz);
    std::cout << "<---------------------------------------------------> \n";

} // Setup::print