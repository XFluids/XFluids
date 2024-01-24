#include "../setupini.h"
#include "../../solver_Ini/Mixing_device.h"

// =======================================================
// =======================================================
void Setup::ReadSpecies()
{
    // compoent in or out bubble differs
    h_thermal.species_ratio_in = middle::MallocHost<real_t>(h_thermal.species_ratio_in, NUM_SPECIES, q);
    h_thermal.species_ratio_out = middle::MallocHost<real_t>(h_thermal.species_ratio_out, NUM_SPECIES, q);

    std::string path = WorkDir + std::string(RFile) + "/species_list.dat";
    std::fstream fins(path);
    std::string buffer, sname;
    // getline(fins, buffer);
    species_name.clear();
    for (int n = 0; n < NUM_SPECIES; n++)
        fins >> sname, species_name.push_back(sname); // name list of the species
    // NOTE: Xe_id and N2_id depends on species_list
    BlSz.Xe_id = NUM_SPECIES - 3, BlSz.N2_id = NUM_SPECIES - 2;
#ifdef COP
    std::string line;
    getline(fins, line);
    for (int n = 0; n < NUM_SPECIES; n++) // molar ratio
        fins >> h_thermal.species_ratio_out[n];
    getline(fins, line);
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
    h_thermal.species_chara = middle::MallocHost<real_t>(h_thermal.species_chara, NUM_SPECIES * SPCH_Sz, q);
    h_thermal.Ri = middle::MallocHost<real_t>(h_thermal.Ri, NUM_SPECIES, q);
    h_thermal.Wi = middle::MallocHost<real_t>(h_thermal.Wi, NUM_SPECIES, q);
    h_thermal._Wi = middle::MallocHost<real_t>(h_thermal._Wi, NUM_SPECIES, q);
    h_thermal.Hia = middle::MallocHost<real_t>(h_thermal.Hia, NUM_SPECIES * 7 * 3, q);
    h_thermal.Hib = middle::MallocHost<real_t>(h_thermal.Hib, NUM_SPECIES * 2 * 3, q);
    h_thermal.Hia_NASA = middle::MallocHost<real_t>(h_thermal.Hia, NUM_SPECIES * 7 * 3, q);
    h_thermal.Hib_NASA = middle::MallocHost<real_t>(h_thermal.Hib, NUM_SPECIES * 2 * 3, q);
    h_thermal.Hia_JANAF = middle::MallocHost<real_t>(h_thermal.Hia, NUM_SPECIES * 7 * 3, q);

    char Key_word[128];
    // // read NASA
    std::fstream fincn(WorkDir + std::string(RPath) + "/thermal_dynamics.dat");
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        // check the name of the species "n"
        std::string my_string = "*" + species_name[n];
        char *species_name_n = new char[my_string.size() + 1];
        std::strcpy(species_name_n, my_string.c_str());
        // reset file point location
        fincn.seekg(0);
        while (!fincn.eof())
        {
            fincn >> Key_word;
            if (!std::strcmp(Key_word, "*END"))
                break;
            if (!std::strcmp(Key_word, species_name_n))
            {
                // low temperature parameters, 200K<T<1000K
                for (int m = 0; m < 7; m++)
                    fincn >> h_thermal.Hia_NASA[n * 7 * 3 + m * 3 + 0]; // a1-a7
                for (int m = 0; m < 2; m++)
                    fincn >> h_thermal.Hib_NASA[n * 2 * 3 + m * 3 + 0]; // b1,b2
                // high temperature parameters, 1000K<T<6000K
                for (int m = 0; m < 7; m++)
                    fincn >> h_thermal.Hia_NASA[n * 7 * 3 + m * 3 + 1]; // a1-a7
                for (int m = 0; m < 2; m++)
                    fincn >> h_thermal.Hib_NASA[n * 2 * 3 + m * 3 + 1]; // b1,b2
                // high temperature parameters, 6000K<T<15000K
                for (int m = 0; m < 7; m++)
                    fincn >> h_thermal.Hia_NASA[n * 7 * 3 + m * 3 + 2]; // a1-a7
                for (int m = 0; m < 2; m++)
                    fincn >> h_thermal.Hib_NASA[n * 2 * 3 + m * 3 + 2]; // b1,b2
#if Thermo
                fincn >> h_thermal.species_chara[n * SPCH_Sz + Wi]; // species[n].Wi; // molar mass, unit: g/mol
#endif
                break;
            }
        }
    }
    fincn.close();

    // // read JANAF
    std::fstream finc(WorkDir + std::string(RPath) + "/thermal_dynamics_janaf.dat");
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
                // high temperature parameters
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia_JANAF[n * 7 * 3 + m * 3 + 0]; // a1-a7
                // low temperature parameters
                for (int m = 0; m < 7; m++)
                    finc >> h_thermal.Hia_JANAF[n * 7 * 3 + m * 3 + 1]; // a1-a7
#if !Thermo
                finc >> h_thermal.species_chara[n * SPCH_Sz + Wi]; // species[n].Wi; // molar mass, unit: g/mol
#endif
                break;
            }
        }
    }
    finc.close();

#if Thermo
    std::memcpy(h_thermal.Hia, h_thermal.Hia_NASA, NUM_SPECIES * 7 * 3 * sizeof(real_t));
    std::memcpy(h_thermal.Hib, h_thermal.Hib_NASA, NUM_SPECIES * 2 * 3 * sizeof(real_t));
#else
    std::memcpy(h_thermal.Hia, h_thermal.Hia_JANAF, NUM_SPECIES * 7 * 3 * sizeof(real_t));
#endif // end Hia and Hib copy

    /**
     */
    std::string spath = WorkDir + std::string(RPath) + "/transport_data.dat";
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
                fint >> h_thermal.species_chara[i * SPCH_Sz + 1]; // real_t epsilon_kB;//epsilon: Lennard-Jones potential well depth;unit:K//势井深度
                fint >> h_thermal.species_chara[i * SPCH_Sz + 2]; // d;//Lennard-Jones collision diameter, unit: angstroms,10e-10m//碰撞直径in 4-3;
                fint >> h_thermal.species_chara[i * SPCH_Sz + 3]; // mue;//dipole moment,unit:Debye(m);//偶极距
                fint >> h_thermal.species_chara[i * SPCH_Sz + 4]; // alpha;//polarizability;unit:cubic angstrom//极化率
                fint >> h_thermal.species_chara[i * SPCH_Sz + 5]; // Zrot_298;//rotational relaxation collision Zrot at 298K;
                break;
            }
        }
        // // // species_chara pre-process
        h_thermal.species_chara[i * SPCH_Sz + 6] *= 1e-3; // kg/mol
        h_thermal.Wi[i] = h_thermal.species_chara[i * SPCH_Sz + Wi];
        h_thermal._Wi[i] = _DF(1.0) / h_thermal.Wi[i];
        h_thermal.Ri[i] = Ru / h_thermal.Wi[i];
        h_thermal.species_chara[i * SPCH_Sz + 7] = 0;   // Miu[i];
        h_thermal.species_chara[i * SPCH_Sz + SID] = i; // SID;
    }
    fint.close();

    h_thermal.xi_in = middle::MallocHost<real_t>(h_thermal.xi_in, NUM_SPECIES, q);
    h_thermal.xi_out = middle::MallocHost<real_t>(h_thermal.xi_out, NUM_SPECIES, q);
    std::memcpy(h_thermal.xi_in, h_thermal.species_ratio_in, NUM_SPECIES * sizeof(real_t));
    std::memcpy(h_thermal.xi_out, h_thermal.species_ratio_out, NUM_SPECIES * sizeof(real_t));

    // transfer mole fraction to mess fraction
    get_yi(h_thermal.species_ratio_in, h_thermal.Wi);
    get_yi(h_thermal.species_ratio_out, h_thermal.Wi);
}
