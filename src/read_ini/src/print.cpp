#include <iostream>
#include "../setupini.h"

// =======================================================
// =======================================================
void Setup::print()
{
    if (mach_shock)
    { // States initializing
        printf("blast_type: %d and blast_center(x = %.6lf , y = %.6lf , z = %.6lf).\n", ini.blast_type, ini.blast_center_x, ini.blast_center_y, ini.blast_center_z);
        printf(" shock Mach number = %lf to reinitialize fluid states upstream the shock.\n", ini.Ma);
        printf("  propagation speed of shock = %lf, normalized time tau_H(bubble_diameter/shock_propagation_speed)= %lf.\n", ini.Ma * ini.blast_c_out, ini.tau_H);
        printf("  states of   upstream:     (P = %.6lf, T = %.6lf, rho = %.6lf, u = %.6lf, v = %.6lf, w = %.6lf).\n", ini.blast_pressure_in, ini.blast_T_in, ini.blast_density_in, ini.blast_u_in, ini.blast_v_in, ini.blast_w_in);
        printf("  states of downstream:     (P = %.6lf, T = %.6lf, rho = %.6lf, u = %.6lf, v = %.6lf, w = %.6lf).\n", ini.blast_pressure_out, ini.blast_T_out, ini.blast_density_out, ini.blast_u_out, ini.blast_v_out, ini.blast_w_out);
    }
    // 后接流体状态输出
    if (1 < NumFluid)
    {
        if (myRank == 0)
            std::cout << "<---------------------------------------------------> \n";

        printf("Extending width: width_xt                                : %lf\n", width_xt);
        printf("Ghost-fluid update width: width_hlf                      : %lf\n", width_hlf);
        printf("cells' volume less than this vule will be mixed          : %lf\n", mx_vlm);
        printf("cells' volume less than states updated based on mixed    : %lf\n", ext_vlm);
        printf("half-width of level set narrow band                      : %lf\n", BandforLevelset);
        printf("Number of fluids                                         : %zu\n", BlSz.num_fluids);
        for (size_t n = 0; n < NumFluid; n++)
        { // 0: phase_indicator, 1: gamma, 2: A, 3: B, 4: rho0, 5: R_0, 6: lambda_0, 7: a(rtificial)s(peed of)s(ound)
            printf("fluid[%zu]: %s, characteristics(Material, Phase_indicator, Gamma, A, B, Rho0, R_0, Lambda_0, artificial speed of sound): \n", n, Fluids_name[n].c_str());
            printf("  %f,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f,   %.6f\n", material_props[n][0],
                   material_props[n][1], material_props[n][2], material_props[n][3], material_props[n][4],
                   material_props[n][5], material_props[n][6], material_props[n][7], material_props[n][8]);
        }
    }

#ifdef COP

    if (myRank == 0)
        std::cout << "<---------------------------------------------------> \n";

    std::cout << species_name.size() << " species mole/mass fraction(in/out): " << std::endl;
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        std::cout << "species[" << n << "]=" << std::left << std::setw(10) << species_name[n]
                  << std::setw(5) << h_thermal.xi_in[n] << std::setw(5) << h_thermal.xi_out[n]
                  << std::setw(15) << h_thermal.species_ratio_in[n] << std::setw(15) << h_thermal.species_ratio_out[n] << std::endl;
    }

    if (Visc)
    {
        if (myRank == 0)
            std::cout << "<---------------------------------------------------> \n";

        printf("Viscisity characteristics(geo, epsilon_kB, L-J collision diameter, dipole moment, polarizability, Zort_298, molar mass): \n");
        for (size_t n = 0; n < NUM_SPECIES; n++)
        {
            printf("species[%zd]: %s,    %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf,   %.6lf\n", n,
                   species_name[n].c_str(), h_thermal.species_chara[n * SPCH_Sz + 0],
                   h_thermal.species_chara[n * SPCH_Sz + 1], h_thermal.species_chara[n * SPCH_Sz + 2],
                   h_thermal.species_chara[n * SPCH_Sz + 3], h_thermal.species_chara[n * SPCH_Sz + 4],
                   h_thermal.species_chara[n * SPCH_Sz + 5], h_thermal.species_chara[n * SPCH_Sz + 6]);
        }
    }
#endif // end COP

    if (myRank == 0)
        std::cout << "<---------------------------------------------------> \n";

    printf("Start time: %.6lf and End time: %.6lf                        \n", StartTime, EndTime);
#ifdef USE_MPI // end USE_MPI
    {          // print information about current setup
        std::cout << "MPI rank mesh setup as below: \n";
        std::cout << "   Global resolution of MPI World: " << BlSz.X_inner * BlSz.mx << " x " << BlSz.Y_inner * BlSz.my << " x " << BlSz.Z_inner * BlSz.mz << "\n";
        std::cout << "   Local  resolution of one Rank : " << BlSz.X_inner << " x " << BlSz.Y_inner << " x " << BlSz.Z_inner << "\n";
        std::cout << "   MPI Cartesian topology        : " << BlSz.mx << " x " << BlSz.my << " x " << BlSz.mz << std::endl;
    }
#else
    std::cout << "Resolution of Domain:                 " << BlSz.X_inner << " x " << BlSz.Y_inner << " x " << BlSz.Z_inner << "\n";
#endif // end USE_MPI
    printf("GhostWidth Cells: Bx, By, Bz:         %d,  %d,  %d\n", BlSz.Bwidth_X, BlSz.Bwidth_Y, BlSz.Bwidth_Z);
    printf("Block size:   bx, by, bz, Dt:         %zu,  %zu,  %zu,  %zu\n", BlSz.dim_block_x, BlSz.dim_block_y, BlSz.dim_block_z, BlSz.BlockSize);
    printf("XYZ dir Domain size:                  %1.3lf x %1.3lf x %1.3lf\n", BlSz.Domain_length, BlSz.Domain_width, BlSz.Domain_height);
    printf("Difference steps: dx, dy, dz:         %lf, %lf, %lf\n", BlSz.dx, BlSz.dy, BlSz.dz);

    if (myRank == 0)
        std::cout << "<---------------------------------------------------> \n";
} // Setup::print
