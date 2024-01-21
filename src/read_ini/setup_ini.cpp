#include <cmath>
#include <fstream>
#include <iostream>

#include "setupini.h"
#include "mixture.hpp"
#include "fworkdir.hpp"
#include "rangeset.hpp"

#include "../solver_Ini/Mixing_device.h"
#include "../solver_Reconstruction/viscosity/Visc_device.h"

// =======================================================
// // // struct Setup Member function definitions
// =======================================================
Setup::Setup(int argc, char **argv, int rank, int nranks) : myRank(rank), nRanks(nranks), apa(argc, argv)
{
#ifdef USE_MPI // Create MPI session if MPI enabled
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif         // USE_MPI
    ReWrite(); // rewrite parameters using appended options
    // // sycl::queue construction
    q = sycl::queue(sycl::platform::get_platforms()[DeviceSelect[1]].get_devices()[DeviceSelect[2]]);
    // // get Work directory
    WorkDir = getWorkDir(std::string(argv[0]), "XFLUIDS");
    // // NOTE: read_grid
    grid = Gridread(q, BlSz, WorkDir + "/" + std::string(INI_SAMPLE), myRank, nRanks);

    // /*begin runtime read , fluid && compoent characteristics set*/
    ReadSpecies();
    if (ReactSources)
        ReadReactions();
        /*end runtime read*/

// read && caculate coffes for visicity
#if Visc
    GetFitCoefficient();
#endif

    init(); // Ini
#ifdef USE_MPI
    mpiTrans = new MpiTrans(BlSz, Boundarys);
    mpiTrans->communicator->synchronize();
#endif // end USE_MPI
    std::cout << "Selected Device: " << middle::DevInfo(q) << "  of rank: " << myRank << std::endl;

    CpyToGPU();
} // Setup::Setup end

// =======================================================
// // // Initialize struct member using json value
// =======================================================
void Setup::ReadIni()
{
    /* initialize RUN parameters */
    nStepmax = nStepmax_json;
    // // initialize output time arrays
    for (size_t nn = 0; nn < OutTimeArrays_json.size(); nn++)
    {
        std::vector<std::string> temp = Stringsplit(OutTimeArrays_json[nn], ':');
        std::vector<std::string> tempt = Stringsplit(temp[0], ';');
        std::vector<real_t> temp0 = Stringsplit<real_t>(tempt[1], '*');
        temp[1].erase(0, 2), temp[1].erase(temp[1].size() - 1, 1);
        std::vector<std::string> temp1 = Stringsplit(temp[1], ';');

        real_t tn_b = stod(tempt[0]);
        for (size_t tn = 1; tn <= size_t(temp0[0]); tn++)
        {
            OutFmt this_time(tn * temp0[1] + tn_b);
            this_time._C = apa.match(temp1, "-C");
            this_time._V = apa.match(temp1, "-V");
            this_time._P = apa.match(temp1, "-P");
            OutTimeStamps.push_back(this_time);
        }

        // // 200*1.0E-6
        // std::vector<std::string> temp = Stringsplit(OutTimeArrays_json[nn], ':');
        // std::vector<std::string> temp0 = Stringsplit(temp[0], '*');
        // std::vector<std::string> times = Stringsplit(temp0[1], 'E');
        // std::vector<int> time_it = Stringsplit<int>(times[0], '.');
        // interval = time_it[0] + real_t(time_it[1]) / _DF(10.0);
        // real_t bak = real_t(stoi(times[1]));
        // interval = std::pow(interval, bak);
        // temp[1].erase(0, 2), temp[1].erase(temp[1].size() - 1, 1);
        // std::vector<std::string> temp1 = Stringsplit(temp[1], ';');
        // std::vector<std::string> temp2 = apa.match(temp1, "-C", ':');
        // for (size_t tn = 0; tn < stoi(temp0[0]); tn++)
        // {
        //     OutFmt this_time(tn * interval + tn_b);
        //     this_time._V = apa.match(temp1, "-V");
        //     this_time._P = apa.match(temp1, "-P");
        //     OutTimeStamps.push_back(this_time);
        // }
        // tn_b = stoi(temp0[0]) * interval;
    }

    // // insert specific output time stamps
    for (size_t nn = 0; nn < OutTimeStamps_json.size(); nn++)
    {
        std::vector<std::string> temp = Stringsplit(OutTimeStamps_json[nn], ':');
        real_t the_time = stod(temp[0]);
        temp[1].erase(0, 2), temp[1].erase(temp[1].size() - 1, 1);
        std::vector<std::string> temp1 = Stringsplit(temp[1], ';');
        if (!std::empty(OutTimeArrays_json))
            for (int tn = 0; tn < OutTimeStamps.size(); tn++)
            {
                bool is_pos = OutTimeStamps[tn].time < the_time;
                is_pos = OutTimeStamps[tn + 1].time > the_time;
                if (is_pos)
                {
                    OutTimeStamps.emplace(OutTimeStamps.begin() + (tn + 1), OutFmt(the_time));
                    OutTimeStamps[tn + 1]._C = apa.match(temp1, "-C");
                    OutTimeStamps[tn + 1]._V = apa.match(temp1, "-V");
                    OutTimeStamps[tn + 1]._P = apa.match(temp1, "-P");
                    break;
                }
            }
        else
        {
            OutFmt this_time(the_time);
            this_time._C = apa.match(temp1, "-C");
            this_time._V = apa.match(temp1, "-V");
            this_time._P = apa.match(temp1, "-P");
            OutTimeStamps.push_back(this_time);
        }
    }

    /* initialize React sources  */
    BlSz.RSources = ReactSources;
    /* initialize MPI parameters */
    BlSz.mx = mx_json, BlSz.my = my_json, BlSz.mz = mz_json;
    // initial rank postion to zero, will be changed in MpiTrans
    BlSz.myMpiPos_x = 0, BlSz.myMpiPos_y = 0, BlSz.myMpiPos_z = 0;
    // for sycl::queue construction and device select
    DeviceSelect = DeviceSelect_json; // [0]:number of alternative devices [1]:platform_id [2]:device_id.

    /* initialize MESH parameters */
    // // initialize reference parameters, for calulate coordinate while readgrid
    BlSz.LRef = Refs[0]; // reference length
    BlSz.CFLnumber = CFLnumber_json;
    BlSz.DimX = Dimensions[0], BlSz.DimY = Dimensions[1], BlSz.DimZ = Dimensions[2];

    // // read block size settings
    BlSz.X_inner = Inner[0], BlSz.Y_inner = Inner[1], BlSz.Z_inner = Inner[2];
    BlSz.Bwidth_X = Bwidth[0], BlSz.Bwidth_Y = Bwidth[1], BlSz.Bwidth_Z = Bwidth[2];
    BlSz.Domain_xmin = Domain_medg[0], BlSz.Domain_ymin = Domain_medg[1], BlSz.Domain_zmin = Domain_medg[2];
    BlSz.Domain_length = DOMAIN_Size[0], BlSz.Domain_width = DOMAIN_Size[1], BlSz.Domain_height = DOMAIN_Size[2];

    // // read GPU block size settings
    BlSz.BlockSize = BlockSize_json;
    BlSz.dim_block_x = dim_block_x_json, BlSz.dim_block_y = dim_block_y_json, BlSz.dim_block_z = dim_block_z_json;

    // // Simple Boundary settings
    // // // Inflow = 0,Outflow = 1,Symmetry = 2,Periodic = 3,nslipWall = 4
    Boundarys[0] = static_cast<BConditions>(Boundarys_json[0]), Boundarys[1] = static_cast<BConditions>(Boundarys_json[1]);
    Boundarys[2] = static_cast<BConditions>(Boundarys_json[2]), Boundarys[3] = static_cast<BConditions>(Boundarys_json[3]);
    Boundarys[4] = static_cast<BConditions>(Boundarys_json[4]), Boundarys[5] = static_cast<BConditions>(Boundarys_json[5]);

    /* initialize fluid flow parameters */
    // // Ini Flow Field dynamic states
    ini.Ma = Ma_json; // shock Mach number
    ini.blast_type = blast_type;
    ini.blast_center_x = blast_pos[0], ini.blast_center_y = blast_pos[1], ini.blast_center_z = blast_pos[2];
    // // Bubble size and shape
    ini.xa = xa_json, ini.yb = yb_json, ini.zc = zc_json;
    // // upstream of blast
    ini.blast_density_in = blast_upstates[0], ini.blast_pressure_in = blast_upstates[1];
    ini.blast_T_in = blast_upstates[2], ini.blast_u_in = blast_upstates[3], ini.blast_v_in = blast_upstates[4], ini.blast_w_in = blast_upstates[5];
    // // downstream of blast
    ini.blast_density_out = blast_downstates[0], ini.blast_pressure_out = blast_downstates[1];
    ini.blast_T_out = blast_downstates[2], ini.blast_u_out = blast_downstates[3], ini.blast_v_out = blast_downstates[4], ini.blast_w_out = blast_downstates[5];
#ifdef COP
    // // states inside mixture bubble
    ini.cop_type = cop_type;
    ini.cop_center_x = cop_pos[0], ini.cop_center_y = cop_pos[1], ini.cop_center_z = cop_pos[2];
    ini.cop_density_in = cop_instates[0], ini.cop_pressure_in = cop_instates[1], ini.cop_T_in = cop_instates[2];
#else  // no COP
#endif // end COP
}

// =======================================================
// // // Using option parameters appended executable file
// =======================================================
void Setup::ReWrite()
{
    // =======================================================
    // // for json file read
    ReadIni(); // Initialize parameters from json

    // // rewrite X_inner, Y_inner, Z_inner, nStpeMax(if given)
    std::vector<int> Inner_size = apa.match<int>("-run");
    if (!std::empty(Inner_size))
    {
        BlSz.X_inner = Inner_size[0], BlSz.Y_inner = Inner_size[1], BlSz.Z_inner = Inner_size[2];
        // // rewrite DimX, DimY, DimZ for computational dimensions
        BlSz.DimX = bool(Inner_size[0]), BlSz.DimY = bool(Inner_size[1]), BlSz.DimZ = bool(Inner_size[2]);
        if (4 == Inner_size.size())
            nStepmax = Inner_size[3];
    }

    // // rewrite dim_block_x, dim_block_y, dim_block_z
    std::vector<int> blkapa = apa.match<int>("-blk");
    if (!std::empty(blkapa))
    {
        BlSz.dim_block_x = blkapa[0], BlSz.dim_block_y = blkapa[1], BlSz.dim_block_z = blkapa[2];
        if (4 == blkapa.size())
            BlSz.BlockSize = blkapa[3];
    }

    // // rewrite mx, my, mz for MPI
    std::vector<int> mpiapa = apa.match<int>("-mpi");
    if (!std::empty(mpiapa))
        BlSz.mx = mpiapa[0], BlSz.my = mpiapa[1], BlSz.mz = mpiapa[2];

    // // open mpi threads debug;
    std::vector<int> mpidbg = apa.match<int>("-mpidbg");
    if (!std::empty(mpidbg))
    {
        // bool a = false;
        // while (!a)
        {
            sleep(10);
        };
        // #ifdef USE_MPI
        //         a = mpiTrans->BocastTrue(a);
        // #endif // end USE_MPI
    }

    // // accelerator_selector device;
    std::vector<int> devapa = apa.match<int>("-dev");
    if (!std::empty(devapa))
        DeviceSelect = devapa;
#if defined(DEFINED_OPENSYCL)
    DeviceSelect[2] += myRank % DeviceSelect[0];
#else  // for oneAPI
    DeviceSelect[1] += myRank % DeviceSelect[0];
#endif // end
}

// =======================================================
// =======================================================
void Setup::init()
{ // set other parameters
    BlSz.DimS = BlSz.DimX + BlSz.DimY + BlSz.DimZ;
    BlSz.DimX_t = BlSz.DimX, BlSz.DimY_t = BlSz.DimY, BlSz.DimZ_t = BlSz.DimZ;

    BlSz.X_inner = BlSz.DimX ? BlSz.X_inner : 1;
    BlSz.Y_inner = BlSz.DimY ? BlSz.Y_inner : 1;
    BlSz.Z_inner = BlSz.DimZ ? BlSz.Z_inner : 1;

    BlSz.Bwidth_X = BlSz.DimX ? BlSz.Bwidth_X : 0;
    BlSz.Bwidth_Y = BlSz.DimY ? BlSz.Bwidth_Y : 0;
    BlSz.Bwidth_Z = BlSz.DimZ ? BlSz.Bwidth_Z : 0;

    BlSz.dim_block_x = BlSz.DimX ? BlSz.dim_block_x : 1;
    BlSz.dim_block_y = BlSz.DimY ? BlSz.dim_block_y : 1;
    BlSz.dim_block_z = BlSz.DimZ ? BlSz.dim_block_z : 1;

    BlSz.Domain_length = BlSz.DimX ? BlSz.Domain_length : 1.0;
    BlSz.Domain_width = BlSz.DimY ? BlSz.Domain_width : 1.0;
    BlSz.Domain_height = BlSz.DimZ ? BlSz.Domain_height : 1.0;

    BlSz.Domain_xmax = BlSz.Domain_xmin + BlSz.Domain_length;
    BlSz.Domain_ymax = BlSz.Domain_ymin + BlSz.Domain_width;
    BlSz.Domain_zmax = BlSz.Domain_zmin + BlSz.Domain_height;

    BlSz.dx = BlSz.DimX ? BlSz.Domain_length / real_t(BlSz.mx * BlSz.X_inner) : _DF(1.0);
    BlSz.dy = BlSz.DimY ? BlSz.Domain_width / real_t(BlSz.my * BlSz.Y_inner) : _DF(1.0);
    BlSz.dz = BlSz.DimZ ? BlSz.Domain_height / real_t(BlSz.mz * BlSz.Z_inner) : _DF(1.0);

    // // maximum number of total cells
    BlSz.Xmax = BlSz.DimX ? (BlSz.X_inner + 2 * BlSz.Bwidth_X) : 1;
    BlSz.Ymax = BlSz.DimY ? (BlSz.Y_inner + 2 * BlSz.Bwidth_Y) : 1;
    BlSz.Zmax = BlSz.DimZ ? (BlSz.Z_inner + 2 * BlSz.Bwidth_Z) : 1;

    BlSz.dl = BlSz.dx + BlSz.dy + BlSz.dz;
    if (BlSz.DimX)
        BlSz.dl = std::min(BlSz.dl, BlSz.dx);
    if (BlSz.DimY)
        BlSz.dl = std::min(BlSz.dl, BlSz.dy);
    if (BlSz.DimZ)
        BlSz.dl = std::min(BlSz.dl, BlSz.dz);

    BlSz.offx = (_DF(0.5) - BlSz.Bwidth_X + BlSz.myMpiPos_x * BlSz.X_inner) * BlSz.dx + BlSz.Domain_xmin;
    BlSz.offy = (_DF(0.5) - BlSz.Bwidth_Y + BlSz.myMpiPos_y * BlSz.Y_inner) * BlSz.dy + BlSz.Domain_ymin;
    BlSz.offz = (_DF(0.5) - BlSz.Bwidth_Z + BlSz.myMpiPos_z * BlSz.Z_inner) * BlSz.dz + BlSz.Domain_zmin;

    BlSz._dx = _DF(1.0) / BlSz.dx, BlSz._dy = _DF(1.0) / BlSz.dy, BlSz._dz = _DF(1.0) / BlSz.dz, BlSz._dl = _DF(1.0) / BlSz.dl;

    // // bubble size: two cell boundary
    ini._xa2 = _DF(1.0) / (ini.xa * ini.xa);
    ini._yb2 = _DF(1.0) / (ini.yb * ini.yb);
    ini._zc2 = _DF(1.0) / (ini.zc * ini.zc);
    ini.C = C_json * BlSz.mx * BlSz.X_inner;

    // // DataBytes set
    bytes = BlSz.Xmax * BlSz.Ymax * BlSz.Zmax * sizeof(real_t), cellbytes = Emax * bytes;

    // // solving system
    BlSz.num_species = NUM_SPECIES;
    BlSz.num_cop = NUM_SPECIES - 1;
#ifdef GhostSpecies
    BlSz.num_species += (-1);
#else
#endif
    BlSz.num_rea = NUM_REA;
    BlSz.num_eqn = BlSz.num_species + 4;

    // Viscosity Kernel limiters
    BlSz.Dim_limiter = std::min(std::max(Dim_limiter_json, _DF(0.0)), _DF(1.0));
    BlSz.Yil_limiter = std::max(std::max(BlSz._dx, BlSz._dy), BlSz._dz) * std::min(std::max(Yil_limiter_json, _DF(0.0)), _DF(1.0));
    // // Initialize shock base mach number
    mach_shock = Mach_Shock();
    // // Initialize shock base mach number
    if (0 == myRank)
        print();
} // Setup::init

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
void Setup::ReadReactions()
{
    h_react.Nu_f_ = middle::MallocHost<int>(h_react.Nu_f_, NUM_REA * NUM_SPECIES, q);
    h_react.Nu_b_ = middle::MallocHost<int>(h_react.Nu_b_, NUM_REA * NUM_SPECIES, q);
    h_react.Nu_d_ = middle::MallocHost<int>(h_react.Nu_d_, NUM_REA * NUM_SPECIES, q);
    h_react.React_ThirdCoef = middle::MallocHost<real_t>(h_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q);
    h_react.Rargus = middle::MallocHost<real_t>(h_react.Rargus, NUM_REA * 6, q);
    h_react.react_type = middle::MallocHost<int>(h_react.react_type, NUM_REA * 2, q);
    h_react.third_ind = middle::MallocHost<int>(h_react.third_ind, NUM_REA, q);

    char Key_word[128];
    std::string rpath = WorkDir + std::string(RFile) + "/reaction_list.dat";
    std::ifstream fint(rpath);
    if (fint.good())
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
            { // reaction rate constant parameters, A, B, E for Arrhenius law
                for (int i = 0; i < NUM_REA; i++)
                {
                    fint >> h_react.Rargus[i * 6 + 0] >> h_react.Rargus[i * 6 + 1] >> h_react.Rargus[i * 6 + 2];
                    if (BackArre)
                        fint >> h_react.Rargus[i * 6 + 3] >> h_react.Rargus[i * 6 + 4] >> h_react.Rargus[i * 6 + 5];
                } //-----------------*backwardArrhenius------------------//
            }
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
            if (h_react.Nu_f_[i * NUM_SPECIES + j] > 0 || h_react.Nu_b_[i * NUM_SPECIES + j] > 0)
                reaction_list[j].push_back(i);
    }

    if (0 == myRank)
        printf("\n%d Reactions been actived                                     \n", NUM_REA);
    bool react_error = false;
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

        if (0 == myRank)
        { // output this reaction kinetic
            if (i + 1 < 10)
                std::cout << "Reaction:" << i + 1 << "  ";
            else
                std::cout << "Reaction:" << i + 1 << " ";
            int numreactant = 0, numproduct = 0;
            // // output reactant
            for (int j = 0; j < NUM_SPECIES; ++j)
                numreactant += h_react.Nu_f_[i * NUM_SPECIES + j];

            if (numreactant == 0)
                std::cout << "0	  <-->  ";
            else
            {
                for (int j = 0; j < NUM_SPECIES; j++)
                {
                    if (h_react.Nu_f_[i * NUM_SPECIES + j] > 0)
                        std::cout << h_react.Nu_f_[i * NUM_SPECIES + j] << " " << species_name[j] << " + ";

                    if (NUM_COP == j)
                    {
                        if (h_react.React_ThirdCoef[i * NUM_SPECIES + j] > _DF(0.0))
                            std::cout << " M ";
                        std::cout << "  <-->  ";
                    }
                }
            }
            // // output product
            for (int j = 0; j < NUM_SPECIES; ++j)
                numproduct += h_react.Nu_b_[i * NUM_SPECIES + j];
            if (numproduct == 0)
                std::cout << "0	";
            else
            {
                for (int j = 0; j < NUM_SPECIES; j++)
                {
                    if (h_react.Nu_b_[i * NUM_SPECIES + j] > 0)
                        std::cout << h_react.Nu_b_[i * NUM_SPECIES + j] << " " << species_name[j] << " + ";

                    if (NUM_COP == j)
                    {
                        if (h_react.React_ThirdCoef[i * NUM_SPECIES + j] > _DF(0.0))
                            std::cout << " M ";
                    }
                }
            }
            std::cout << " with forward rate: " << h_react.Rargus[i * 6 + 0] << " " << h_react.Rargus[i * 6 + 1] << " " << h_react.Rargus[i * 6 + 2];
            //-----------------*backwardArrhenius------------------//
            if (BackArre)
                std::cout << ", backward rate: " << h_react.Rargus[i * 6 + 3] << " " << h_react.Rargus[i * 6 + 4] << " " << h_react.Rargus[i * 6 + 5];
            std::cout << std::endl;

            if (1 == h_react.third_ind[i])
            {
                std::cout << " Third Body coffcients:";
                for (int j = 0; j < NUM_SPECIES - 1; j++)
                    std::cout << "  " << species_name[j] << ":" << h_react.React_ThirdCoef[i * NUM_SPECIES + j];
                std::cout << "\n";
            }
        }

        react_error || ReactionType(0, i, h_react.Nu_f_, h_react.Nu_b_); // forward reaction kinetic
        react_error || ReactionType(1, i, h_react.Nu_b_, h_react.Nu_f_); // backward reaction kinetic
    }

    // if (react_error)
    //     exit(EXIT_FAILURE);
}
// =======================================================
// =======================================================
/**
 * @brief determine types of reaction:
 * 		1: "0 -> A";		2: "A -> 0";		    3: "A -> B";		    4: "A -> B + C";
 * 	    5: "A -> A + B"     6: "A + B -> 0"		    7: "A + B -> C"		    8: "A + B -> C + D"
 *      9: "A + B -> A";	10: "A + B -> A + C"    11: "2A -> B"		    12: "A -> 2B"
 *      13: "2A -> B + C"	14: "A + B -> 2C"	    15: "2A + B -> C + D"   16: "2A -> 2C + D"
 * @note  only a few simple reaction has analytical solution, for other cases one can use quasi-statical-state-approxiamte(ChemeQ2)
 * @param i: index of the reaction
 * @param flag: 0: forward reaction; 1: backward reaction
 * @param Nuf,Nub: forward and backward reaction matrix
 * @return true: has unsupported reaction kinetic type
 * @return false: all supported reaction kinetic type
 */
bool Setup::ReactionType(int flag, int i, int *Nuf, int *Nub)
{
    std::string error_str;
    std::vector<int> forward_list, backward_list;
    if (flag == 0)
    {
        error_str = "forward ";
        forward_list = reactant_list[i], backward_list = product_list[i];
    }
    else
    {
        error_str = "backward";
        forward_list = product_list[i], backward_list = reactant_list[i];
    }

    /**
     * @brief: reaction type
     * @note: the support is not complete
     */
    h_react.react_type[i * 2 + flag] = 0;
    int Od_Rec = 0, Od_Pro = 0, Num_Repeat = 0; // // the order of the reaction
    // // loop all species in reaction "i"
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
    // // get reaction type
    switch (Od_Rec)
    {
    case 0: // // 0th-order
        h_react.react_type[i * 2 + flag] = 1;
        break;
    case 1: // // 1st-order
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
    case 2: // // 2nd-order
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
            if ((Nub[i * NUM_SPECIES + backward_list[0]] == 2) && !(Num_Repeat))
                h_react.react_type[i * 2 + flag] = 16;
        }
        break;
    case 3: // // 3rd-order
        if (Od_Pro == 2)
        {
            if (Nuf[i * NUM_SPECIES + forward_list[0]] == 2 || Nuf[i * NUM_SPECIES + forward_list[1]] == 2 && Nuf[i * NUM_SPECIES + backward_list[0]] == 1 && !(Num_Repeat))
                h_react.react_type[i * 2 + flag] = 15;
        }
        break;
    }
    if (h_react.react_type[i * 2 + flag] == 0)
    {
        if (myRank == 0)
            std::cout << " Note:no analytical solutions for" << error_str << " reaction of the " << i + 1 << " kinetic.\n";

        return true;
    }

    return false;
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
            if (k <= j)                                                                    // upper triangle
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
        theo << "," << species_name[k] << "_Cp_NASA,";
        theo << "," << species_name[k] << "_Cp_JANAF,";
    }
    for (size_t k = 0; k < species_name.size(); k++)
    {
        theo << "," << species_name[k] << "_hi_NASA,";
        theo << "," << species_name[k] << "_hi_JANAF,";
    }
    for (size_t k = 0; k < species_name.size(); k++)
    {
        theo << "," << species_name[k] << "_S_NASA,";
        theo << "," << species_name[k] << "_S_JANAF,";
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
        out << ",visc_" << species_name[k] << ",";
        out << ",furier_" << species_name[k] << ",";
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

// =======================================================
// =======================================================
void Setup::CpyToGPU()
{
#ifdef USE_MPI
    mpiTrans->communicator->synchronize();
    if (0 == myRank)
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

    if (ReactSources)
    {
        d_react.Nu_f_ = middle::MallocDevice<int>(d_react.Nu_f_, NUM_REA * NUM_SPECIES, q);
        d_react.Nu_b_ = middle::MallocDevice<int>(d_react.Nu_b_, NUM_REA * NUM_SPECIES, q);
        d_react.Nu_d_ = middle::MallocDevice<int>(d_react.Nu_d_, NUM_REA * NUM_SPECIES, q);
        d_react.react_type = middle::MallocDevice<int>(d_react.react_type, NUM_REA * 2, q);
        d_react.third_ind = middle::MallocDevice<int>(d_react.third_ind, NUM_REA, q);
        d_react.React_ThirdCoef = middle::MallocDevice<real_t>(d_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q);
        d_react.Rargus = middle::MallocDevice<real_t>(d_react.Rargus, NUM_REA * 6, q);

        middle::MemCpy<int>(d_react.Nu_f_, h_react.Nu_f_, NUM_REA * NUM_SPECIES, q);
        middle::MemCpy<int>(d_react.Nu_b_, h_react.Nu_b_, NUM_REA * NUM_SPECIES, q);
        middle::MemCpy<int>(d_react.Nu_d_, h_react.Nu_d_, NUM_REA * NUM_SPECIES, q);
        middle::MemCpy<int>(d_react.react_type, h_react.react_type, NUM_REA * 2, q);
        middle::MemCpy<int>(d_react.third_ind, h_react.third_ind, NUM_REA, q);
        middle::MemCpy<real_t>(d_react.React_ThirdCoef, h_react.React_ThirdCoef, NUM_REA * NUM_SPECIES, q);
        middle::MemCpy<real_t>(d_react.Rargus, h_react.Rargus, NUM_REA * 6, q);

        int reaction_list_size = 0;
        h_react.rns = middle::MallocHost<int>(h_react.rns, NUM_SPECIES, q);
        for (size_t i = 0; i < NUM_SPECIES; i++)
            h_react.rns[i] = reaction_list[i].size(), reaction_list_size += h_react.rns[i];

        int *h_reaction_list, *d_reaction_list;
        h_reaction_list = middle::MallocHost<int>(h_reaction_list, reaction_list_size, q);
        h_react.reaction_list = middle::MallocHost2D<int>(h_reaction_list, NUM_SPECIES, h_react.rns, q);
        for (size_t i = 0; i < NUM_SPECIES; i++)
            if (h_react.rns[i] > 0)
                std::memcpy(h_react.reaction_list[i], &(reaction_list[i][0]), sizeof(int) * h_react.rns[i]);
        d_reaction_list = middle::MallocDevice<int>(d_reaction_list, reaction_list_size, q);
        middle::MemCpy<int>(d_reaction_list, h_reaction_list, reaction_list_size, q);
        d_react.reaction_list = middle::MallocDevice2D<int>(d_reaction_list, NUM_SPECIES, h_react.rns, q);

        h_react.rts = middle::MallocHost<int>(h_react.rts, NUM_REA, q);
        h_react.pls = middle::MallocHost<int>(h_react.pls, NUM_REA, q);
        h_react.sls = middle::MallocHost<int>(h_react.sls, NUM_REA, q);
        int rts_size = 0, pls_size = 0, sls_size = 0;
        for (size_t i = 0; i < NUM_REA; i++)
        {
            h_react.rts[i] = reactant_list[i].size(), rts_size += h_react.rts[i];
            h_react.pls[i] = product_list[i].size(), pls_size += h_react.pls[i];
            h_react.sls[i] = species_list[i].size(), sls_size += h_react.sls[i];
        }
        d_react.rns = middle::MallocDevice<int>(d_react.rns, NUM_SPECIES, q);
        d_react.rts = middle::MallocDevice<int>(d_react.rts, NUM_REA, q);
        d_react.pls = middle::MallocDevice<int>(d_react.pls, NUM_REA, q);
        d_react.sls = middle::MallocDevice<int>(d_react.sls, NUM_REA, q);
        middle::MemCpy<int>(d_react.rns, h_react.rns, NUM_SPECIES, q);
        middle::MemCpy<int>(d_react.rts, h_react.rts, NUM_REA, q);
        middle::MemCpy<int>(d_react.pls, h_react.pls, NUM_REA, q);
        middle::MemCpy<int>(d_react.sls, h_react.sls, NUM_REA, q);

        int *h_reactant_list, *h_product_list, *h_species_list, *d_reactant_list, *d_product_list, *d_species_list;
        h_reactant_list = middle::MallocHost<int>(h_reactant_list, rts_size, q);
        h_product_list = middle::MallocHost<int>(h_product_list, pls_size, q);
        h_species_list = middle::MallocHost<int>(h_species_list, sls_size, q);
        h_react.reactant_list = middle::MallocHost2D<int>(h_reactant_list, NUM_REA, h_react.rts, q);
        h_react.product_list = middle::MallocHost2D<int>(h_product_list, NUM_REA, h_react.pls, q);
        h_react.species_list = middle::MallocHost2D<int>(h_species_list, NUM_REA, h_react.sls, q);

        for (size_t i = 0; i < NUM_REA; i++)
        {
            std::memcpy(h_react.reactant_list[i], &(reactant_list[i][0]), sizeof(int) * h_react.rts[i]);
            std::memcpy(h_react.product_list[i], &(product_list[i][0]), sizeof(int) * h_react.pls[i]);
            std::memcpy(h_react.species_list[i], &(species_list[i][0]), sizeof(int) * h_react.sls[i]);
        }

        d_reactant_list = middle::MallocDevice<int>(d_reactant_list, rts_size, q);
        d_product_list = middle::MallocDevice<int>(d_product_list, pls_size, q);
        d_species_list = middle::MallocDevice<int>(d_species_list, sls_size, q);
        middle::MemCpy<int>(d_reactant_list, h_reactant_list, rts_size, q);
        middle::MemCpy<int>(d_product_list, h_product_list, pls_size, q);
        middle::MemCpy<int>(d_species_list, h_species_list, sls_size, q);
        d_react.reactant_list = middle::MallocDevice2D<int>(d_reactant_list, NUM_REA, h_react.rts, q);
        d_react.product_list = middle::MallocDevice2D<int>(d_product_list, NUM_REA, h_react.pls, q);
        d_react.species_list = middle::MallocDevice2D<int>(d_species_list, NUM_REA, h_react.sls, q);

        // std::cout << "\n";
        // for (size_t i = 0; i < NUM_SPECIES; i++)
        // {
        //     for (size_t j = 0; j < h_react.rns[i]; j++)
        //         std::cout << h_react.reaction_list[i][j] << " ";
        //     std::cout << ", ";
        // }
        // std::cout << "\n";

        // q.submit([&](sycl::handler &h) { // PARALLEL;
        //      sycl::stream stream_ct1(64 * 1024, 80, h);
        //      h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
        //          for (size_t i = 0; i < NUM_SPECIES; i++)
        //          {
        //              for (size_t j = 0; j < d_react.rns[i]; j++)
        //              {
        //                  stream_ct1 << d_react.reaction_list[i][j] << " ";
        //              }
        //              stream_ct1 << ", ";
        //          }
        //          stream_ct1 << "\n";
        //      });
        //  })
        //     .wait();

        // for (size_t i = 0; i < NUM_REA; i++)
        // {
        //     for (size_t j = 0; j < h_react.rts[i]; j++)
        //         std::cout << h_react.reactant_list[i][j] << " ";
        //     std::cout << ", ";
        // }
        // std::cout << "\n";

        // q.submit([&](sycl::handler &h) { // PARALLEL;
        //      sycl::stream stream_ct1(64 * 1024, 80, h);
        //      h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
        //          for (size_t i = 0; i < NUM_REA; i++)
        //          {
        //              for (size_t j = 0; j < d_react.rts[i]; j++)
        //                  stream_ct1 << d_react.reactant_list[i][j] << " ";
        //              stream_ct1 << ", ";
        //          }
        //          stream_ct1 << "\n";
        //      });
        //  })
        //     .wait();

        // for (size_t i = 0; i < NUM_REA; i++)
        // {
        //     for (size_t j = 0; j < h_react.pls[i]; j++)
        //         std::cout << h_react.product_list[i][j] << " ";
        //     std::cout << ", ";
        // }
        // std::cout << "\n";

        // q.submit([&](sycl::handler &h) { // PARALLEL;
        //      sycl::stream stream_ct1(64 * 1024, 80, h);
        //      h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
        //          for (size_t i = 0; i < NUM_REA; i++)
        //          {
        //              for (size_t j = 0; j < d_react.pls[i]; j++)
        //                  stream_ct1 << d_react.product_list[i][j] << " ";
        //              stream_ct1 << ", ";
        //          }
        //          stream_ct1 << "\n";
        //      });
        //  })
        //     .wait();

        // for (size_t i = 0; i < NUM_REA; i++)
        // {
        //     for (size_t j = 0; j < h_react.sls[i]; j++)
        //         std::cout << h_react.species_list[i][j] << " ";
        //     std::cout << ", ";
        // }
        // std::cout << "\n";

        // q.submit([&](sycl::handler &h) { // PARALLEL;
        //      sycl::stream stream_ct1(64 * 1024, 80, h);
        //      h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> index) { // BODY;
        //          for (size_t i = 0; i < NUM_REA; i++)
        //          {
        //              for (size_t j = 0; j < d_react.sls[i]; j++)
        //                  stream_ct1 << d_react.species_list[i][j] << " ";
        //              stream_ct1 << ", ";
        //          }
        //          stream_ct1 << "\n";
        //      });
        //  })
        //     .wait();
    }

#ifdef USE_MPI
    mpiTrans->communicator->synchronize();
#endif // end USE_MPI
    if (0 == myRank)
    {
        std::cout << " . Done \n";
    }
}

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

#if Visc
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
#endif // Visc
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
