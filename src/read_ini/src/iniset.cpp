#include <unistd.h>
#include "../setupini.h"
#include "../../solver_Ini/Mixing_device.h"

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
#ifdef USE_MPI
    BlSz.mx = mx_json, BlSz.my = my_json, BlSz.mz = mz_json;
#else
    BlSz.mx = 1, BlSz.my = 1, BlSz.mz = 1;
#endif
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

    // // rewrite domain size
    std::vector<real_t> Domain_size = apa.match<real_t>("-domain");
    if (!std::empty(Domain_size))
    {
        if (Domain_size[0] > 0)
            BlSz.Domain_length = Domain_size[0];
        if (Domain_size[1] > 0)
            BlSz.Domain_width = Domain_size[1];
        if (Domain_size[2] > 0)
            BlSz.Domain_height = Domain_size[2];
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

    std::vector<std::string> mpis = apa.match("-mpi-s");
    if (!std::empty(mpis))
    {
        if (0 == mpis[0].compare("strong")) // strong scaling
            if (!(BlSz.X_inner % BlSz.mx + BlSz.Y_inner % BlSz.my + BlSz.Z_inner % BlSz.mz))
                BlSz.X_inner /= BlSz.mx, BlSz.Y_inner /= BlSz.my, BlSz.Z_inner /= BlSz.mz;
            else
            {
                std::cout << "Error: the number of blocks in each direction is not divisible by the number of MPI processes in that direction!" << std::endl;
                exit(EXIT_FAILURE);
            }
        else if (0 == mpis[0].compare("weak")) // weak scaling
            BlSz.Domain_length *= BlSz.mx, BlSz.Domain_width *= BlSz.my, BlSz.Domain_height *= BlSz.mz;
    }

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
#if defined(__ACPP__)
    DeviceSelect[2] += myRank % DeviceSelect[0];
#else  // for oneAPI
    DeviceSelect[1] += myRank % DeviceSelect[0];
#endif // end

    // // adaptive range assignment init
    adv_id = 0;
    std::vector<sycl::range<3>> options;
    options.clear();
    std::vector<size_t> adv_ndx{1, 2, 4, 8, 16, 32, 64}; //, 128, 256, 512, 1024
    size_t advx = BlSz.DimX ? adv_ndx.size() : 1;
    size_t advy = BlSz.DimY ? adv_ndx.size() : 1;
    size_t advz = BlSz.DimZ ? adv_ndx.size() : 1;
    for (size_t zz = 0; zz < advz; zz++)
        for (size_t yy = 0; yy < advy; yy++)
            for (size_t xx = 0; xx < advx; xx++)
                if (adv_ndx[xx] * adv_ndx[yy] * adv_ndx[zz] > 2)
                    if (adv_ndx[xx] * adv_ndx[yy] * adv_ndx[zz] <= __LBMt)
                    {
                        if (adv_ndx[xx] < 2 && (adv_ndx[yy] > 256 || adv_ndx[zz] > 256))
                            continue;

                        int repeat = 0;
                        sycl::range<3> temprg(adv_ndx[xx], adv_ndx[yy], adv_ndx[zz]);
                        for (size_t dd = 0; dd < options.size(); dd++)
                        {
                            if (temprg == options[dd])
                                repeat++;
                        }
                        if (!repeat)
                        {
                            if ((BlSz.dim_block_x == adv_ndx[xx]) && (BlSz.dim_block_y == adv_ndx[yy]) && (BlSz.dim_block_z == adv_ndx[zz]))
                                options.insert(options.begin(), temprg);
                            else
                                options.push_back(temprg);
                        }
                    }
    adv_nd.resize(options.size());
    for (size_t dd = 0; dd < adv_nd.size(); dd++)
    {
        adv_nd[dd].clear();
        adv_nd[dd].push_back(Assign(options[dd]));
    }
    if (!UseAdvRange_json)
        adv_nd.resize(1);
}

// =======================================================
// =======================================================
void Setup::init()
{ // set other parameters
    BlSz.DimX_t = BlSz.DimX, BlSz.DimY_t = BlSz.DimY, BlSz.DimZ_t = BlSz.DimZ;
    BlSz.DimS = BlSz.DimX_t + BlSz.DimY_t + BlSz.DimZ_t, BlSz.DimS_t = BlSz.DimS;

    BlSz.X_inner = BlSz.DimX ? BlSz.X_inner : 1;
    BlSz.Y_inner = BlSz.DimY ? BlSz.Y_inner : 1;
    BlSz.Z_inner = BlSz.DimZ ? BlSz.Z_inner : 1;

    BlSz.Bwidth_X = BlSz.DimX ? BlSz.Bwidth_X : 0;
    BlSz.Bwidth_Y = BlSz.DimY ? BlSz.Bwidth_Y : 0;
    BlSz.Bwidth_Z = BlSz.DimZ ? BlSz.Bwidth_Z : 0;

    BlSz.dim_block_x = BlSz.DimX ? std::max<size_t>(BlSz.dim_block_x, 2) : 1;
    BlSz.dim_block_y = BlSz.DimY ? std::max<size_t>(BlSz.dim_block_y, 2) : 1;
    BlSz.dim_block_z = BlSz.DimZ ? std::max<size_t>(BlSz.dim_block_z, 2) : 1;

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

    BlSz.Ms.Bwidth_X = BlSz.Bwidth_X, BlSz.Ms.X_inner = BlSz.X_inner, BlSz.Ms.Xmax = BlSz.Xmax;
    BlSz.Ms.Bwidth_Y = BlSz.Bwidth_Y, BlSz.Ms.Y_inner = BlSz.Y_inner, BlSz.Ms.Ymax = BlSz.Ymax;
    BlSz.Ms.Bwidth_Z = BlSz.Bwidth_Z, BlSz.Ms.Z_inner = BlSz.Z_inner, BlSz.Ms.Zmax = BlSz.Zmax;
} // Setup::init
