#include "../setupini.h"

// =======================================================
// =======================================================
void Setup::CpyToGPU()
{
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

#ifdef USE_MPI
    mpiTrans->communicator->synchronize();
#endif // end USE_MPI
    if (0 == myRank)
    {
        std::cout << " . Done \n";
    }
}
