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
