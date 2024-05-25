/*!
 * @file kinetics1.cpp
 *
 * Zero-dimensional kinetics
 *
 * This example simulates autoignition of hydrogen in a constant pressure
 * reactor and saves the time history to files that can be used for plotting.
 *
 * Keywords: combustion, reactor network, ignition delay, saving output
 */

// This file is redistributed form Cantera sample. See License.txt in the top-level directory of XFluids and Lixcense of cantera at XFluids/external/cantera or
// at https://cantera.org/license.txt for license and copyright information.

#include "example_utils.h"

int kinetics1(int np, void *p)
{
     //     std::cout << "Constant-pressure ignition of a "
     //          << "hydrogen/oxygen/nitrogen"
     //             " mixture \nbeginning at T = 1001 K and P = 1 atm."
     //          << std::endl;

     // create an ideal gas mixture that corresponds to OH submech from GRI-Mech 3.0
     auto sol = Cantera::newSolution("h2o2.yaml", "ohmech", "none");
     auto gas = sol->thermo();

     // set the state
     gas->setState_TPX(1001.0, Cantera::OneAtm, "H2:2.0, O2:1.0, N2:4.0");
     int nsp = gas->nSpecies();

     // create a reactor
     Cantera::IdealGasConstPressureReactor r;

     // 'insert' the gas into the reactor and environment.  Note
     // that it is ok to insert the same gas object into multiple
     // reactors or reservoirs. All this means is that this object
     // will be used to evaluate thermodynamic or kinetic
     // quantities needed.
     r.insert(sol);

     double dt = 1.e-5; // interval at which output is written
     int nsteps = 100;  // number of intervals

     // create a 2D array to hold the output variables,
     // and store the values for the initial state
     Cantera::Array2D states(nsp + 4, 1);
     saveSoln(0, 0.0, *(sol->thermo()), states);

     // create a container object to run the simulation
     // and add the reactor to it
     Cantera::ReactorNet sim;
     sim.addReactor(r);

     // main loop
     clock_t t0 = clock(); // save start time
     for (int i = 1; i <= nsteps; i++)
     {
          double tm = i * dt;
          sim.advance(tm);
          //    std::cout << "time = " << tm << " s" << std::endl;
          saveSoln(tm, *(sol->thermo()), states);
     }
     clock_t t1 = clock(); // save end time

     // make a CSV output file
     writeCsv("kin1.csv", *sol->thermo(), states);

     // print final temperature and timing data
     double tmm = 1.0 * (t1 - t0) / CLOCKS_PER_SEC;
     std::cout << " Tfinal = " << r.temperature() << std::endl;
     std::cout << " time = " << tmm << std::endl;
     std::cout << " number of residual function evaluations = "
               << sim.integrator().nEvals() << std::endl;
     std::cout << " time per evaluation = " << tmm / sim.integrator().nEvals()
               << std::endl
               << std::endl;
     std::cout << "Output files:" << std::endl
               << "  kin1.csv    (Excel CSV file)" << std::endl;

     return 0;
}

// int main()
// {
//     try {
//         int retn = kinetics1(0, 0);
//         appdelete();
//         return retn;
//     } catch (CanteraError& err) {
//         // handle exceptions thrown by Cantera
//         std::std::cout << err.what() << std::std::endl;
//         std::cout << " terminating... " << std::endl;
//         appdelete();
//         return -1;
//     }
// }
