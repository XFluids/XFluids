# This work:

### Reaction model

- H2O_18_reaction released by ChemKin, see Ref1.PREMIX: AFORTRAN Program for Modeling Steady Laminar One-Dimensional Premixed Flames Pater48
- H2O_19/23_reaction relaesed by M Ó Conaire , see Ref3.A Comprehensive Modeling Study of Hydrogen Oxidation
- H2O-N2_19_reaction for 2D-Denotation, see Ref.https://github.com/deepmodeling/deepflame-dev/blob/master/examples/dfHighSpeedFoam/twoD_detonationH2/H2_Ja.yaml
- H2O-N2_21_reaction for mixing-layer-interaction, see Ref2.A detailed verification procedure for compressible reactivemulticomponent Navier–Stokes solvers
- H2O-N2_25_reaction for chemeq2 validation, see NEW QUASI-STEADY-STATE AND PARTIAL-EQUILIBRIUM M ETHODS FOR INTEGRATING CHEMICALLY REACTING SYSTEMS
- RSBI_18/19REA modified from H2O_18/19_reaction are Reaction model for reacting shock-bubble interaction(RSBI) as Ref4.Three-dimensional reacting shock–bubble interaction:
  - replace AR with Xe, third-body of Xe coeffs are the same as AR
  - add N2 as the tenth inert gas species

#### Arrhenius Law form

- Standrd ArrheniusLaw form, Ref.Evaluated Kinetic Data for High‐Temperature Reactions.
  - *Arrhenius arguments:list A B C , k=A*T^B*exp(-C/T)
  - *input units C: K, A: cm^3/molecule/s, NOTE: 1 mol=NA molecule
  - *output units: k: cm^3/mol/s

- Default ArrheniusLaw form, Ref.PREMIX:AFORTRAN Program for Modeling Steady Laminar One-Dimensional Premixed Flames Paper48
  - *Arrhenius arguments:list A B E , k=A*T^B*exp(-E/R/T)
  - *input units E: cal/mol, A: cm^3/mole/s, NOTE: 1 cal=4.184 J*mol/K
  - *output units: k: cm^3/mol/s

- Actually, C=E/R, in which E as activation energy, R as gas constant(8.314J/mol/K, 1.987cal/mol/K, 82.05cm^3*atm/mol/K);

#### Three-body(3B) and three-body coffcients(3Bs)

  - 3Bs of specific species are given by reaction model
  - 3Bs is by default set as 1.0 for "M" according to cantera and following reference:
      Ref4.Temperature and Third-Body Dependence of the Rate Constant for the Reaction O+O_2+M->O_3+M
      Ref: class Cantera::ThirdBody::default_efficiency 
  - not only "M" but also a species named "O2" as a example, exist in both reactants and products, act as the 3B:
      3Bs of "O2" is set to 1, and other species ia set to 0;
      Ref: Cantera::Reaction::setEquation(const string& equation, const Kinetics* kin){} to find the species
           Cantera::ThirdBody::setName(const string& third_body){} to set the 3Bs



### Transport model

Transport coefficients are copied form source dir of ChemKin, add Xe and NCOP(for no-component flow)

### Thermo model

JANAF or NASA fit, Only NASA fit has original support for Xe and is used by default, thermo of Xe in JANAF is set the same as AR by Author of this work

# Alternative Refs

#### Reaction mechanism, thermodynamics, transport also released but not used in this work by NUI:

- Single reaction kinetics released by [NIST Chemical Kinetics Database(https://kinetics.nist.gov/kinetics/)](https://kinetics.nist.gov/kinetics/)
- Reaction mechanism,thermodynamics and transport released by [Combustion Chemistry Centre, University of Galway(https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/)](https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/)