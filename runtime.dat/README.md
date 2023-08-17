# This work:

### Reaction model

H2O_18_reaction released by ChemKin, see Ref1.PREMIX: AFORTRAN Program for Modeling Steady Laminar One-Dimensional Premixed Flames Pater48
H2O_19/23_reacton relaesed by M Ó Conaire , see Ref2.A Comprehensive Modeling Study of Hydrogen Oxidation

Reaction model for reacting shock-bubble interaction(RSBI) modified from H2O_18/19/23_reaction as Ref3.Three-dimensional reacting shock–bubble interaction:

- replace AR with Xe, third-body of Xe coeffs are the same as AR
- add N2 as the tenth insert gas species, its third body is set as 1.0 according to Ref4.Temperature and Third-Body Dependence of the Rate Constant for the Reaction O+O_2+M->O_3+M

### Transport model

Transport coefficients are copied form source dir of ChemKin

### Thermo model

JANAF or NASA fit, Only NASA fit has original support for Xe, thermo of Xe in JANAF is set the same as AR by Author of this work

# Alternative Refs

#### Reaction mechanism, thermodynamics, transport also released but not used in this work by NUI:

https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/
