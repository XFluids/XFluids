#pragma once

// =======================================================
//    Global Dimension settings
#ifndef DIM_X
#define DIM_X 1
#endif
#ifndef DIM_Y
#define DIM_Y 1
#endif
#ifndef DIM_Z
#define DIM_Z 1
#endif

#ifndef SBICounts
#define SBICounts 0
#endif

#ifndef NUM_SPECIES
#define NUM_SPECIES 1
#endif
#ifndef COP_CHEME
#define COP_CHEME 1
#endif
#ifndef EIGEN_ALLOC
#define EIGEN_ALLOC 0
#endif
#ifndef Visc
#define Visc 0
#endif
#ifndef Visc_Heat
#define Visc_Heat 0
#endif
#ifndef Visc_Diffu
#define Visc_Diffu 0
#endif
#ifndef NUM_REA
#define NUM_REA 1
#endif
#ifndef CHEME_SPLITTING
#define CHEME_SPLITTING "Strang"
#endif
#ifndef BackArre
#define BackArre false
#endif

#ifndef ESTIM_NAN
#define ESTIM_NAN 1
#endif

#ifndef IniFile
#define IniFile "IniFile-undefined"
#endif
#ifndef INI_SAMPLE
#define INI_SAMPLE "sample-undefined"
#endif
#ifndef RFile
#define RFile "./"
#endif
#ifndef RPath
#define RPath "./"
#endif
