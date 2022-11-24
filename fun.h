#pragma once
#include "setup.h"

typedef struct{
	Real *rho, *p, *c, *H, *u, *v, *w;
}FlowData;

typedef struct{
	int Mtrl_ind;
	Real Rgn_ind;							//indicator for region: inside interface, -1.0 or outside 1.0
	Real Gamma, A, B, rho0;		//Eos Parameters and maxium sound speed
	Real R_0, lambda_0; 			//gas constant and heat conductivity
}MaterialProperty;