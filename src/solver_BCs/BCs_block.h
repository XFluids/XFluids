#pragma once

#include "../read_ini/setupini.h"

float FluidBoundaryCondition(sycl::queue &q, Setup setup, BConditions BCs[6], real_t *d_UI);
