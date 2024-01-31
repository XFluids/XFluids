#pragma once

#include "../read_ini/setupini.h"

std::pair<bool, std::vector<float>> UpdateFluidStateFlux(sycl::queue &q, Setup Ss, Thermal thermal, real_t *UI,
														 FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH,
														 real_t const Gamma, int &error_patched_times, const int rank);

void UpdateURK3rd(sycl::queue &q, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag);
