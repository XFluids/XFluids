#pragma once

#include "global_class.h"
#include "marco.h"

const real_t _six = _DF(1.0) / _DF(6.0);
// TODO: NO std::cmath functions used if schemes function referenced, use sycl::math_function<real_t>

// NOTE: for WENO-CU6
const real_t _ohtz = _DF(1.0) / _DF(120.0);
const real_t _ohff = _DF(1.0) / _DF(144.0);
const real_t _ftss = _DF(1.0) / _DF(5760.0);
// const real_t _twfr = _DF(1.0) / _DF(24.0); // defined in marco.h
// const real_t _sxtn = _DF(1.0) / _DF(16.0); // defined in marco.h
const real_t wu6a2a2 = _DF(13.0) / _DF(3.0), wu6a3a3 = _DF(3129.0) / _DF(80.0);
const real_t wu6a4a4 = _DF(87617.0) / _DF(140.0), wu6a3a5 = _DF(14127.0) / _DF(224.0), wu6a5a5 = _DF(252337135.0) / _DF(16128.0);