#pragma once

#include "Utils_schemes.hpp"

inline real_t TENO5_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t b1, b2, b3;
	real_t a1, a2, a3, w1, w2, w3;
	real_t Variation1, Variation2, Variation3;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);

	real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
	real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
	real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);

	real_t tau5 = sycl::fabs(s3 - s2);

	a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40), 6.0);
	a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40), 6.0);
	a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40), 6.0);

	b1 = a1 / (a1 + a2 + a3);
	b2 = a2 / (a1 + a2 + a3);
	b3 = a3 / (a1 + a2 + a3);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0 * v2 + 5.0 * v3 + 2.0 * v4;
	Variation2 = 2.0 * v3 + 5.0 * v4 - 1.0 * v5;
	Variation3 = 2.0 * v1 - 7.0 * v2 + 11.0 * v3;

	a1 = 0.600 * b1;
	a2 = 0.300 * b2;
	a3 = 0.100 * b3;

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	return 1.0 / 6.0 * (w1 * Variation1 + w2 * Variation2 + w3 * Variation3);
}

inline real_t TENO5_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t b1, b2, b3;
	real_t a1, a2, a3, w1, w2, w3;
	real_t Variation1, Variation2, Variation3;

	// assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);

	real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
	real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
	real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);

	real_t tau5 = sycl::fabs(s3 - s2);

	a1 = std::pow(1.0 + tau5 / (s1 + 1.0e-40), 6.0);
	a2 = std::pow(1.0 + tau5 / (s2 + 1.0e-40), 6.0);
	a3 = std::pow(1.0 + tau5 / (s3 + 1.0e-40), 6.0);

	b1 = a1 / (a1 + a2 + a3);
	b2 = a2 / (a1 + a2 + a3);
	b3 = a3 / (a1 + a2 + a3);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0 * v2 + 5.0 * v3 + 2.0 * v4;
	Variation2 = 2.0 * v3 + 5.0 * v4 - 1.0 * v5;
	Variation3 = 2.0 * v1 - 7.0 * v2 + 11.0 * v3;

	a1 = 0.600 * b1;
	a2 = 0.300 * b2;
	a3 = 0.100 * b3;

	w1 = a1 / (a1 + a2 + a3);
	w2 = a2 / (a1 + a2 + a3);
	w3 = a3 / (a1 + a2 + a3);

	return 1.0 / 6.0 * (w1 * Variation1 + w2 * Variation2 + w3 * Variation3);
}

inline real_t TENO6_OPT_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6;
	real_t b1, b2, b3, b4, b5;
	real_t a1, a2, a3, a4, a5, w1, w2, w3, w4, w5;
	real_t Variation1, Variation2, Variation3, Variation4, Variation5;

	// assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1);
	v5 = *(f + k + 2);
	v6 = *(f + k + 3);

	real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
	real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
	real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	real_t s4 = 1. / 240. * std::fabs((2107.0 * v3 * v3 - 9402.0 * v3 * v4 + 11003.0 * v4 * v4 + 7042.0 * v3 * v5 - 17246.0 * v4 * v5 + 7043.0 * v5 * v5 - 1854.0 * v3 * v6 + 4642.0 * v4 * v6 - 3882.0 * v5 * v6 + 547.0 * v6 * v6));

	real_t s64 = 1.0 / 12.0 * std::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 3824847.0 * v5 * v5 + 2863984.0 * v4 * v6 - 15880404.0 * v4 * v5 + 17195652.0 * v4 * v4 - 2792660.0 * v3 * v6 - 35817664.0 * v3 * v4 + 19510972.0 * v3 * v3 + 1325006.0 * v2 * v6 - 7727988.0 * v2 * v5 + 17905032.0 * v2 * v4 - 20427884.0 * v2 * v3 + 5653317.0 * v2 * v2 - 245620.0 * v1 * v6 + 1458762.0 * v1 * v5 - 3462252.0 * v1 * v4 + 4086352.0 * v1 * v3 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1 + 15929912.0 * v3 * v5) / 10080.0;

	real_t tau6 = sycl::fabs(s64 - (s3 + s2 + 4.0 * s1) / 6.0);

	a1 = 1. / 4. * std::pow(1.0 + tau6 / (s1 + 1.0e-40), 6.0);
	a2 = 1. / 4. * std::pow(1.0 + tau6 / (s2 + 1.0e-40), 6.0);
	a3 = 1. / 4. * std::pow(1.0 + tau6 / (s3 + 1.0e-40), 6.0);
	a4 = 1. / 4. * std::pow(1.0 + tau6 / (s4 + 1.0e-40), 6.0);

	b1 = a1 / (a1 + a2 + a3 + a4);
	b2 = a2 / (a1 + a2 + a3 + a4);
	b3 = a3 / (a1 + a2 + a3 + a4);
	b4 = a4 / (a1 + a2 + a3 + a4);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;
	b4 = b4 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 2.0 / 6.0 * v4 - v3;
	Variation2 = 2. / 6. * v3 + 5. / 6. * v4 - 1. / 6. * v5 - v3;
	Variation3 = 2. / 6. * v1 - 7. / 6. * v2 + 11. / 6. * v3 - v3;
	Variation4 = 3. / 12. * v3 + 13. / 12. * v4 - 5. / 12. * v5 + 1. / 12. * v6 - v3;

	a1 = 0.462 * b1;
	a2 = 0.300 * b2;
	a3 = 0.054 * b3;
	a4 = 0.184 * b4;

	w1 = a1 / (a1 + a2 + a3 + a4);
	w2 = a2 / (a1 + a2 + a3 + a4);
	w3 = a3 / (a1 + a2 + a3 + a4);
	w4 = a4 / (a1 + a2 + a3 + a4);

	return v3 + w1 * Variation1 + w2 * Variation2 + w3 * Variation3 + w4 * Variation4;
}

inline real_t TENO6_OPT_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5, v6;
	real_t b1, b2, b3, b4;
	real_t a1, a2, a3, a4, w1, w2, w3, w4;
	real_t Variation1, Variation2, Variation3, Variation4;

	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1);
	v5 = *(f + k - 2);
	v6 = *(f + k - 3);

	real_t s1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) + 3.0 / 12.0 * (v2 - v4) * (v2 - v4);
	real_t s2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5) * (v3 - 2.0 * v4 + v5) + 3.0 / 12.0 * (3.0 * v3 - 4.0 * v4 + v5) * (3.0 * v3 - 4.0 * v4 + v5);
	real_t s3 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) + 3.0 / 12.0 * (v1 - 4.0 * v2 + 3.0 * v3) * (v1 - 4.0 * v2 + 3.0 * v3);
	real_t s4 = 1. / 240. * std::fabs((2107.0 * v3 * v3 - 9402.0 * v3 * v4 + 11003.0 * v4 * v4 + 7042.0 * v3 * v5 - 17246.0 * v4 * v5 + 7043.0 * v5 * v5 - 1854.0 * v3 * v6 + 4642.0 * v4 * v6 - 3882.0 * v5 * v6 + 547.0 * v6 * v6));

	real_t s64 = 1.0 / 12.0 * std::fabs(139633.0 * v6 * v6 - 1429976.0 * v5 * v6 + 3824847.0 * v5 * v5 + 2863984.0 * v4 * v6 - 15880404.0 * v4 * v5 + 17195652.0 * v4 * v4 - 2792660.0 * v3 * v6 - 35817664.0 * v3 * v4 + 19510972.0 * v3 * v3 + 1325006.0 * v2 * v6 - 7727988.0 * v2 * v5 + 17905032.0 * v2 * v4 - 20427884.0 * v2 * v3 + 5653317.0 * v2 * v2 - 245620.0 * v1 * v6 + 1458762.0 * v1 * v5 - 3462252.0 * v1 * v4 + 4086352.0 * v1 * v3 - 2380800.0 * v1 * v2 + 271779.0 * v1 * v1 + 15929912.0 * v3 * v5) / 10080.0;

	real_t tau6 = sycl::fabs(s64 - (s3 + s2 + 4.0 * s1) / 6.0);

	a1 = 1. / 4. * std::pow(1.0 + tau6 / (s1 + 1.0e-40), 6.0);
	a2 = 1. / 4. * std::pow(1.0 + tau6 / (s2 + 1.0e-40), 6.0);
	a3 = 1. / 4. * std::pow(1.0 + tau6 / (s3 + 1.0e-40), 6.0);
	a4 = 1. / 4. * std::pow(1.0 + tau6 / (s4 + 1.0e-40), 6.0);

	b1 = a1 / (a1 + a2 + a3 + a4);
	b2 = a2 / (a1 + a2 + a3 + a4);
	b3 = a3 / (a1 + a2 + a3 + a4);
	b4 = a4 / (a1 + a2 + a3 + a4);

	b1 = b1 < 1.0e-7 ? 0. : 1.;
	b2 = b2 < 1.0e-7 ? 0. : 1.;
	b3 = b3 < 1.0e-7 ? 0. : 1.;
	b4 = b4 < 1.0e-7 ? 0. : 1.;

	Variation1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 2.0 / 6.0 * v4 - v3;
	Variation2 = 2. / 6. * v3 + 5. / 6. * v4 - 1. / 6. * v5 - v3;
	Variation3 = 2. / 6. * v1 - 7. / 6. * v2 + 11. / 6. * v3 - v3;
	Variation4 = 3. / 12. * v3 + 13. / 12. * v4 - 5. / 12. * v5 + 1. / 12. * v6 - v3;

	a1 = 0.462 * b1;
	a2 = 0.300 * b2;
	a3 = 0.054 * b3;
	a4 = 0.184 * b4;

	w1 = a1 / (a1 + a2 + a3 + a4);
	w2 = a2 / (a1 + a2 + a3 + a4);
	w3 = a3 / (a1 + a2 + a3 + a4);
	w4 = a4 / (a1 + a2 + a3 + a4);

	return v3 + w1 * Variation1 + w2 * Variation2 + w3 * Variation3 + w4 * Variation4;
}
