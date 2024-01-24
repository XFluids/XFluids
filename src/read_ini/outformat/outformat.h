#pragma once

#include "outvars.hpp"

struct OutFmt
{
	real_t time;
	std::string inter;
	bool CPOut, SPOut;

	OutSlice pos;

	std::vector<std::string> _C;  // list of compressed dimensions
	std::vector<std::string> _V;  // list of output variables
	std::vector<OutVar> out_vars; // list of output variables

	std::vector<std::string> _P;	 // list of criterion variables
	std::vector<Criterion> cri_list; // list of criterion variables

	void Initialize_C(std::vector<std::string> c = std::vector<std::string>{});

	void Initialize_P(std::vector<std::string> sp, FlowData &data,
					  std::vector<std::string> p = std::vector<std::string>{});

	void Initialize_V(Block &Bl, std::vector<std::string> sp, FlowData &h_data,
					  std::vector<std::string> v = std::vector<std::string>{},
					  bool error = false, const int NUM = 1, const int ID = 0);

	void Initialize(Block &Bl, std::vector<std::string> &sp, FlowData &data,
					std::vector<std::string> c = std::vector<std::string>{},
					std::vector<std::string> p = std::vector<std::string>{},
					std::vector<std::string> v = std::vector<std::string>{});

	OutFmt Reinitialize_step(const std::string &Step);

	OutFmt Reinitialize_time(const real_t Time);

	OutFmt Reinitialize(const real_t Time, const std::string &Step);

	OutFmt(const real_t Time = _DF(0.0), std::string Step = "",
		   const bool Ox = true, const bool Oy = true, const bool Oz = true,
		   const real_t O_x = 0, const real_t O_y = 0, const real_t O_z = 0);

	OutFmt(std::vector<Criterion> &v, const real_t Time = 0, std::string Step = "",
		   const bool Ox = true, const bool Oy = true, const bool Oz = true,
		   const real_t O_x = 0, const real_t O_y = 0, const real_t O_z = 0);

	OutFmt(const real_t Time, std::vector<std::string> &c, std::string Step = "");

	OutFmt(std::vector<Criterion> &v, std::vector<std::string> &c,
		   const real_t Time = 0, std::string Step = "");
};
