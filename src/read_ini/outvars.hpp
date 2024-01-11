#pragma once

#include <fstream>
#include "criterion.hpp"
#include "../include/global_setup.h"
// ================================================================================
// // // class OutVars Member function definitions
// // // hiding implementation of template function causes undefined linking error
// ================================================================================
typedef struct
{
	int nbX, nbY, nbZ;	  // number of points output along each DIR
	int minX, minY, minZ; // beginning point of output along each DIR
	int maxX, maxY, maxZ; // ending point of output along each DIR
} OutSize;

typedef struct
{
	bool OutBoundary;				  // if out ghost Boundary cell
	bool OutDirX, OutDirY, OutDirZ;	  // out dirs
	real_t outpos_x, outpos_y, outpos_z; // slice offset id
} OutSlice;

struct OutString
{
	// Write time in string timeFormat
	std::ostringstream timeFormat;
	// Write istep in string stepFormat
	std::ostringstream stepFormat;
	// Write Mpi Rank in string rankFormat
	std::ostringstream rankFormat;

	OutString(real_t time, int rank, std::string step)
	{
		// Write time in string timeFormat
		timeFormat.width(11);
		timeFormat.fill('0');
		timeFormat << time * 1E9;
		// Write istep in string stepFormat
		stepFormat.width(7);
		stepFormat.fill('0');
		stepFormat << step;
		// Write Mpi Rank in string rankFormat
		rankFormat.width(5);
		rankFormat.fill('0');
		rankFormat << rank;
	}
};

class OutVar
{
	real_t *var;
	size_t num, num_id;
	bool axis[3], outdir;

public:
	std::string name;

	OutVar(){};

	OutVar(std::string Name, const int dir) : name(Name)
	{
		num = 1, num_id = 0;
		axis[0] = false, axis[1] = false, axis[2] = false, axis[dir] = true;
		outdir = axis[0] || axis[1] || axis[2];
	};

	OutVar(std::string Name, real_t *ptr, const int NUM = 1, const int ID = 0)
	{
		name = Name;
		var = ptr, num = NUM, num_id = ID;
		axis[0] = false, axis[1] = false, axis[2] = false;
		outdir = axis[0] || axis[1] || axis[2];
	};

	template <typename T>
	void vti_binary(Block &bl, std::fstream &out, OutSize &os)
	{
		unsigned int nbOfWords = os.nbX * os.nbY * os.nbZ * sizeof(T);
		out.write((char *)&nbOfWords, sizeof(unsigned int));
		if (!outdir)
		{
			for (size_t k = os.minZ; k < os.maxZ; k++)
				for (size_t j = os.minY; j < os.maxY; j++)
					for (size_t i = os.minX; i < os.maxX; i++)
					{
						size_t id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						T temp = static_cast<T>(var[num * id + num_id]);
						out.write((char *)&temp, sizeof(T));
					}
		}
		else if (outdir)
		{
			real_t axisx = axis[0], axisy = axis[1], axisz = axis[2];
			for (size_t k = os.minZ; k < os.maxZ; k++)
				for (size_t j = os.minY; j < os.maxY; j++)
					for (size_t i = os.minX; i < os.maxX; i++)
					{
						real_t tmp = _DF(0.0);
						tmp += axisx * ((i - bl.Bwidth_X + bl.myMpiPos_x * (bl.X_inner) + _DF(0.5)) * bl.dx + bl.Domain_xmin);
						tmp += axisy * ((j - bl.Bwidth_Y + bl.myMpiPos_y * (bl.Y_inner) + _DF(0.5)) * bl.dy + bl.Domain_ymin);
						tmp += axisz * ((k - bl.Bwidth_Z + bl.myMpiPos_z * (bl.Z_inner) + _DF(0.5)) * bl.dz + bl.Domain_zmin);
						T temp = static_cast<T>(tmp);
						out.write((char *)&temp, sizeof(T));
					}
		}
	};

	void plt_ascii_block(Block &bl, std::fstream &out, OutSize &os)
	{
		for (size_t k = os.minZ; k < os.maxZ; k++)
			for (size_t j = os.minY; j < os.maxY; j++)
			{
				for (size_t i = os.minX; i < os.maxX; i++)
				{
					size_t id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
					out << (var[num * id + num_id]) << " ";
				}
				out << "\n";
			}
	};

	void plt_ascii_point(std::fstream &out, const size_t id)
	{
		out << (var[num * id + num_id]) << " ";
	};
};

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

	/**
	 * @brief
	 * @param c
	 */
	void Initialize_C(std::vector<std::string> c = std::vector<std::string>{})
	{
		if (!std::empty(c))
			_C = c;

		if (!std::empty(_C))
		{
			if (0 == _C[0].compare("X"))
				pos.OutDirX = true;
			else
				pos.OutDirX = false, pos.outpos_x = stod(_C[0]);

			if (0 == _C[1].compare("Y"))
				pos.OutDirY = true;
			else
				pos.OutDirY = false, pos.outpos_y = stod(_C[1]);

			if (0 == _C[2].compare("Z"))
				pos.OutDirZ = true;
			else
				pos.OutDirZ = false, pos.outpos_z = stod(_C[2]);
		}
		CPOut = !(pos.OutDirX && pos.OutDirY && pos.OutDirZ);
	}

	/**
	 * @brief
	 * @param p
	 */
	void Initialize_P(std::vector<std::string> sp, FlowData &data,
					  std::vector<std::string> p = std::vector<std::string>{})
	{
		if (!std::empty(p))
			_P = p;

		if (!std::empty(_P))
		{
			for (size_t i = 0; i < _P.size(); i++)
				cri_list.push_back(Criterion(Stringsplit(_P[i], ' '), sp, data));
			SPOut = (pos.OutDirX && pos.OutDirY && pos.OutDirZ);
		}
	}

	/**
	 * @brief
	 * @param v
	 */
	void Initialize_V(Block &Bl, std::vector<std::string> sp, FlowData &h_data,
					  std::vector<std::string> v = std::vector<std::string>{},
					  bool error = false, const int NUM = 1, const int ID = 0)
	{
		if (!std::empty(v))
			_V = v;

		if (!std::empty(_V))
		{
			// Init var names
			out_vars.clear();
			for (size_t ii = 0; ii < _V.size(); ii++)
			{
				if (Bl.DimX)
				{
					if (0 == _V[ii].compare("axis"))
						out_vars.push_back(OutVar("axis_x", 0));
					if (0 == _V[ii].compare("u"))
						out_vars.push_back(OutVar("velocity_u", h_data.u));
				}
				if (Bl.DimY)
				{
					if (0 == _V[ii].compare("axis"))
						out_vars.push_back(OutVar("axis_y", 1));
					if (0 == _V[ii].compare("v"))
						out_vars.push_back(OutVar("velocity_v", h_data.v));
				}
				if (Bl.DimZ)
				{
					if (0 == _V[ii].compare("axis"))
						out_vars.push_back(OutVar("axis_z", 2));
					if (0 == _V[ii].compare("w"))
						out_vars.push_back(OutVar("velocity_w", h_data.w));
				}
				if (0 == _V[ii].compare("rho"))
					out_vars.push_back(OutVar("rho", h_data.rho));
				if ((0 == _V[ii].compare("p")) || (0 == _V[ii].compare("P")))
					out_vars.push_back(OutVar("p", h_data.p));
				if (0 == _V[ii].compare("T"))
					out_vars.push_back(OutVar("T", h_data.T));
				if (0 == _V[ii].compare("e"))
					out_vars.push_back(OutVar("e", h_data.e));
				if (0 == _V[ii].compare("c"))
					out_vars.push_back(OutVar("c", h_data.c));
				if (0 == _V[ii].compare("Gamma"))
					out_vars.push_back(OutVar("g", h_data.gamma));
				if (0 == _V[ii].compare("vorticity"))
				{
					out_vars.push_back(OutVar("vorticity", h_data.vx));
					if ((0 == _V[ii].compare("vorticity_x")) && (Bl.DimY) && (Bl.DimZ))
						out_vars.push_back(OutVar("vorticity_x", h_data.vxs[0]));
					if ((0 == _V[ii].compare("vorticity_y")) && (Bl.DimX) && (Bl.DimZ))
						out_vars.push_back(OutVar("vorticity_y", h_data.vxs[1]));
					if ((0 == _V[ii].compare("vorticity_z")) && (Bl.DimX) && (Bl.DimY))
						out_vars.push_back(OutVar("vorticity_z", h_data.vxs[2]));
				}
#ifdef COP
				if (_V[ii].find("yi[") != std::string::npos)
					for (size_t nn = 0; nn < sp.size(); nn++)
					{
						if ((0 == _V[ii].compare("yi[" + sp[nn] + "]")) || (0 == _V[ii].compare("yi[all]")))
							out_vars.push_back(OutVar("y" + std::to_string(nn) + "[" + sp[nn] + "]", h_data.y, sp.size(), nn));
					}
#endif // COP
			}
		}
	}

	/**
	 * @brief
	 * @param Bl
	 * @param sp
	 * @param data
	 * @param c
	 * @param p
	 * @param v
	 */
	void Initialize(Block &Bl, std::vector<std::string> &sp, FlowData &data,
					std::vector<std::string> c = std::vector<std::string>{},
					std::vector<std::string> p = std::vector<std::string>{},
					std::vector<std::string> v = std::vector<std::string>{})
	{
		Initialize_C(c);
		Initialize_P(sp, data, p);
		Initialize_V(Bl, sp, data, v);
	};

	OutFmt Reinitialize_step(const std::string &Step)
	{
		OutFmt nfmt = *this;
		nfmt.inter = Step;

		return nfmt;
	}

	OutFmt Reinitialize_time(const real_t Time)
	{
		OutFmt nfmt = *this;
		nfmt.time = Time;

		return nfmt;
	}

	OutFmt Reinitialize(const real_t Time, const std::string &Step)
	{
		OutFmt nfmt = *this;
		nfmt.time = Time;
		nfmt.inter = Step;

		return nfmt;
	}

	/**
	 * @brief Construct a new Out Fmt object
	 * @param Step: computational step of the output
	 * @param Time: physical time of the output
	 * @param Ox: if x axis dimension output
	 * @param Oy: if y axis dimension output
	 * @param Oz: if z axis dimension output
	 */
	OutFmt(const real_t Time = _DF(0.0), std::string Step = "",
		   const bool Ox = true, const bool Oy = true, const bool Oz = true,
		   const real_t O_x = 0, const real_t O_y = 0, const real_t O_z = 0)
	{
		inter = Step, time = Time;
		SPOut = false, cri_list.clear(), out_vars.clear();
		pos.OutDirX = Ox, pos.OutDirY = Oy, pos.OutDirZ = Oz;
		CPOut = !(pos.OutDirX && pos.OutDirY && pos.OutDirZ);
		pos.outpos_x = O_x, pos.outpos_y = O_y, pos.outpos_z = O_z;
	};

	/**
	 * @brief alternative compress dimension output support with partial output
	 * @param v partial output criterions
	 */
	OutFmt(std::vector<Criterion> &v, const real_t Time = 0, std::string Step = "",
		   const bool Ox = true, const bool Oy = true, const bool Oz = true,
		   const real_t O_x = 0, const real_t O_y = 0, const real_t O_z = 0)
	{
		inter = Step, time = Time;
		cri_list = v, out_vars.clear();
		pos.OutDirX = Ox, pos.OutDirY = Oy, pos.OutDirZ = Oz;
		CPOut = !(pos.OutDirX && pos.OutDirY && pos.OutDirZ);
		pos.outpos_x = O_x, pos.outpos_y = O_y, pos.outpos_z = O_z;
		SPOut = (pos.OutDirX && pos.OutDirY && pos.OutDirZ);
	};

	/**
	 * @brief Construct a new Out Fmt object
	 * @param Step: computational step of the output
	 * @param Time: physical time of the output
	 * @param Ox: if x axis dimension output
	 * @param Oy: if y axis dimension output
	 * @param Oz: if z axis dimension output
	 */
	// = std::vector<std::string>{}
	OutFmt(const real_t Time, std::vector<std::string> &c, std::string Step = "")
	{
		inter = Step, time = Time;
		SPOut = false, cri_list.clear(), out_vars.clear(), _C = c;
		pos.outpos_x = 0, pos.outpos_y = 0, pos.outpos_z = 0;

		if (0 == _C[0].compare("X"))
			pos.OutDirX = true;
		else
			pos.OutDirX = false, pos.outpos_x = stod(_C[0]);

		if (0 == _C[1].compare("Y"))
			pos.OutDirY = true;
		else
			pos.OutDirY = false, pos.outpos_y = stod(_C[1]);

		if (0 == _C[2].compare("Z"))
			pos.OutDirZ = true;
		else
			pos.OutDirZ = false, pos.outpos_z = stod(_C[2]);

		CPOut = !(pos.OutDirX && pos.OutDirY && pos.OutDirZ);
	};

	/**
	 * @brief alternative compress dimension output support with partial output
	 * @param v partial output criterions
	 */
	// = std::vector<std::string>{}
	OutFmt(std::vector<Criterion> &v, std::vector<std::string> &c,
		   const real_t Time = 0, std::string Step = "")
	{
		inter = Step, time = Time;
		cri_list = v, out_vars.clear(), _C = c;
		pos.outpos_x = 0, pos.outpos_y = 0, pos.outpos_z = 0;

		if (0 == _C[0].compare("X"))
			pos.OutDirX = true;
		else
			pos.OutDirX = false, pos.outpos_x = stod(_C[0]);

		if (0 == _C[1].compare("Y"))
			pos.OutDirY = true;
		else
			pos.OutDirY = false, pos.outpos_y = stod(_C[1]);

		if (0 == _C[2].compare("Z"))
			pos.OutDirZ = true;
		else
			pos.OutDirZ = false, pos.outpos_z = stod(_C[2]);

		CPOut = !(pos.OutDirX && pos.OutDirY && pos.OutDirZ);
		SPOut = (pos.OutDirX && pos.OutDirY && pos.OutDirZ);
	};
};