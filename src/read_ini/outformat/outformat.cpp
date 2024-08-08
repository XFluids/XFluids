#include "outformat.h"
#include "strsplit/strsplit.h"

/**
 * @brief
 * @param c
 */
void OutFmt::Initialize_C(std::vector<std::string> c)
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
void OutFmt::Initialize_P(std::vector<std::string> sp, FlowData &data,
						  std::vector<std::string> p)
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
void OutFmt::Initialize_V(Block &Bl, std::vector<std::string> sp, FlowData &h_data,
						  std::vector<std::string> v, bool error, const int NUM, const int ID)
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
			if (0 == _V[ii].compare("vorticity") && Visc)
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
void OutFmt::Initialize(Block &Bl, std::vector<std::string> &sp, FlowData &data,
						std::vector<std::string> c, std::vector<std::string> p, std::vector<std::string> v)
{
	Initialize_C(c);
	Initialize_P(sp, data, p);
	Initialize_V(Bl, sp, data, v);
};

OutFmt OutFmt::Reinitialize_step(const std::string &Step)
{
	OutFmt nfmt = *this;
	nfmt.inter = Step;

	return nfmt;
}

OutFmt OutFmt::Reinitialize_time(const real_t Time)
{
	OutFmt nfmt = *this;
	nfmt.time = Time;

	return nfmt;
}

OutFmt OutFmt::Reinitialize(const real_t Time, const std::string &Step)
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
OutFmt::OutFmt(const real_t Time, std::string Step, const bool Ox, const bool Oy, const bool Oz,
			   const real_t O_x, const real_t O_y, const real_t O_z)
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
OutFmt::OutFmt(std::vector<Criterion> &v, const real_t Time, std::string Step,
			   const bool Ox, const bool Oy, const bool Oz, const real_t O_x, const real_t O_y, const real_t O_z)
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
OutFmt::OutFmt(const real_t Time, std::vector<std::string> &c, std::string Step)
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
OutFmt::OutFmt(std::vector<Criterion> &v, std::vector<std::string> &c, const real_t Time, std::string Step)
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
