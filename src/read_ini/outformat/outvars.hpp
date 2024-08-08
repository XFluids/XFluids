#pragma once

#include "criterion.hpp"
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
	bool OutBoundary;					 // if out ghost Boundary cell
	bool OutDirX, OutDirY, OutDirZ;		 // out dirs
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
