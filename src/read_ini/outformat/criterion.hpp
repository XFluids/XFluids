#pragma once

#include "include/global_setup.h"

// ================================================================================
// // // class Criterion Member function definitions
// // // hiding implementation of template function causes undefined linking error
// ================================================================================

class Criterion
{
	bool is_num;
	real_t *var, line;
	int opera;
	size_t num, num_id;

public:
	Criterion(std::vector<std::string> str, std::vector<std::string> species_list,
			  FlowData &h_data) : is_num(false), num(1), num_id(0)
	{
		std::string svar = str[0];
		if (0 == svar.compare("rho"))
			var = h_data.rho;
		else if ((0 == svar.compare("p")) || (0 == svar.compare("P")))
			var = h_data.p;
		else if (0 == svar.compare("T"))
			var = h_data.T;
		else if (0 == svar.compare("u"))
			var = h_data.u;
		else if (0 == svar.compare("v"))
			var = h_data.v;
		else if (0 == svar.compare("w"))
			var = h_data.w;
		else if (0 == svar.compare("Gamma"))
			var = h_data.gamma;
		else if (0 == svar.compare("vorticity") && Visc)
			var = h_data.vx;
		else if (svar.find("yi[") != std::string::npos)
		{
			var = h_data.y;
			num = species_list.size();
			for (size_t nn = 0; nn < species_list.size(); nn++)
			{
				if (0 == svar.compare("yi[" + species_list[nn] + "]"))
					num_id = nn;
			}
		}
		else
		{
			std::cout << "\n  Criterion: Invalid variable name: " << svar << std::endl;
			std::cout << "Criterion: available variable name: rho, p(P), u, v, w, Gamma, vorticity" << str[1] << std::endl;
			for (size_t nn = 0; nn < species_list.size(); nn++)
				std::cout << ", yi[" << species_list[nn] << "]";
			std::cout << "\n";
		}

		line = stod(str[2]);
		if (0 == str[1].compare("="))
			opera = 0;
		else if (0 == str[1].compare("<"))
			opera = 1;
		else if (0 == str[1].compare(">"))
			opera = 2;
		else if (0 == str[1].compare("<="))
			opera = -1;
		else if (0 == str[1].compare(">="))
			opera = -2;
		else
		{
			std::cout << "\n  Criterion: Invalid compare operation: " << str[1] << std::endl;
			std::cout << "Criterion: available compare operation: =, <, >, <=, >=" << str[1] << std::endl;
		}
	};

	bool in_range(const int id)
	{
		real_t rvar = var[id * num + num_id];
		bool in_size = false;
		switch (opera)
		{
		case 0:
			in_size = (rvar == line);
			break;
		case 1:
			in_size = (rvar < line);
			break;
		case 2:
			in_size = (rvar > line);
			break;
		case -1:
			in_size = (rvar <= line);
			break;
		case -2:
			in_size = (rvar >= line);
			break;
		}

		bool temp = in_size;

		return in_size;
	}
};
