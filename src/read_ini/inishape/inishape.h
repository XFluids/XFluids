#pragma once

#include "global_setup.h"

typedef struct
{
	// // box range
	real_t xmin_coor, xmax_coor, ymin_coor, ymax_coor, zmin_coor, zmax_coor;
	// // states inside box
	real_t rho, P, T, u, v, w, gamma, c, *yi, *hi, *ei, e, h, H;
} IniBox;

typedef struct
{
	// // bubble center
	real_t center_x, center_y, center_z;
	// // bubble shape
	real_t C, _xa2, _yb2, _zc2;
	// // states inside bubble
	real_t rho, P, T, u, v, w, gamma, c, *yi, *hi, *ei, e, h, H;
} IniBubble;

struct IniShape
{
	// // cop_type: 0 for 1d set, 1 for bubble of cop
	// // blast_type: 0 for 1d shock, 1 for circular shock
	int cop_type, blast_type, bubble_type;
	// // blast position
	real_t blast_center_x, blast_center_y, blast_center_z;
	// // shock much number
	real_t Ma, tau_H;
	// // blast  states
	real_t blast_density_in, blast_pressure_in, blast_T_in, blast_u_in, blast_v_in, blast_w_in, blast_c_in, blast_gamma_in;
	real_t blast_density_out, blast_pressure_out, blast_T_out, blast_u_out, blast_v_out, blast_w_out, blast_c_out, blast_gamma_out;
	// // bubble position
	real_t cop_center_x, cop_center_y, cop_center_z;
	// // bubble states
	real_t cop_density_in, cop_pressure_in, cop_T_in;
	// // bubble position; NOTE: Domain_length may be the max value of the Domain size
	real_t bubble_center_x, bubble_center_y, bubble_center_z;
	// // bubble shape
	real_t xa, yb, zc, C, _xa2, _yb2, _zc2;

	// // Utils initializing model
	size_t num_box, num_bubble;
	IniBox *iboxs;
	IniBubble *ibubbles;

	// IniShape(){};
	// ~IniShape(){};
	// IniShape(sycl::queue &q, size_t num_box, size_t num_bubble);
};
