#pragma once

#include "../../../include/global_setup.h"

BoundaryRange Vector2Boundary(std::vector<int> vec)
{
	BoundaryRange br;
	br.type = BConditions(vec[0]);
	br.xmin = vec[1], br.xmax = vec[2], br.ymin = vec[3];
	br.ymax = vec[4], br.zmin = vec[5], br.zmax = vec[6], br.tag = vec[7];

	return br;
}
