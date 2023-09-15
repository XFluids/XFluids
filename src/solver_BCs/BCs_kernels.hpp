#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"

extern void FluidBCKernel(BoundaryRange &BC, Block &bl, real_t *d_UI)
{
}

extern void FluidBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	int id = Xmax * Ymax * k + Xmax * j + i;

	if (j >= Ymax)
		return;
	if (k >= Zmax)
		return;

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case Inflow:
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
		d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
		for (int n = Emax - NUM_COP; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
	}
	break;

	case viscWall:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
		d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
		for (int n = Emax - NUM_COP; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
	}
	break;

	case slipWall:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
		d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
		for (int n = Emax - NUM_COP; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
	}
	break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern void FluidBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	int id = Xmax * Ymax * k + Xmax * j + i;

	if (i >= Xmax)
		return;
	if (k >= Zmax)
		return;

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case Inflow:
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
		d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
		d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
		for (int n = Emax - NUM_COP; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
	}
	break;

	case viscWall:
		break;

	case slipWall:
		break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern void FluidBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *d_UI, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	int id = Xmax * Ymax * k + Xmax * j + i;

	if (i >= Xmax)
		return;
	if (j >= Ymax)
		return;

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
		int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case Inflow:
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
		for (int n = 0; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
		int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
		d_UI[Emax * id + 0] = d_UI[Emax * target_id + 0];
		d_UI[Emax * id + 1] = -d_UI[Emax * target_id + 1];
		d_UI[Emax * id + 2] = -d_UI[Emax * target_id + 2];
		d_UI[Emax * id + 3] = -d_UI[Emax * target_id + 3];
		d_UI[Emax * id + 4] = d_UI[Emax * target_id + 4];
#ifdef COP
		for (int n = Emax - NUM_COP; n < Emax; n++)
			d_UI[Emax * id + n] = d_UI[Emax * target_id + n];
#endif // end COP
	}
	break;

	case viscWall:
		break;

	case slipWall:
		break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern void FluidMpiCopyKernelX(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
											  const int Bwidth_Xset, const MpiCpyType Cpytype)
{
	int Xmax = bl.Xmax, Ymax = bl.Ymax;
	if (j >= Ymax)
		return;
	if (k >= bl.Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	int tid = sycl::abs(Bwidth_Xset) * Ymax * k + sycl::abs(Bwidth_Xset) * j + (i - index_offset);
	int fid = Xmax * Ymax * k + Xmax * j + (i - Bwidth_Xset);

	for (size_t n = 0; n < Emax; n++)
	{
		if (BorToBuf == Cpytype)
			d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
		if (BufToBC == Cpytype)
			d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
	}
}

extern void FluidMpiCopyKernelY(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
											  const int Bwidth_Yset, const MpiCpyType Cpytype)
{
	int Xmax = bl.Xmax, Ymax = bl.Ymax;
	if (j >= Ymax)
		return;
	if (k >= bl.Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	int tid = Xmax * sycl::abs(Bwidth_Yset) * k + Xmax * (j - index_offset) + i;
	int fid = Xmax * Ymax * k + Xmax * (j - Bwidth_Yset) + i;

	for (size_t n = 0; n < Emax; n++)
	{
		if (BorToBuf == Cpytype)
			d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
		if (BufToBC == Cpytype)
			d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
	}
}

extern void FluidMpiCopyKernelZ(int i, int j, int k, Block bl, real_t *d_TransBuf, real_t *d_UI, const int index_offset,
											  const int Bwidth_Zset, const MpiCpyType Cpytype)
{
	int Xmax = bl.Xmax, Ymax = bl.Ymax;
	if (j >= Ymax)
		return;
	if (k >= bl.Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	int tid = Xmax * Ymax * (k - index_offset) + Xmax * j + i;
	int fid = Xmax * Ymax * (k - Bwidth_Zset) + Xmax * j + i;

	for (size_t n = 0; n < Emax; n++)
	{
		if (BorToBuf == Cpytype)
			d_TransBuf[Emax * tid + n] = d_UI[Emax * fid + n];
		if (BufToBC == Cpytype)
			d_UI[Emax * id + n] = d_TransBuf[Emax * tid + n];
	}
}
