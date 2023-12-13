#pragma once

#include "global_setup.h"

typedef struct
{
    real_t *X, *Y, *Z, *V;
} Volume;

class Gridread
{
private:
    Volume vol;
    Block bl;

public:
    real_t LRef;
    std::string GridPath;
    int *x_num, *y_num, *z_num, rank, nranks;
    int vector_num, bytes, Xmax, Ymax, Zmax;

    real_t *h_coordinate, *d_coordinate;
    real_t *fArea_I, *fArea_J, *fArea_K, *centroid, *volume_cell;

    Gridread(){};
    ~Gridread(){};
    Gridread(sycl::queue &Q, Block &Bl, std::string GridPath, int rank = 0, int nranks = 1);
    void CopyDataFromHost(sycl::queue &q);
    bool ReadGridBlock(Block &Bl);
    void FaceAreaI();
    void FaceAreaJ();
    void FaceAreaK();
    void GetVolume();
    void OutPut();
    real_t PyramidVolume(real_t p[3], real_t a[3], real_t b[3], real_t c[3], real_t d[3]);
};
