#include <map>
#include <iomanip>
#include <fstream>
#include <iostream>

#include "readgrid.h"

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::istringstream;
using std::ofstream;
// using std::istream_iterator;
using std::stod;
using std::stoi;
using std::string;
using std::unique_ptr;
// using std::set;

Gridread::Gridread(sycl::queue &Q, Block &Bl, std::string Grid_Path, int Rank, int NRanks) : bl(Bl), GridPath(Grid_Path), rank(Rank), nranks(NRanks), LRef(_DF(1.0))
{
    vector_num = 4;
    if (ReadGridBlock(Bl))
    {
        FaceAreaI();
        FaceAreaJ();
        FaceAreaK();
        GetVolume();
        CopyDataFromHost(Q);
        OutPut();
    }
}

bool Gridread::ReadGridBlock(Block &bl)
{
    // open binary plot3d grid file
    // .xyz file struct
    // number of block
    // number of i, number of j, number of k
    // coordinate of i, coordinate of j, coordinate k

    ifstream fName;
    std::string readName = GridPath;
    fName.open(readName, ios::in | ios::binary);
    // check to see if file opened correctly
    if (fName.fail())
    {
        if (0 == rank)
            std::cout << "Read_grid submodule closed:\n"
                      << "  Plot3DGrid file \"<" << readName
                      << ">\" did not open correctly.\n";
        return false;
    }

    cout << "Reading grid file: " << readName << "...";
    auto numBlks = 1;
    fName.read(reinterpret_cast<char *>(&numBlks), sizeof(numBlks)); // cout << "Block Number: " << ii << "     ";

    for (auto ii = 0; ii < numBlks; ii++)
    {
        fName.read(reinterpret_cast<char *>(&Xmax), sizeof(Xmax)); // cout << "I-DIM: " << tempInt << "     ";
        fName.read(reinterpret_cast<char *>(&Ymax), sizeof(Ymax)); // cout << "J-DIM: " << tempInt << "     ";
        fName.read(reinterpret_cast<char *>(&Zmax), sizeof(Zmax)); // cout << "K-DIM: " << tempInt << endl;
    }
    bl.X_inner = Xmax - 1, bl.Y_inner = Ymax - 1, bl.Z_inner = Zmax - 1;
    LRef = bl.LRef;
    bytes = Xmax * Ymax * Zmax * sizeof(real_t);
    volume_cell = static_cast<real_t *>(std::malloc(bytes));
    fArea_I = static_cast<real_t *>(std::malloc(bytes * vector_num));
    fArea_J = static_cast<real_t *>(std::malloc(bytes * vector_num));
    fArea_K = static_cast<real_t *>(std::malloc(bytes * vector_num));
    centroid = static_cast<real_t *>(std::malloc(bytes * vector_num));
    h_coordinate = static_cast<real_t *>(std::malloc(bytes * vector_num));

    auto tempDouble = 0.0;
    for (int kk = 0; kk < Zmax; kk++)
    {
        for (int jj = 0; jj < Ymax; jj++)
        {
            for (int ii = 0; ii < Xmax; ii++)
            {
                int id = Xmax * Ymax * kk + Xmax * jj + ii;
                fName.read(reinterpret_cast<char *>(&tempDouble), sizeof(tempDouble));
                h_coordinate[vector_num * id + 0] = tempDouble / LRef;
            }
        }
    }

    for (int kk = 0; kk < Zmax; kk++)
    {
        for (int jj = 0; jj < Ymax; jj++)
        {
            for (int ii = 0; ii < Xmax; ii++)
            {
                int id = Xmax * Ymax * kk + Xmax * jj + ii;
                fName.read(reinterpret_cast<char *>(&tempDouble), sizeof(tempDouble));
                h_coordinate[vector_num * id + 1] = tempDouble / LRef;
            }
        }
    }

    for (int kk = 0; kk < Zmax; kk++)
    {
        for (int jj = 0; jj < Ymax; jj++)
        {
            for (int ii = 0; ii < Xmax; ii++)
            {
                int id = Xmax * Ymax * kk + Xmax * jj + ii;
                fName.read(reinterpret_cast<char *>(&tempDouble), sizeof(tempDouble));
                h_coordinate[vector_num * id + 2] = tempDouble / LRef;
            }
        }
    }

    fName.close();
    std::cout << "Grid file read completed" << endl;

    // -------------------------------------test coordinate pass-------------------------------------
    // for (int i = 0; i < numcoordinate; i++)
    // {
    //     cout << h_coordinate[vector_num*i+0] << "," << h_coordinate[vector_num*i+1] << ","
    //         << h_coordinate[vector_num*i+2] << "," << h_coordinate[vector_num*i+3] << "\n";
    // }

    return true;
}

void Gridread::FaceAreaI()
{
    int AreaI_Inum = Xmax;
    int AreaI_Jnum = Ymax - 1;
    int AreaI_Knum = Zmax - 1;
    real_t xac[3], xbd[3];

    // -------------------------------test at single point----------------------------------
    // int id_before = Xmax*Ymax*0 + Xmax*0 + 2;
    // int id_after = Xmax*Ymax*(0+1) + Xmax*(0+1) + 2;
    // cout << "coord_(2,0,0):" << h_coordinate[id_before*vector_num+0] << ", " << h_coordinate[id_before*vector_num+1] << ", "
    //     << h_coordinate[id_before*vector_num+2] << "\n";

    // cout << "coord_(2,1,1):" << h_coordinate[id_after*vector_num+0] << ", " << h_coordinate[id_after*vector_num+1] << ", "
    //     << h_coordinate[id_after*vector_num+2] << "\n";

    for (int kk = 0; kk < AreaI_Knum; kk++)
    {
        for (int jj = 0; jj < AreaI_Jnum; jj++)
        {
            for (int ii = 0; ii < AreaI_Inum; ii++)
            {
                int id_ori = Xmax * Ymax * kk + Xmax * jj + ii;
                int id_jup = Xmax * Ymax * kk + Xmax * (jj + 1) + ii;
                int id_kup = Xmax * Ymax * (kk + 1) + Xmax * jj + ii;
                int id_jup_kup = Xmax * Ymax * (kk + 1) + Xmax * (jj + 1) + ii;
                xac[0] = h_coordinate[vector_num * id_jup_kup + 0] - h_coordinate[vector_num * id_ori + 0];
                xac[1] = h_coordinate[vector_num * id_jup_kup + 1] - h_coordinate[vector_num * id_ori + 1];
                xac[2] = h_coordinate[vector_num * id_jup_kup + 2] - h_coordinate[vector_num * id_ori + 2];
                // ----------------------------------xac test pass-------------------------------------
                // cout << "xac: " << xac[0] << ", " << xac[1] << ", " << xac[2] << "\n";

                xbd[0] = h_coordinate[vector_num * id_jup + 0] - h_coordinate[vector_num * id_kup + 0];
                xbd[1] = h_coordinate[vector_num * id_jup + 1] - h_coordinate[vector_num * id_kup + 1];
                xbd[2] = h_coordinate[vector_num * id_jup + 2] - h_coordinate[vector_num * id_kup + 2];
                // ----------------------------------xbd test pass-------------------------------------
                // cout << "xbd: " << xbd[0] << ", " << xbd[1] << ", " << xbd[2] << "\n";

                fArea_I[vector_num * id_ori + 0] = (xbd[1] * xac[2] - xbd[2] * xac[1]) / 2.0;
                fArea_I[vector_num * id_ori + 1] = (-1.0) * (xbd[0] * xac[2] - xbd[2] * xac[0]) / 2.0;
                fArea_I[vector_num * id_ori + 2] = (xbd[0] * xac[1] - xbd[1] * xac[0]) / 2.0;
                // ----------------------------------xbd.crossprod test pass-------------------------------------
                // cout << "xbd.crossprod: " << fArea_I[vector_num*id_ori+0] << ", "
                //                             << fArea_I[vector_num*id_ori+1] << ", "
                //                             << fArea_I[vector_num*id_ori+2] << "\n";

                fArea_I[vector_num * id_ori + 3] = sqrt(fArea_I[vector_num * id_ori + 0] * fArea_I[vector_num * id_ori + 0] + fArea_I[vector_num * id_ori + 1] * fArea_I[vector_num * id_ori + 1] + fArea_I[vector_num * id_ori + 2] * fArea_I[vector_num * id_ori + 2]);
                // ----------------------------------sqrt test pass-------------------------------------
                // cout << "sqrt: " << fArea_I[vector_num*id_ori+3] << "\n";

                fArea_I[vector_num * id_ori + 0] = fArea_I[vector_num * id_ori + 0] / fArea_I[vector_num * id_ori + 3];
                fArea_I[vector_num * id_ori + 1] = fArea_I[vector_num * id_ori + 1] / fArea_I[vector_num * id_ori + 3];
                fArea_I[vector_num * id_ori + 2] = fArea_I[vector_num * id_ori + 2] / fArea_I[vector_num * id_ori + 3];
                // ----------------------------------faceAreaI test pass-------------------------------------
                // cout << std::setprecision(30) << "vector: " << fArea_I[vector_num*id_ori+0] << ", "
                //     << fArea_I[vector_num*id_ori+1] << ", "
                //     << fArea_I[vector_num*id_ori+2] << ", "
                //     << fArea_I[vector_num*id_ori+3] << "\n";
            }
        }
    }
}

void Gridread::FaceAreaJ()
{
    int AreaJ_Inum = Xmax - 1;
    int AreaJ_Jnum = Ymax;
    int AreaJ_Knum = Zmax - 1;
    real_t xac[3], xbd[3];

    for (int kk = 0; kk < AreaJ_Knum; kk++)
    {
        for (int jj = 0; jj < AreaJ_Jnum; jj++)
        {
            for (int ii = 0; ii < AreaJ_Inum; ii++)
            {
                int id_ori = Xmax * Ymax * kk + Xmax * jj + ii;
                int id_iup = Xmax * Ymax * kk + Xmax * jj + (ii + 1);
                int id_kup = Xmax * Ymax * (kk + 1) + Xmax * jj + ii;
                int id_iup_kup = Xmax * Ymax * (kk + 1) + Xmax * jj + (ii + 1);
                xac[0] = h_coordinate[vector_num * id_kup + 0] - h_coordinate[vector_num * id_iup + 0];
                xac[1] = h_coordinate[vector_num * id_kup + 1] - h_coordinate[vector_num * id_iup + 1];
                xac[2] = h_coordinate[vector_num * id_kup + 2] - h_coordinate[vector_num * id_iup + 2];

                xbd[0] = h_coordinate[vector_num * id_ori + 0] - h_coordinate[vector_num * id_iup_kup + 0];
                xbd[1] = h_coordinate[vector_num * id_ori + 1] - h_coordinate[vector_num * id_iup_kup + 1];
                xbd[2] = h_coordinate[vector_num * id_ori + 2] - h_coordinate[vector_num * id_iup_kup + 2];

                fArea_J[vector_num * id_ori + 0] = (xbd[1] * xac[2] - xbd[2] * xac[1]) / 2.0;
                fArea_J[vector_num * id_ori + 1] = (-1.0) * (xbd[0] * xac[2] - xbd[2] * xac[0]) / 2.0;
                fArea_J[vector_num * id_ori + 2] = (xbd[0] * xac[1] - xbd[1] * xac[0]) / 2.0;

                fArea_J[vector_num * id_ori + 3] = sqrt(fArea_J[vector_num * id_ori + 0] * fArea_J[vector_num * id_ori + 0] + fArea_J[vector_num * id_ori + 1] * fArea_J[vector_num * id_ori + 1] + fArea_J[vector_num * id_ori + 2] * fArea_J[vector_num * id_ori + 2]);

                fArea_J[vector_num * id_ori + 0] = fArea_J[vector_num * id_ori + 0] / fArea_J[vector_num * id_ori + 3];
                fArea_J[vector_num * id_ori + 1] = fArea_J[vector_num * id_ori + 1] / fArea_J[vector_num * id_ori + 3];
                fArea_J[vector_num * id_ori + 2] = fArea_J[vector_num * id_ori + 2] / fArea_J[vector_num * id_ori + 3];
                // ----------------------------------faceAreaJ test pass-------------------------------------
                // cout << std::setprecision(30) << "vector: " << fArea_J[vector_num*id_ori+0] << ", "
                //     << fArea_J[vector_num*id_ori+1] << ", "
                //     << fArea_J[vector_num*id_ori+2] << ", "
                //     << fArea_J[vector_num*id_ori+3] << "\n";
            }
        }
    }
}

void Gridread::FaceAreaK()
{
    int AreaK_Inum = Xmax - 1;
    int AreaK_Jnum = Ymax - 1;
    int AreaK_Knum = Zmax;
    real_t xac[3], xbd[3];

    for (int kk = 0; kk < AreaK_Knum; kk++)
    {
        for (int jj = 0; jj < AreaK_Jnum; jj++)
        {
            for (int ii = 0; ii < AreaK_Inum; ii++)
            {
                int id_ori = Xmax * Ymax * kk + Xmax * jj + ii;
                int id_iup = Xmax * Ymax * kk + Xmax * jj + (ii + 1);
                int id_jup = Xmax * Ymax * kk + Xmax * (jj + 1) + ii;
                int id_iup_jup = Xmax * Ymax * kk + Xmax * (jj + 1) + (ii + 1);
                xac[0] = h_coordinate[vector_num * id_jup + 0] - h_coordinate[vector_num * id_iup + 0];
                xac[1] = h_coordinate[vector_num * id_jup + 1] - h_coordinate[vector_num * id_iup + 1];
                xac[2] = h_coordinate[vector_num * id_jup + 2] - h_coordinate[vector_num * id_iup + 2];

                xbd[0] = h_coordinate[vector_num * id_iup_jup + 0] - h_coordinate[vector_num * id_ori + 0];
                xbd[1] = h_coordinate[vector_num * id_iup_jup + 1] - h_coordinate[vector_num * id_ori + 1];
                xbd[2] = h_coordinate[vector_num * id_iup_jup + 2] - h_coordinate[vector_num * id_ori + 2];

                fArea_K[vector_num * id_ori + 0] = (xbd[1] * xac[2] - xbd[2] * xac[1]) / 2.0;
                fArea_K[vector_num * id_ori + 1] = (-1.0) * (xbd[0] * xac[2] - xbd[2] * xac[0]) / 2.0;
                fArea_K[vector_num * id_ori + 2] = (xbd[0] * xac[1] - xbd[1] * xac[0]) / 2.0;

                fArea_K[vector_num * id_ori + 3] = sqrt(fArea_K[vector_num * id_ori + 0] * fArea_K[vector_num * id_ori + 0] + fArea_K[vector_num * id_ori + 1] * fArea_K[vector_num * id_ori + 1] + fArea_K[vector_num * id_ori + 2] * fArea_K[vector_num * id_ori + 2]);

                fArea_K[vector_num * id_ori + 0] = fArea_K[vector_num * id_ori + 0] / fArea_K[vector_num * id_ori + 3];
                fArea_K[vector_num * id_ori + 1] = fArea_K[vector_num * id_ori + 1] / fArea_K[vector_num * id_ori + 3];
                fArea_K[vector_num * id_ori + 2] = fArea_K[vector_num * id_ori + 2] / fArea_K[vector_num * id_ori + 3];
                // ----------------------------------faceAreaK test pass-------------------------------------
                // cout << std::setprecision(30) << "vector: " << fArea_K[vector_num*id_ori+0] << ", "
                //     << fArea_K[vector_num*id_ori+1] << ", "
                //     << fArea_K[vector_num*id_ori+2] << ", "
                //     << fArea_K[vector_num*id_ori+3] << "\n";
            }
        }
    }
}

real_t Gridread::PyramidVolume(real_t p[3], real_t a[3], real_t b[3], real_t c[3], real_t d[3])
{
    real_t xp[3], xac[3], xbd[3];

    for (int n = 0; n < 3; n++)
    {
        xp[n] = 1.0 / 4.0 * ((a[n] - p[n]) + (b[n] - p[n]) + (c[n] - p[n]) + (d[n] - p[n]));
        xac[n] = c[n] - a[n];
        xbd[n] = d[n] - b[n];
    }

    real_t crossprod_temp[3], PyramidVolume_return_temp;
    crossprod_temp[0] = xac[1] * xbd[2] - xac[2] * xbd[1];
    crossprod_temp[1] = (-1.0) * (xac[0] * xbd[2] - xac[2] * xbd[0]);
    crossprod_temp[2] = xac[0] * xbd[1] - xac[1] * xbd[0];

    PyramidVolume_return_temp = (xp[0] * crossprod_temp[0] + xp[1] * crossprod_temp[1] + xp[2] * crossprod_temp[2]) * (1.0 / 6.0);

    return PyramidVolume_return_temp;
}

void Gridread::GetVolume()
{
    int volume_numI = Xmax - 1;
    int volume_numJ = Ymax - 1;
    int volume_numK = Zmax - 1;

    for (int k = 0; k < volume_numK; k++)
    {
        for (int j = 0; j < volume_numJ; j++)
        {
            for (int i = 0; i < volume_numI; i++)
            {
                int id_ori = Xmax * Ymax * k + Xmax * j + i;
                int id_iup = Xmax * Ymax * k + Xmax * j + (i + 1);
                int id_jup = Xmax * Ymax * k + Xmax * (j + 1) + i;
                int id_kup = Xmax * Ymax * (k + 1) + Xmax * j + i;
                int id_iup_jup = Xmax * Ymax * k + Xmax * (j + 1) + (i + 1);
                int id_iup_kup = Xmax * Ymax * (k + 1) + Xmax * j + (i + 1);
                int id_jup_kup = Xmax * Ymax * (k + 1) + Xmax * (j + 1) + i;
                int id_iup_jup_kup = Xmax * Ymax * (k + 1) + Xmax * (j + 1) + (i + 1);

                // for initial volume of signle cell
                volume_cell[id_ori] = 0.0;

                // calculate the centroid of a given cell
                for (int n = 0; n < 3; n++)
                    centroid[id_ori * vector_num + n] = (h_coordinate[id_ori * vector_num + n] + h_coordinate[id_iup * vector_num + n] + h_coordinate[id_jup * vector_num + n] + h_coordinate[id_kup * vector_num + n] + h_coordinate[id_iup_jup * vector_num + n] + h_coordinate[id_iup_kup * vector_num + n] + h_coordinate[id_jup_kup * vector_num + n] + h_coordinate[id_iup_jup_kup * vector_num + n]) * 0.125;

                real_t pyramid_centroid[3], pyramid_ori[3], pyramid_iup[3], pyramid_jup[3], pyramid_kup[3], pyramid_iup_jup[3], pyramid_iup_kup[3], pyramid_jup_kup[3], pyramid_iup_jup_kup[3];
                for (int n = 0; n < 3; n++)
                {
                    pyramid_centroid[n] = centroid[id_ori * vector_num + n];
                    pyramid_ori[n] = h_coordinate[id_ori * vector_num + n];
                    pyramid_iup[n] = h_coordinate[id_iup * vector_num + n];
                    pyramid_jup[n] = h_coordinate[id_jup * vector_num + n];
                    pyramid_kup[n] = h_coordinate[id_kup * vector_num + n];
                    pyramid_iup_jup[n] = h_coordinate[id_iup_jup * vector_num + n];
                    pyramid_iup_kup[n] = h_coordinate[id_iup_kup * vector_num + n];
                    pyramid_jup_kup[n] = h_coordinate[id_jup_kup * vector_num + n];
                    pyramid_iup_jup_kup[n] = h_coordinate[id_iup_jup_kup * vector_num + n];
                }

                // calculate area of i-lower pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_ori, pyramid_kup, pyramid_jup_kup, pyramid_jup);
                // calculate area of i-upper pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_iup, pyramid_iup_jup, pyramid_iup_jup_kup, pyramid_iup_kup);
                // calculate area of j-lower pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_ori, pyramid_iup, pyramid_iup_kup, pyramid_kup);
                // calculate area of j-upper pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_jup, pyramid_jup_kup, pyramid_iup_jup_kup, pyramid_iup_jup);
                // calculate area of k-lower pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_ori, pyramid_jup, pyramid_iup_jup, pyramid_iup);
                // calculate area of k-upper pyramid
                volume_cell[id_ori] += PyramidVolume(pyramid_centroid, pyramid_kup, pyramid_iup_kup, pyramid_iup_jup_kup, pyramid_jup_kup);

                // ------------------------------------volume test pass----------------------------------------------
                // cout << "volume: " << volume_cell[id_ori] << "\n";
            }
        }
    }
}

void Gridread::OutPut()
{
    // Output ASCII Grid file read
    auto file_name = GridPath;
#ifdef USE_MPI
    file_name += "_rank_" + std::to_string(rank);
#endif
    file_name += ".dat";

    int Onbvar = 16; // one fluid no COP
    std::map<int, std::string> variables_names;
    int index = 0;
    variables_names[index] = "x[m]", index++;
    variables_names[index] = "y[m]", index++;
    variables_names[index] = "z[m]", index++;
    variables_names[index] = "X-i", index++;
    variables_names[index] = "X-j", index++;
    variables_names[index] = "X-k", index++;
    variables_names[index] = "X-Mag", index++;
    variables_names[index] = "Y-i", index++;
    variables_names[index] = "Y-j", index++;
    variables_names[index] = "Y-k", index++;
    variables_names[index] = "Y-Mag", index++;
    variables_names[index] = "Z-i", index++;
    variables_names[index] = "Z-j", index++;
    variables_names[index] = "Z-k", index++;
    variables_names[index] = "Z-Mag", index++;
    variables_names[index] = "Volume", index++;

    std::ofstream out(file_name, std::fstream::out);
    out.setf(std::ios::right);
    out << "title='Out'\nvariables=";
    for (int iVar = 0; iVar < Onbvar - 1; iVar++)
        out << " " << variables_names.at(iVar) << ", ";
    out << variables_names.at(Onbvar - 1) << "\n";
    out << "zone t='Out_";
#ifdef USE_MPI
    out << "_rank_" << std::to_string(rank);
#endif // end USE_MPI
    out << "', i= " << Xmax << ", j= " << Ymax << ", k= " << Zmax << "\n";
    for (size_t kk = 0; kk < Zmax; kk++)
        for (size_t jj = 0; jj < Ymax; jj++)
            for (size_t ii = 0; ii < Xmax; ii++)
            {
                out << std::setw(4) << std::setprecision(6) << h_coordinate[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + 0] << " ";
                out << std::setw(4) << std::setprecision(6) << h_coordinate[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + 1] << " ";
                out << std::setw(4) << std::setprecision(6) << h_coordinate[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + 2] << " ";
                for (size_t n = 0; n < vector_num; n++)
                    out << std::setw(10) << fArea_I[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + n] << " ";
                for (size_t n = 0; n < vector_num; n++)
                    out << std::setw(10) << fArea_J[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + n] << " ";
                for (size_t n = 0; n < vector_num; n++)
                    out << std::setw(10) << fArea_K[(kk * Ymax * Xmax + jj * Xmax + ii) * vector_num + n] << " ";
                out << std::setw(10) << volume_cell[kk * Ymax * Xmax + jj * Xmax + ii] << "\n";
            }
    out.close();
    std::cout << "Write data to file " << file_name << " done." << std::endl;
}

void Gridread::CopyDataFromHost(sycl::queue &q)
{
    // Allocate GPU Memory
    d_coordinate = static_cast<real_t *>(malloc_device(bytes * vector_num, q));
    vol.X = static_cast<real_t *>(malloc_device(bytes * vector_num, q));
    vol.Y = static_cast<real_t *>(malloc_device(bytes * vector_num, q));
    vol.Z = static_cast<real_t *>(malloc_device(bytes * vector_num, q));
    vol.V = static_cast<real_t *>(malloc_device(bytes, q));

    // copy data for GPU
    q.memcpy(d_coordinate, h_coordinate, bytes * vector_num);
    q.memcpy(vol.X, fArea_I, bytes * vector_num);
    q.memcpy(vol.Y, fArea_J, bytes * vector_num);
    q.memcpy(vol.Z, fArea_K, bytes * vector_num);
    q.memcpy(vol.V, volume_cell, bytes);
}