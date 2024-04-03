#pragma once

// =======================================================
//    Set Domain size
#define MARCO_DOMAIN()        \
    int Xmax = bl.Xmax;       \
    int Ymax = bl.Ymax;       \
    int Zmax = bl.Zmax;       \
    int X_inner = bl.X_inner; \
    int Y_inner = bl.Y_inner; \
    int Z_inner = bl.Z_inner;

#define MARCO_DOMAIN_GHOST()    \
    int Xmax = bl.Xmax;         \
    int Ymax = bl.Ymax;         \
    int Zmax = bl.Zmax;         \
    int X_inner = bl.X_inner;   \
    int Y_inner = bl.Y_inner;   \
    int Z_inner = bl.Z_inner;   \
    int Bwidth_X = bl.Bwidth_X; \
    int Bwidth_Y = bl.Bwidth_Y; \
    int Bwidth_Z = bl.Bwidth_Z;

// =======================================================
//    get Roe values insde Reconstructflux
#define MARCO_ROE()                               \
    real_t D = sycl::sqrt(rho[id_r] / rho[id_l]); \
    real_t D1 = _DF(1.0) / (D + _DF(1.0));        \
    real_t _u = (u[id_l] + D * u[id_r]) * D1;     \
    real_t _v = (v[id_l] + D * v[id_r]) * D1;     \
    real_t _w = (w[id_l] + D * w[id_r]) * D1;     \
    real_t _H = (H[id_l] + D * H[id_r]) * D1;     \
    real_t _P = (p[id_l] + D * p[id_r]) * D1;     \
    real_t _rho = sycl::sqrt(rho[id_r] * rho[id_l]);

// =======================================================
//    Get c2
#ifdef COP
#define MARCO_GETC2() MARCO_COPC2()
// MARCO_COPC2() //MARCO_COPC2_ROB()
#else
#define MARCO_GETC2() MARCO_NOCOPC2()
#endif // end COP

// //    Get error out of c2 arguments
#if ESTIM_OUT

#define MARCO_ERROR_OUT()                   \
    eb1[id_l] = b1;                         \
    eb3[id_l] = b3;                         \
    ec2[id_l] = c2;                         \
    for (size_t nn = 0; nn < NUM_COP; nn++) \
    {                                       \
        ezi[nn + NUM_COP * id_l] = z[nn];   \
    }
#else

#define MARCO_ERROR_OUT() ;

#endif // end ESTIM_OUT

// =======================================================
//    Loop in Output
#define MARCO_OUTLOOP                                        \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)                \
        for (int j = VTI.minY; j < VTI.maxY; j++)            \
            for (int i = VTI.minX; i < VTI.maxX; i++)

#define MARCO_POUTLOOP(BODY)                          \
    for (int k = VTI.minZ; k < VTI.maxZ; k++)         \
        for (int j = VTI.minY; j < VTI.maxY; j++)     \
        {                                             \
            for (int i = VTI.minX; i < VTI.maxX; i++) \
                out << BODY << " ";                   \
            out << "\n";                              \
        }

#define MARCO_COUTLOOP                                       \
    outFile.write((char *)&nbOfWords, sizeof(unsigned int)); \
    for (int k = minZ; k < maxZ; k++)                        \
        for (int j = minY; j < maxY; j++)                    \
            for (int i = minX; i < maxX; i++)
