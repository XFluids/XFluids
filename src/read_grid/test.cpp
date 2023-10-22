#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string.h>
#include <float.h>
#include <ctime>

#include "setup.h"
#include "readgrid.h"
#include <string.h>

int main()
{

    // auto device = sycl::platform::get_platforms()[2].get_devices()[0];
    // sycl::queue q(device, dpc_common::exception_handler);

    int begin = 1;
    Gridread grid(begin);
    // Global.ReadINP();
    grid.AllocateMemory();
    grid.ReadGridBlock();
    grid.FaceAreaI();
    grid.FaceAreaJ();
    grid.FaceAreaK();
    grid.volume();

    return 0;
}