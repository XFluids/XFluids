export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

make clean

make USE_DP=1 -j

./EulerSYCL