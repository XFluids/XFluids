#!/bin/bash
set -e
GREEN='\033[0;32m'; NC='\033[0m'
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)

echo -e "${GREEN}>>> [Build] Initializing Local AdaptiveCpp Build Environment... ${NC}"
source "$PROJECT_ROOT/run_load_env_scaleX.sh"

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"
rm -rf *

echo -e "${GREEN}>>> [Build] Running CMake for XFLUIDS (SSCP Mode)... ${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DACPP_PATH="$HOME/Apps/AdaptiveCpp" \
    -DBOOST_ROOT="$HOME/Apps/boost-1.83.0" \
    -DCMAKE_CXX_COMPILER="$HOME/Apps/llvm-17.06/bin/clang++" \
    -DCMAKE_INSTALL_RPATH="$HOME/Apps/AdaptiveCpp/lib;$HOME/Apps/boost-1.83.0/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo -e "${GREEN}>>> [Build] Compiling... ${NC}"
make -j$(nproc)

echo -e "${GREEN}>>> [Build] Done! Binary in $PROJECT_ROOT/build${NC}"
