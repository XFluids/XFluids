#!/bin/bash
# ==============================================================================
# Script Name: run_install_AdaptiveCpp.sh
# Description: Automated dependency installer for XFLUIDS (AdaptiveCpp Branch).
#              Supports NVIDIA (CUDA), AMD (ROCm), and Intel (Level Zero/OpenCL).
#              Includes Miniconda, LLVM 16, Boost 1.83, and AdaptiveCpp.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# ================= Configuration & Colors =================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'      # No Color

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
EXTERNAL_DIR="$PROJECT_ROOT/external"
DOWNLOAD_DIR="$EXTERNAL_DIR/downloads"
INSTALL_DIR="$EXTERNAL_DIR/install"
LOG_DIR="$EXTERNAL_DIR/logs"

# URLs
URL_BASE="https://github.com/GPI03248/XFLUIDS_GPI/releases/download/deps_AdaptiveCpp"
URL_ACPP_SRC="$URL_BASE/AdaptiveCpp_source.tar.gz"
URL_BOOST="$URL_BASE/boost-1.83.0.tar.xz"
URL_CONDA="$URL_BASE/Miniconda3-latest-Linux-x86_64.sh"

# SHA256 Checksums (User Verified)
SHA_ACPP="cf036ed766b63b36e737aa92dfdd0306ed5ece76fb5b0c410cd468094d6c0b89"
SHA_BOOST="c5a0688e1f0c05f354bbd0b32244d36085d9ffc9f932e8a18983a9908096f614"
SHA_CONDA="e0b10e050e8928e2eb9aad2c522ee3b5d31d30048b8a9997663a8a460d538cef"

# Filenames
FILE_ACPP_SRC="AdaptiveCpp_source.tar.gz"
FILE_BOOST="boost-1.83.0.tar.xz"
FILE_CONDA="miniconda.sh"
ENV_NAME="XFLUIDS"

# ================= Initialization =================
echo -e "${YELLOW}>>> [Init] Resetting install/log directories (Keeping downloads)...${NC}"
rm -rf "$INSTALL_DIR" "$LOG_DIR"
mkdir -p "$DOWNLOAD_DIR" "$INSTALL_DIR" "$LOG_DIR"

echo -e "${GREEN}>>> [Init] Starting AdaptiveCpp Environment Installation...${NC}"
echo "    Log files will be stored in: $LOG_DIR"

# ================= Helper Function: Robust Download =================
LOG_DOWNLOAD_STATUS="$LOG_DIR/0_download_status.log"

download_file() {
    local url=$1; local output=$2; local expected_sha=$3; local filename=$(basename "$output")
    
    echo -n "    Checking $filename ... "
    
    # 1. Check Local File Integrity
    if [ -f "$output" ]; then
        local current_sha=$(sha256sum "$output" | awk '{print $1}')
        if [ "$current_sha" == "$expected_sha" ]; then
            echo -e "${GREEN}[OK] Verified.${NC}"
            return 0
        else
            echo -e "${YELLOW}[Mismatch] Checksum failed. Deleting...${NC}"
            rm -f "$output"
        fi
    fi

    # 2. Download (Only executed if missing or corrupted)
    echo "    Downloading..."
    if ! command -v curl >/dev/null 2>&1; then
        echo -e "${RED}Error: 'curl' not found. Cannot download dependencies.${NC}"; exit 1
    fi

    if curl -L -C - --retry 5 --retry-delay 2 --connect-timeout 10 --fail --progress-bar -o "$output" "$url"; then
        echo "[$(date)] SUCCESS: Downloaded $filename" >> "$LOG_DOWNLOAD_STATUS"
        local new_sha=$(sha256sum "$output" | awk '{print $1}')
        if [ "$new_sha" != "$expected_sha" ]; then
             echo -e "${RED}Error: Download corrupted! SHA256 mismatch.${NC}"
             exit 1
        fi
    else
        echo -e "${RED}Error: Failed to download $filename.${NC}"
        exit 1
    fi
}

# ================= Step 1: GPU Backend Detection =================
echo -e "${GREEN}[Step 1/7] Detecting System GPU Hardware...${NC}"
BACKEND_FLAG=""; GPU_TOOLKIT_ROOT=""; EXTRA_LINK_FLAGS=""

if command -v nvcc >/dev/null 2>&1 || [ -d "/usr/local/cuda" ]; then
    # --- NVIDIA CUDA ---
    echo -e "      ${BLUE}NVIDIA GPU detected (CUDA)${NC}"
    BACKEND_FLAG="-DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF"
    GPU_TOOLKIT_ROOT="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -d "$GPU_TOOLKIT_ROOT" ]; then 
        GPU_TOOLKIT_ROOT=$(dirname $(dirname $(readlink -f $(command -v nvcc))))
    fi

elif [ -d "/opt/rocm" ] || command -v hipcc >/dev/null 2>&1; then
    # --- AMD ROCm ---
    echo -e "      ${BLUE}AMD GPU detected (ROCm)${NC}"
    BACKEND_FLAG="-DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=ON -DWITH_LEVEL_ZERO_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF"
    GPU_TOOLKIT_ROOT="/opt/rocm"
    
    # [Critical Fix for ROCm Linker Errors]
    # 1. -rpath and -L for ROCm libs (hipRTC)
    # 2. -L/usr/lib/x86_64-linux-gnu for system libs (libnuma) which Conda ld ignores
    EXTRA_LINK_FLAGS="-Wl,-rpath,$GPU_TOOLKIT_ROOT/lib -L$GPU_TOOLKIT_ROOT/lib -L/usr/lib/x86_64-linux-gnu"

else
    # --- Intel Fallback ---
    echo -e "      ${YELLOW}No NVIDIA or AMD GPU detected. Assuming Intel GPU environment.${NC}"
    echo -e "      ${BLUE}Enabling Level Zero and OpenCL backends.${NC}"
    BACKEND_FLAG="-DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=ON -DWITH_OPENCL_BACKEND=ON"
    GPU_TOOLKIT_ROOT=""
fi

# ================= Step 2: Download Dependencies =================
echo -e "${GREEN}[Step 2/7] Verifying and Downloading Dependencies...${NC}"
download_file "$URL_CONDA" "$DOWNLOAD_DIR/$FILE_CONDA" "$SHA_CONDA"
download_file "$URL_BOOST" "$DOWNLOAD_DIR/$FILE_BOOST" "$SHA_BOOST"
download_file "$URL_ACPP_SRC" "$DOWNLOAD_DIR/$FILE_ACPP_SRC" "$SHA_ACPP"
chmod +x "$DOWNLOAD_DIR/$FILE_CONDA"

# ================= Step 3: Install Miniconda & LLVM =================
echo -e "${GREEN}[Step 3/7] Installing Miniconda & LLVM Environment...${NC}"
DIR_CONDA="$INSTALL_DIR/miniconda"
LOG_CONDA="$LOG_DIR/1_conda_install.log"

if [ ! -d "$DIR_CONDA" ]; then
    echo "      Extracting Miniconda..."
    if ! bash "$DOWNLOAD_DIR/$FILE_CONDA" -b -p "$DIR_CONDA" -f > "$LOG_CONDA" 2>&1; then
        echo -e "${RED}Error: Miniconda installation failed! Check log: $LOG_CONDA${NC}"
        exit 1
    fi
fi

# Activate Conda (Temporary for installation)
source "$DIR_CONDA/bin/activate"

# [Critical Fix for Anaconda ToS Error]
# Force use of conda-forge and remove defaults to avoid Terms of Service interactive prompt
echo "      Configuring Conda channels..."
conda config --set always_yes yes >> "$LOG_CONDA" 2>&1
conda config --remove channels defaults >> "$LOG_CONDA" 2>&1 || true
conda config --add channels conda-forge >> "$LOG_CONDA" 2>&1

echo "      Creating environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python=3.10 --override-channels -c conda-forge -y >> "$LOG_CONDA" 2>&1 || {
    echo -e "${RED}Error: Failed to create Conda environment. Check network connectivity.${NC}"
    tail -n 10 "$LOG_CONDA"
    exit 1
}

conda activate "$ENV_NAME"
echo "      Installing LLVM 16 & Toolchain..."
# [Critical Fix] Removed compiler-rt-static (unavailable), added compiler-rt, numactl, sysroot
conda install clang=16 clangxx=16 clangdev=16 llvmdev=16 llvm=16 llvm-tools=16 llvm-openmp \
    compiler-rt=16 numactl sysroot_linux-64 libgcc-devel_linux-64 libstdcxx-devel_linux-64 \
    cmake make ninja pkg-config ncurses zlib zstd libffi libxml2 \
    --override-channels -c conda-forge -y >> "$LOG_CONDA" 2>&1

# [Fix] Conda LLVM path hack: Symlink builtins to where AdaptiveCpp expects them
BUILTIN_DEST_DIR="$CONDA_PREFIX/lib/clang/16/lib/linux"
mkdir -p "$BUILTIN_DEST_DIR"
ACTUAL_BUILTIN=$(find "$CONDA_PREFIX/lib/clang/16" -name "libclang_rt.builtins-x86_64.a" | head -n 1)

if [ -n "$ACTUAL_BUILTIN" ]; then
    # Only create symlink if source and destination are different paths
    TARGET_PATH="$BUILTIN_DEST_DIR/libclang_rt.builtins-x86_64.a"
    if [ "$ACTUAL_BUILTIN" != "$TARGET_PATH" ]; then
        echo "      Creating compiler-rt symlink..."
        ln -sf "$ACTUAL_BUILTIN" "$TARGET_PATH"
    fi
fi

# ================= Step 4: Install Boost =================
echo -e "${GREEN}[Step 4/7] Installing Boost 1.83.0...${NC}"
DIR_BOOST="$INSTALL_DIR/boost"
LOG_BOOST="$LOG_DIR/2_boost_install.log"

# [Critical Fix] Prevent contamination from system-installed Boost (e.g., /usr/include)
unset CPLUS_INCLUDE_PATH
unset CPATH

# Extract
tar -xf "$DOWNLOAD_DIR/$FILE_BOOST" -C "$INSTALL_DIR"
BOOST_SRC_DIR=$(find "$INSTALL_DIR" -maxdepth 1 -type d -name "boost*" | head -n 1)
cd "$BOOST_SRC_DIR"

# Configure & Install
./bootstrap.sh --prefix="$DIR_BOOST" > "$LOG_BOOST" 2>&1
echo "using clang : local : $CONDA_PREFIX/bin/clang++ : <cxxflags>\"-std=c++17 -fPIC -I$BOOST_SRC_DIR\" ;" > user-config.jam

if ! ./b2 install -j$(nproc) -q --user-config=user-config.jam toolset=clang-local link=shared,static threading=multi \
    --with-fiber --with-context --with-thread --with-system --with-atomic --with-filesystem --with-program_options \
    >> "$LOG_BOOST" 2>&1; then
    echo -e "${RED}Error: Boost build failed! Check log: $LOG_BOOST${NC}"
    exit 1
fi

# Clean up source to save space
cd "$EXTERNAL_DIR"
rm -rf "$BOOST_SRC_DIR"

# ================= Step 5: Install AdaptiveCpp =================
echo -e "${GREEN}[Step 5/7] Installing AdaptiveCpp...${NC}"
DIR_ACPP="$INSTALL_DIR/AdaptiveCpp"
LOG_ACPP="$LOG_DIR/3_acpp_build.log"

tar -xf "$DOWNLOAD_DIR/$FILE_ACPP_SRC" -C "$EXTERNAL_DIR"
SRC_ACPP=$(find "$EXTERNAL_DIR" -maxdepth 1 -type d -name "AdaptiveCpp*" | grep -v "install" | head -n 1)
mkdir -p "$SRC_ACPP/build" && cd "$SRC_ACPP/build"

# [Critical Fix] Inject LD_LIBRARY_PATH for the build process (Required for ROCm/HIP discovery)
if [ -n "$GPU_TOOLKIT_ROOT" ]; then 
    export LD_LIBRARY_PATH="$GPU_TOOLKIT_ROOT/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else 
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

echo "      Configuring CMake..."
# [Critical Fix] Added OpenMP CXX flags to ensure CMake detects OpenMP for C++
if ! cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DIR_ACPP" \
    -DBOOST_ROOT="$DIR_BOOST" \
    -DLLVM_DIR="$CONDA_PREFIX/lib/cmake/llvm" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/clang" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/clang++" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_C_FLAGS="-I$CONDA_PREFIX/include" \
    -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include" \
    -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_EXE_LINKER_FLAGS="$EXTRA_LINK_FLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$EXTRA_LINK_FLAGS" \
    -DOpenMP_C_FLAGS="-fopenmp=libomp" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="$CONDA_PREFIX/lib/libomp.so" \
    $BACKEND_FLAG \
    -DCUDA_TOOLKIT_ROOT_DIR="$GPU_TOOLKIT_ROOT" \
    > "$LOG_ACPP" 2>&1; then
    
    echo -e "${RED}Error: CMake configuration failed! Check log: $LOG_ACPP${NC}"
    exit 1
fi

echo "      Compiling AdaptiveCpp..."
if ! make -j$(nproc) >> "$LOG_ACPP" 2>&1; then 
    echo -e "${RED}Error: Compilation failed! Check log: $LOG_ACPP${NC}"
    exit 1
fi

make install >> "$LOG_ACPP" 2>&1
rm -rf "$SRC_ACPP"

# ================= Step 6: Generate Environment Script =================
echo -e "${GREEN}[Step 6/7] Generating XFLUIDS_AdaptiveCpp_setvars.sh...${NC}"
SETVARS_FILE="$PROJECT_ROOT/XFLUIDS_AdaptiveCpp_setvars.sh"

cat > "$SETVARS_FILE" <<EOF
#!/bin/bash
# ==============================================================================
# XFLUIDS Environment Loader (AdaptiveCpp)
# Generated by run_install_AdaptiveCpp.sh on $(date)
# ==============================================================================

# [Sanitization] Clear conflicting variables (Intel/oneAPI) to prevent collision
unset ONEAPI_ROOT
unset CMPLR_ROOT
unset CPATH
unset CPLUS_INCLUDE_PATH

# 1. Boost Environment
export BOOST_ROOT="$DIR_BOOST"
export LD_LIBRARY_PATH="\$BOOST_ROOT/lib:\$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="\$BOOST_ROOT/include:\$CPLUS_INCLUDE_PATH"

# 2. AdaptiveCpp Environment
export ADAPTIVECPP_ROOT="$DIR_ACPP"
export PATH="\$ADAPTIVECPP_ROOT/bin:\$PATH"
export CPATH="\$ADAPTIVECPP_ROOT/include/AdaptiveCpp:\$CPATH"
export LD_LIBRARY_PATH="\$ADAPTIVECPP_ROOT/lib:\$LD_LIBRARY_PATH"

# 3. GPU Backend (if applicable)
if [ -d "$GPU_TOOLKIT_ROOT/lib" ]; then 
    export LD_LIBRARY_PATH="$GPU_TOOLKIT_ROOT/lib:\$LD_LIBRARY_PATH"
fi

echo -e "\033[0;32m>>> [Env] XFLUIDS AdaptiveCpp environment loaded.\033[0m"
EOF

chmod +x "$SETVARS_FILE"

# ================= Step 7: Generate One-Click Build Script =================
echo -e "${GREEN}[Step 7/7] Generating run_build_XFLUIDS_AdaptiveCpp.sh...${NC}"
BUILD_SCRIPT="$PROJECT_ROOT/run_build_XFLUIDS_AdaptiveCpp.sh"

cat > "$BUILD_SCRIPT" <<EOF
#!/bin/bash
set -e
GREEN='\033[0;32m'; NC='\033[0m'

echo -e "\${GREEN}>>> [Build] Initializing AdaptiveCpp Build Environment... \${NC}"
source "$SETVARS_FILE"

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"
rm -rf *

echo -e "\${GREEN}>>> [Build] Preparing CMakeLists.txt...\${NC}"
cp "$PROJECT_ROOT/CMakeLists.txt.ACPP.apk" "$PROJECT_ROOT/CMakeLists.txt"

echo -e "\${GREEN}>>> [Build] Running CMake... \${NC}"
cmake .. \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DACPP_PATH="$DIR_ACPP" \\
    -DBOOST_ROOT="$DIR_BOOST" \\
    -DCOMPILER_PATH="$DIR_CONDA/envs/$ENV_NAME" \\
    -DCMAKE_PREFIX_PATH="$DIR_CONDA/envs/$ENV_NAME" \\
    -DCMAKE_CXX_COMPILER="$DIR_CONDA/envs/$ENV_NAME/bin/clang++" \\
    -DCMAKE_INSTALL_RPATH="$DIR_CONDA/envs/$ENV_NAME/lib;$DIR_BOOST/lib" \\
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo -e "\${GREEN}>>> [Build] Compiling... \${NC}"
make -j\$(nproc)

echo -e "\${GREEN}>>> [Build] Done! Binary in $PROJECT_ROOT/build\${NC}"
EOF

chmod +x "$BUILD_SCRIPT"

echo -e "${GREEN}>>> [Success] Installation Complete! Run: source XFLUIDS_AdaptiveCpp_setvars.sh${NC}"