#!/bin/bash
# ==============================================================================
# Script Name: run_install_oneAPI.sh
# Description: Automated dependency installer for XFLUIDS (oneAPI Branch).
#              Installs Intel oneAPI BaseKit 2025.0, Plugins, and Boost 1.83.
# ==============================================================================

set -e

# --- Configuration & Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
EXTERNAL_DIR="$PROJECT_ROOT/external"
DOWNLOAD_DIR="$EXTERNAL_DIR/downloads"
INSTALL_DIR="$EXTERNAL_DIR/install"
LOG_DIR="$EXTERNAL_DIR/logs"

# URLs
URL_BASE="https://github.com/XFluids/XFluids/releases/download/deps_oneAPI"
URL_BOOST="$URL_BASE/boost-1.83.0.tar.xz"
URL_PART_00="$URL_BASE/oneAPI_installer_part_00"
URL_PART_01="$URL_BASE/oneAPI_installer_part_01"
URL_PLUGIN_CUDA="$URL_BASE/oneapi-for-nvidia-gpus-2025.0.0-cuda-12.0-linux.sh"
URL_PLUGIN_ROCM5="$URL_BASE/oneapi-for-amd-gpus-2025.0.0-rocm-5.4.3-linux.sh"
URL_PLUGIN_ROCM6="$URL_BASE/oneapi-for-amd-gpus-2025.0.0-rocm-6.1.0-linux.sh"

# Verified SHA256 Checksums
SHA_BOOST="c5a0688e1f0c05f354bbd0b32244d36085d9ffc9f932e8a18983a9908096f614"
SHA_PART_00="298a046996e9b609c7e00e529662e3480dd5d540e15fb85da68bde1473797aea"
SHA_PART_01="33bdaf51b308c9dc0c38afbeebaac6b04309712a91971ea915826c9f85117e6c"
SHA_PLUGIN_CUDA="264a43d2e07c08eb31d6483fb1c289a6b148709e48e9a250efc1b1e9a527feb6"
SHA_PLUGIN_ROCM5="04270fd737460478840573b61616ae9b6fbb512901c01d5894d6a70904e1de9d"
SHA_PLUGIN_ROCM6="2c5a147e82f0e995b9c0457b53967cc066d5741d675cb64cb9eba8e3c791a064"

FILE_ONEAPI_INSTALLER="oneapi_basekit_2025.0.sh"
FILE_BOOST_ARCHIVE="boost-1.83.0.tar.xz"

# --- Initialization ---
echo -e "${YELLOW}>>> [Init] Resetting installation directories (Keeping downloads)...${NC}"
rm -rf "$INSTALL_DIR" "$LOG_DIR"
mkdir -p "$DOWNLOAD_DIR" "$INSTALL_DIR" "$LOG_DIR"

echo -e "${GREEN}>>> [Init] Starting oneAPI Environment Installation...${NC}"
echo "    Log files will be stored in: $LOG_DIR"

# --- Helper Function: Verify and Download ---
LOG_DOWNLOAD_STATUS="$LOG_DIR/0_download_status.log"

download_file() {
    local url=$1; local output=$2; local expected_sha=$3; local filename=$(basename "$output")
    echo -n "    Checking $filename ... "
    if [ -f "$output" ]; then
        local current_sha=$(sha256sum "$output" | awk '{print $1}')
        if [ "$current_sha" == "$expected_sha" ]; then
            echo -e "${GREEN}[OK] Verified.${NC}"; return 0
        fi
        rm -f "$output"
    fi
    echo "    Downloading..."
    if curl -L -C - --retry 5 --retry-delay 2 --connect-timeout 15 --fail --progress-bar -o "$output" "$url"; then
        echo "[$(date)] SUCCESS: $filename" >> "$LOG_DOWNLOAD_STATUS"
        local new_sha=$(sha256sum "$output" | awk '{print $1}')
        if [ "$new_sha" != "$expected_sha" ]; then
             echo -e "${RED}Error: SHA256 mismatch for $filename${NC}"; exit 1
        fi
    else
        echo -e "${RED}Error: Download failed for $filename.${NC}"; exit 1
    fi
}

# --- Step 1: Detect Hardware ---
echo -e "${GREEN}[Step 1/6] Detecting System GPU Hardware...${NC}"
TARGET_VENDOR="INTEL"; PLUGIN_URL=""; PLUGIN_FILE=""; PLUGIN_SHA=""

if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "      ${BLUE}NVIDIA GPU detected (CUDA).${NC}"
    TARGET_VENDOR="NVIDIA"; PLUGIN_URL="$URL_PLUGIN_CUDA"; PLUGIN_FILE="oneapi_plugin_cuda.sh"; PLUGIN_SHA="$SHA_PLUGIN_CUDA"
elif [ -d "/opt/rocm" ] || command -v hipcc >/dev/null 2>&1; then
    echo -e "      ${BLUE}AMD GPU detected (ROCm).${NC}"
    TARGET_VENDOR="AMD"
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "6.0")
    if [[ "$ROCM_VER" == 6.* ]]; then
        PLUGIN_URL="$URL_PLUGIN_ROCM6"; PLUGIN_FILE="oneapi_plugin_rocm6.sh"; PLUGIN_SHA="$SHA_PLUGIN_ROCM6"
    else
        PLUGIN_URL="$URL_PLUGIN_ROCM5"; PLUGIN_FILE="oneapi_plugin_rocm5.sh"; PLUGIN_SHA="$SHA_PLUGIN_ROCM5"
    fi
else
    echo -e "      ${BLUE}No 3rd-party GPU found. Defaulting to Intel GPU support.${NC}"
fi

# --- Step 2: Download ---
echo -e "${GREEN}[Step 2/6] Verifying Dependencies...${NC}"
download_file "$URL_PART_00" "$DOWNLOAD_DIR/oneapi_part_00" "$SHA_PART_00"
download_file "$URL_PART_01" "$DOWNLOAD_DIR/oneapi_part_01" "$SHA_PART_01"
if [ ! -f "$DOWNLOAD_DIR/$FILE_ONEAPI_INSTALLER" ]; then
    echo "    Merging oneAPI installer parts..."
    cat "$DOWNLOAD_DIR/oneapi_part_00" "$DOWNLOAD_DIR/oneapi_part_01" > "$DOWNLOAD_DIR/$FILE_ONEAPI_INSTALLER"
    chmod +x "$DOWNLOAD_DIR/$FILE_ONEAPI_INSTALLER"
fi
download_file "$URL_BOOST" "$DOWNLOAD_DIR/$FILE_BOOST_ARCHIVE" "$SHA_BOOST"
if [ "$TARGET_VENDOR" != "INTEL" ]; then
    download_file "$PLUGIN_URL" "$DOWNLOAD_DIR/$PLUGIN_FILE" "$PLUGIN_SHA"
    chmod +x "$DOWNLOAD_DIR/$PLUGIN_FILE"
fi

# --- Step 3: Install oneAPI Toolkit ---
echo -e "${GREEN}[Step 3/6] Installing Intel oneAPI Base Toolkit...${NC}"
LOG_ONEAPI="$LOG_DIR/1_oneapi_install.log"
DIR_ONEAPI="$INSTALL_DIR/oneapi"
FAKE_ROOT="$INSTALL_DIR/fake_root"
mkdir -p "$DIR_ONEAPI" "$FAKE_ROOT"

run_isolated() {
    env -i HOME="$FAKE_ROOT" XDG_CONFIG_HOME="$FAKE_ROOT/.config" \
        XDG_CACHE_HOME="$FAKE_ROOT/.cache" XDG_DATA_HOME="$FAKE_ROOT/.local/share" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" USER="$(whoami)" "$@"
}

echo "      Running Silent Installer (Wait...)"
if ! run_isolated bash "$DOWNLOAD_DIR/$FILE_ONEAPI_INSTALLER" -a -s --action install --components default --eula accept --install-dir "$DIR_ONEAPI" > "$LOG_ONEAPI" 2>&1; then
    echo -e "${RED}Error: oneAPI install failed. Check $LOG_ONEAPI${NC}"; exit 1
fi

if [ "$TARGET_VENDOR" != "INTEL" ]; then
    echo -e "      Installing Codeplay GPU Plugin..."
    LOG_PLUGIN="$LOG_DIR/2_plugin_install.log"
    if ! run_isolated bash "$DOWNLOAD_DIR/$PLUGIN_FILE" -y --install-dir "$DIR_ONEAPI" > "$LOG_PLUGIN" 2>&1; then
        echo -e "${RED}Error: Plugin install failed. Check $LOG_PLUGIN${NC}"; exit 1
    fi
fi

# --- Step 4: Install Boost ---
echo -e "${GREEN}[Step 4/6] Installing Boost 1.83.0 via Intel Clang...${NC}"
DIR_BOOST="$INSTALL_DIR/boost"
LOG_SETVARS="$LOG_DIR/3_boost_setvars.log"
ONEAPI_SETVARS="$DIR_ONEAPI/setvars.sh"

source "$ONEAPI_SETVARS" --include-intel-llvm --force > "$LOG_SETVARS" 2>&1 || true
unset CPLUS_INCLUDE_PATH; unset CPATH # Prevent system header contamination

REAL_CLANG=$(command -v clang++ || true)
if [[ -z "$REAL_CLANG" ]]; then echo -e "${RED}Error: Intel clang++ not found!${NC}"; exit 1; fi

tar -xf "$DOWNLOAD_DIR/$FILE_BOOST_ARCHIVE" -C "$INSTALL_DIR"
BOOST_SRC_DIR=$(find "$INSTALL_DIR" -maxdepth 1 -type d -name "boost*" | head -n 1)
cd "$BOOST_SRC_DIR"

WRAPPER_SCRIPT="$INSTALL_DIR/clang_wrapper.sh"
echo -e "#!/bin/bash\nif [[ \"\$@\" == *\"--version\"* ]]; then echo \"clang version 16.0.0\"; else exec \"$REAL_CLANG\" \"\$@\"; fi" > "$WRAPPER_SCRIPT"
chmod +x "$WRAPPER_SCRIPT"

LOG_BOOST="$LOG_DIR/4_boost_build.log"
./bootstrap.sh --prefix="$DIR_BOOST" --with-toolset=clang > "$LOG_BOOST" 2>&1
echo "using clang : : $WRAPPER_SCRIPT : <cxxflags>\"-std=c++17 -fPIC\" ;" > user-config.jam

if ! ./b2 install -j$(nproc) -q --user-config=user-config.jam toolset=clang link=shared,static threading=multi \
    --with-fiber --with-context --with-thread --with-system --with-atomic --with-filesystem --with-program_options \
    >> "$LOG_BOOST" 2>&1; then
    echo -e "${RED}Error: Boost build failed! Check $LOG_BOOST${NC}"; exit 1
fi
cd "$EXTERNAL_DIR"; rm -rf "$BOOST_SRC_DIR" "$FAKE_ROOT" "$WRAPPER_SCRIPT"

# --- Step 5: Environment Script ---
echo -e "${GREEN}[Step 5/6] Generating XFLUIDS_oneAPI_setvars.sh...${NC}"
SETVARS_FILE="$PROJECT_ROOT/XFLUIDS_oneAPI_setvars.sh"
cat > "$SETVARS_FILE" <<EOF
#!/bin/bash
# Sanitization: Clear conflicting variables from other stacks
unset ADAPTIVECPP_ROOT; unset CPATH; unset CPLUS_INCLUDE_PATH; unset LIBRARY_PATH
ONEAPI_SETVARS="$DIR_ONEAPI/setvars.sh"
if [ -f "\$ONEAPI_SETVARS" ]; then source "\$ONEAPI_SETVARS" --include-intel-llvm --force > /dev/null 2>&1 || true; fi
export BOOST_ROOT="$DIR_BOOST"
export LD_LIBRARY_PATH="\$BOOST_ROOT/lib:\$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="\$BOOST_ROOT/include:\$CPLUS_INCLUDE_PATH"
export MPI_PATH="$DIR_ONEAPI/mpi/2021.14"
echo -e "\033[0;32m>>> [Env] XFLUIDS oneAPI environment loaded.\033[0m"
EOF
chmod +x "$SETVARS_FILE"

# --- Step 6: Build Script ---
echo -e "${GREEN}[Step 6/6] Generating run_build_XFLUIDS_oneAPI.sh...${NC}"
BUILD_SCRIPT="$PROJECT_ROOT/run_build_XFLUIDS_oneAPI.sh"
cat > "$BUILD_SCRIPT" <<EOF
#!/bin/bash
set -e
GREEN='\033[0;32m'; NC='\033[0m'
echo -e "\${GREEN}>>> [Build] Initializing oneAPI Build Environment...\${NC}"
source "$SETVARS_FILE"
echo -e "\${GREEN}>>> [Build] Preparing CMakeLists.txt...\${NC}"
cp "$PROJECT_ROOT/CMakeLists.txt.oneAPI.apk" "$PROJECT_ROOT/CMakeLists.txt"
mkdir -p "$PROJECT_ROOT/build" && cd "$PROJECT_ROOT/build" && rm -rf *
echo -e "\${GREEN}>>> [Build] Running CMake...\${NC}"
cmake .. -DBOOST_ROOT="$DIR_BOOST" -DCMAKE_EXE_LINKER_FLAGS="-L$DIR_BOOST/lib -lboost_filesystem -lboost_system"
echo -e "\${GREEN}>>> [Build] Compiling...\${NC}"
make -j\$(nproc)
echo -e "\${GREEN}>>> [Build] Done! Binary in $PROJECT_ROOT/build\${NC}"
EOF
chmod +x "$BUILD_SCRIPT"

echo -e "${GREEN}>>> [Success] Installation Complete! Run: source XFLUIDS_oneAPI_setvars.sh${NC}"