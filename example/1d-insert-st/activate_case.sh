#!/bin/bash
# ==============================================================================
# Script: activate_case.sh
# Description: Configures CMakeLists.txt for 1D Shock Tube case.
# ==============================================================================
set -e
GREEN='\033[0;32m'; NC='\033[0m'

# Get the path to CMakeLists.txt (Grandparent directory)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
CMAKE_FILE="$PROJECT_ROOT/CMakeLists.txt"

TARGET_SAMPLE="1d-insert-st"
TARGET_MODEL="1d-mc-insert-shock-tube"

if [ ! -f "$CMAKE_FILE" ]; then
    echo "Error: CMakeLists.txt not found at $CMAKE_FILE"
    exit 1
fi

echo -e "${GREEN}>>> Configuring CMake for case: $TARGET_SAMPLE${NC}"

# Update CMakeLists.txt
sed -i 's/set(INIT_SAMPLE ".*")/set(INIT_SAMPLE "'"$TARGET_SAMPLE"'")/' "$CMAKE_FILE"
sed -i 's/set(MIXTURE_MODEL ".*")/set(MIXTURE_MODEL "'"$TARGET_MODEL"'")/' "$CMAKE_FILE"

# Verify
grep "set(INIT_SAMPLE" "$CMAKE_FILE"
grep "set(MIXTURE_MODEL" "$CMAKE_FILE"
echo -e "${GREEN}>>> Done. Please re-run cmake and make in the build directory.${NC}"