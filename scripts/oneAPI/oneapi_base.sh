#!/bin/bash
path_script=$(realpath $(dirname "${BASH_SOURCE[0]}"))
chmod +x $path_script/*.txt
chmod +x $path_script/*.sh
source /opt/intel/oneapi/setvars.sh --force --config=$path_script/config_base.txt --include-intel-llvm
# >/dev/null 
# only basic compiler with clang/clang++ and tbb tools are included.
