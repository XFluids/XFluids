#!/bin/bash
name_script=$BASH_SOURCE
path_script=$(cd `dirname $name_script`; pwd)
chmod +x $path_script/*.txt
chmod +x $path_script/*.sh
source /opt/intel/oneapi/setvars.sh --force --config=$path_script/config_base.txt --include-intel-llvm > /dev/null