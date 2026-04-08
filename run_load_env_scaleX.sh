#!/bin/bash
# ==============================================================================
# XFLUIDS Local Environment Loader (AdaptiveCpp SSCP)
# ==============================================================================

# 1. Local Boost Environment
export BOOST_ROOT="$HOME/Apps/boost-1.83.0"
export LD_LIBRARY_PATH="$BOOST_ROOT/lib:$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="$BOOST_ROOT/include:$CPLUS_INCLUDE_PATH"

# 2. Local AdaptiveCpp Environment
export ADAPTIVECPP_ROOT="$HOME/Apps/AdaptiveCpp"
export PATH="$ADAPTIVECPP_ROOT/bin:$PATH"
export CPATH="$ADAPTIVECPP_ROOT/include/AdaptiveCpp:$CPATH"
export LD_LIBRARY_PATH="$ADAPTIVECPP_ROOT/lib:$LD_LIBRARY_PATH"

echo -e "\033[0;32m>>>[Env] Local XFLUIDS AdaptiveCpp environment (SSCP) loaded.\033[0m"
