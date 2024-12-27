##################################################################
# This bash is to compile an internal AdaptiveCpp of XFluids
# $1 is where the cxx library boost located
# $2 is the prefix of installing the AdaptiveCpp 
# $3 is where the compiler like LLVM, ROCm or CUDA located
# $4 is the target backend for compiling the AdaptiveCpp
# $5 is the target architecture for compiling the AdaptiveCpp
# This bash is compatible with AdaptiveCpp v24.10.0
##################################################################

## settings
export SCRIPTS_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))
export WKDIR=$(dirname $SCRIPTS_PATH)
echo ""
echo "-- Using BOOST_ROOT:" $BOOST_ROOT
echo "-- Working DIR:" $WKDIR
echo "-- Compiling bash from" $SCRIPTS_PATH, "Working DIR:" $WKDIR
BOOST_ROOT=${1:-$(echo $BOOST_ROOT)}
ACPP_SRC=$WKDIR/external/AdaptiveCpp
ACPP_BUILD=$ACPP_SRC/build
ACPP_INSTALL=${2:-$WKDIR/external/install/AdaptiveCpp}
rm -rf $ACPP_BUILD $ACPP_INSTALL
mkdir -p $ACPP_BUILD $ACPP_INSTALL

## cmake
if [ "$4" == "hip" ]; then
ROCM_PATH=${3:-"/opt/rocm"}
  cmake -S $ACPP_SRC -B $ACPP_BUILD -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX=$ACPP_INSTALL \
	-DWITH_ROCM_BACKEND=ON \
	-DWITH_CUDA_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF	\
	-DROCM_PATH=$ROCM_PATH -DLLVM_DIR=$ROCM_PATH/llvm/lib/cmake/llvm \
	-DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
	-DDEFAULT_TARGETS=$4:$5 \
	-DACPP_COMPILER_FEATURE_PROFILE=minimal 
	# \ -DHIPSYCL_NO_DEVICE_MANGLER=ON -DWITH_ACCELERATED_CPU=OFF -DWITH_SSCP_COMPILER=OFF 
# elif [ "$4" == "cuda" ]; then
else
  LLVM_PATH=${3:-"/usr/lib/llvm-14"}
  cmake -S $ACPP_SRC -B $ACPP_BUILD -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX=$ACPP_INSTALL \
  -DLLVM_DIR=$LLVM_PATH/cmake
fi

## info
echo ""
echo "-- Begin building AdaptiveCpp"
echo "-- Internal AdaptiveCpp BUILD TYPE:" $4
echo "-- Internal AdaptiveCpp SRC:" $ACPP_SRC
echo "-- Internal AdaptiveCpp BUILD DIR:" $ACPP_BUILD
echo "-- Internal AdaptiveCpp INSTALL PREFIX:" $ACPP_INSTALL

## build
cd $ACPP_BUILD 
if make -j4 install >> $ACPP_INSTALL/../build_adaptivecpp_$4.log 2>&1; then
  echo "-- End building AdaptiveCpp"
else
  echo "-- Error while Building AdaptiveCpp"
  exit 1
fi
echo "-- AdaptiveCpp buliding info written into:" $INSTALL_DIR/build_adaptivecpp.log
echo ""
