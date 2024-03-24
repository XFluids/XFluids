export SCRIPTS_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))
export WKDIR=$(dirname $SCRIPTS_PATH)
echo "-- Begin building AdaptiveCpp"
BOOST_ROOT=${1:-$(echo $BOOST_ROOT)}
AdaptiveCpp_SRC=$WKDIR/external/AdaptiveCpp
AdaptiveCpp_BUILD=$AdaptiveCpp_SRC/build
AdaptiveCpp_INSTALL=${2:-$WKDIR/external/install/AdaptiveCpp}
rm -rf $AdaptiveCpp_BUILD $AdaptiveCpp_INSTALL
mkdir -p $AdaptiveCpp_BUILD $AdaptiveCpp_INSTALL
echo "-- Sh from" $SCRIPTS_PATH, "Working DIR:" $WKDIR
echo "-- Using BOOST_ROOT:" $BOOST_ROOT
echo "-- External AdaptiveCpp SRC:" $AdaptiveCpp_SRC
echo "-- External AdaptiveCpp BUILD:" $AdaptiveCpp_BUILD
echo "-- External AdaptiveCpp INSTALL:" $AdaptiveCpp_INSTALL
cmake -S $AdaptiveCpp_SRC -B $AdaptiveCpp_BUILD -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_INSTALL_PREFIX=$AdaptiveCpp_INSTALL
cd $AdaptiveCpp_BUILD && make -j install >> $AdaptiveCpp_INSTALL/../build_adaptivecpp.log 2>&1
echo "-- End building AdaptiveCpp"
echo "-- AdaptiveCpp buliding info written into:" $INSTALL_DIR/build_adaptivecpp.log
