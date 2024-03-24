export SCRIPTS_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))
export WKDIR=$(dirname $SCRIPTS_PATH)
echo "-- Begin building BOOST"
BOOST_SRC=$WKDIR/external/boost_1_83_0
BOOST_BUILD=$BOOST_SRC
BOOST_INSTALL=${1:-$WKDIR/external/install/boost}
echo "-- Sh from" $SCRIPTS_PATH, "Working DIR:" $WKDIR
echo "-- External BOOST SRC:" $BOOST_SRC
echo "-- External BOOST BUILD:" $BOOST_BUILD
echo "-- External BOOST INSTALL:" $BOOST_INSTALL
cd $WKDIR/external && wget -N https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.bz2
mkdir -p $BOOST_INSTALL
echo "-- Extract BOOST to:" $BOOST_SRC
cd $WKDIR/external && tar --bzip2 -xf $WKDIR/external/boost_1_83_0.tar.bz2
echo "-- Build BOOST into:" $BOOST_INSTALL
cd $BOOST_BUILD && ./bootstrap.sh --prefix=$BOOST_INSTALL --with-libraries=context,fiber,filesystem > $BOOST_INSTALL/../build_boost.log
cd $BOOST_BUILD && ./b2 -q install >> $BOOST_INSTALL/../build_boost.log
echo "-- End building BOOST"
echo "-- BOOST buliding info written into:" $BOOST_INSTALL/../build_boost.log
