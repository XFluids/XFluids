export SCRIPTS_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))
export WKDIR=$(dirname $SCRIPTS_PATH)
echo "-- Begin building cantera"
sudo apt-get install cmake scons -y		# resolve dependencies
CANTERA_SRC=$WKDIR/external/cantera
CANTERA_BUILD=$CANTERA_SRC/build
CANTERA_INSTALL=$WKDIR/external/install/cantera/$2
rm -rf $CANTERA_BUILD $CANTERA_INSTALL $CANTERA_SRC/.sconf_temp $CANTERA_SRC/.sconsign.dblite
mkdir -p $CANTERA_INSTALL
echo "-- Sh from" $SCRIPTS_PATH, "Working DIR:" $WKDIR
echo "-- External cantera SRC:" $CANTERA_SRC
echo "-- External cantera BUILD:" $CANTERA_BUILD
echo "-- External cantera INSTALL:" $CANTERA_INSTALL
source $1/bin/activate base		# initial conda basic python environment
rm -rf $CANTERA_SRC/cantera.conf && cp -rf $SCRIPTS_PATH/cantera_$2.conf $CANTERA_SRC/cantera.conf
cd $CANTERA_SRC && scons build prefix="$CANTERA_INSTALL" > $WKDIR/external/install/build_cantera_$2.log
cd $CANTERA_SRC && scons install prefix="$CANTERA_INSTALL" > $WKDIR/external/install/build_cantera_$2.log
echo "-- End building and installing cantera"
echo "-- cantera buliding info written into:" $WKDIR/external/install/build_cantera_$2.log
