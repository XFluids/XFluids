export APP_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))
export WKDIR=$APP_PATH/../../
echo "sh from" $APP_PATH, "Working DIR:" $WKDIR

# # OROC
sed -i "s/EIGEN_ALLOC \"RGIF\"/EIGEN_ALLOC \"OROC\"/g" $WKDIR/cmake/init_options.cmake
cd $WKDIR && rm -rf $WKDIR/libs $WKDIR/build && mkdir $WKDIR/build
cd $WKDIR/build && cmake .. && make -j
for i in {1..128};do
    echo "Test for "$i "species"
	cd $WKDIR/build && $WKDIR/build/XFLUIDS -dev=$1,$2,$3 -run=400,0,0 > $WKDIR/OROC-$i.log
	sed -i "s/$i/`expr $i + 1`/g" $WKDIR/runtime.dat/1d-mc-eigen-shock-tube/case_setup.h
	cd $WKDIR/build && rm -rf $WKDIR/build/output/* && make clean && make -j 
done
	sed -i "s/129/1/g" $WKDIR/runtime.dat/1d-mc-eigen-shock-tube/case_setup.h

# # RGIF
sed -i "s/EIGEN_ALLOC \"OROC\"/EIGEN_ALLOC \"RGIF\"/g" $WKDIR/cmake/init_options.cmake
cd $WKDIR && rm -rf $WKDIR/libs $WKDIR/build && mkdir $WKDIR/build
cd $WKDIR/build && cmake .. && make -j
for i in {1..128};do
    echo "Test for "$i "species"
	cd $WKDIR/build && $WKDIR/build/XFLUIDS -dev=$1,$2,$3 -run=400,0,0 > $WKDIR/RGIF-$i.log
	sed -i "s/$i/`expr $i + 1`/g" $WKDIR/runtime.dat/1d-mc-eigen-shock-tube/case_setup.h
	cd $WKDIR/build && rm -rf $WKDIR/build/output/* && make clean && make -j 
done
	sed -i "s/129/1/g" $WKDIR/runtime.dat/1d-mc-eigen-shock-tube/case_setup.h
