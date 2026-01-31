### 项目结构

项目根目录下面有external文件夹，里面有cantera和AdaptiveCpp的源码，同时有pkgs文件夹，里面是boost-1.83.0.tar.xz压缩包和Miniconda3-latest-Linux-x86_64.sh脚本文件。

install.sh的脚本文件放在项目根目录下，需要安装的一共有五个，顺序分别是miniconda, llvm, boost, AdaptiveCpp, cantera。miniconda通过pkgs路径下的脚本文件安装，安装好之后创建一个名为XFLUIDS的虚拟环境，python版本要求3.10，llvm是通过conda-forge频道安装的conda版本，用安装好的conda版本的llvm去安装boost和AdaptiveCpp，最后安装cantera。

在external路径下创建名为install的文件夹，安装的所有内容都放在install路径下。

安装好之后，boost, AdaptiveCpp, cantera三个依赖的文件需要的export内容写到XFLUIDS根目录下的XFLUIDS_setvars.sh文件。

加载XFLUIDS_setvars.sh文件后，在XFLUIDS根目录下创建build文件夹，编译XFLUIDS。

五个依赖的安装过程需要生成对应的log文件，log文件放在external的log文件夹，终端只显示进度，例如"1/5 installing miniconda"字样。

五个依赖安装时，有一些需要cuda的支持，所以，在第一步需要去检索CUDA。

### 安装miniconda

.sh脚本

### 安装llvm

```bash
conda create -n XFLUIDS python=3.10
conda install clang=16 clangxx=16 clangdev=16 llvmdev=16 llvm=16 llvm-tools=16 llvm-openmp libstdcxx-ng cmake make ninja pkg-config ncurses zlib zstd libffi libxml2 -c conda-forge -y
```

### 安装boost

```bash
./bootstrap.sh --prefix="$DIR_BOOST" > "$LOG_DIR/boost_bootstrap.log" 2>&1

# 2. 配置 user-config.jam
# 强制 C++14 (为了兼容 1.83 源码)
# 强制 -I. (为了确保优先读取当前源码的头文件，而不是系统的 1.85)
echo "using clang : local : $DIR_LLVM/bin/clang++ : <cxxflags>\"-std=c++14 -fPIC -I$REAL_BOOST_SRC\" ;" > user-config.jam

# 3. 编译
echo "      Compiling Boost (Clean Environment)..."

./b2 install -j$(nproc) -q \
    --user-config=user-config.jam \
    toolset=clang-local \
    link=shared,static \
    threading=multi \
    --with-fiber \
    --with-context \
    --with-thread \
    --with-system \
    --with-atomic \
    --with-chrono \
    --with-filesystem \
    --with-program_options \
    --with-serialization \
    --with-iostreams \
    --with-regex \
    --with-date_time \
    --with-random \
    --with-container \
    > "$LOG_DIR/boost_install.log" 2>&1; then
```

### 安装AdaptiveCpp

```bash
# export llvm_dir
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="/data/gpi/code/XFLUIDS_cuda/external/install/AdaptiveCpp" \
    -DBOOST_ROOT="/data/gpi/code/XFLUIDS_cuda/external/install/boost" \
    -DLLVM_DIR="$CONDA_PREFIX/lib/cmake/llvm" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/clang" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/clang++" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_C_FLAGS="-I$CONDA_PREFIX/include" \
    -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include" \
	-DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
    -DWITH_CUDA_BACKEND=ON \
    -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF \
    -DWITH_LEVEL_ZERO_BACKEND=OFF 
```

### 安装cantera

```bash
conda install scons packaging numpy cython typing_extensions ruamel.yaml jinja2

# 写入配置文件cantera.conf
prefix = '/data/gpi/code/XFLUIDS_cuda/external/install/cantera'
python_cmd = '$(which python)'
hdf_support = 'n'
system_eigen = 'n'
system_fmt = 'n'
system_highfive = 'n'
system_yamlcpp = 'n'
system_sundials = 'n'
system_blas_lapack = 'n'
boost_inc_dir = '/data/gpi/code/XFLUIDS_cuda/external/install/boost/include'
boost_lib_dir = '/data/gpi/code/XFLUIDS_cuda/external/install/boost/lib'

scons build
scons install
```

### 生成XFLUIDS_setvars.sh

```bash
# boost
export BOOST_ROOT=/path/to/boost
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$BOOST_ROOT/include:$CPLUS_INCLUDE_PATH

# AdaptiveCpp
export ADAPTIVECPP_ROOT=/path/to/AdaptiveCpp
export PATH=$ADAPTIVECPP_ROOT/bin:$PATH
export CPATH=$ADAPTIVECPP_ROOT/include/AdaptiveCpp:$CPATH
export LD_LIBRARY_PATH=$ADAPTIVECPP_ROOT/lib:$LD_LIBRARY_PATH

# cantera
export CANTERA_ROOT=/path/to/cantera
export CPATH=$CANTERA_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$CANTERA_ROOT/lib:$LD_LIBRARY_PATH
```

### 编译XFLUIDS

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCANTERA_ROOT="/data/gpi/code/XFLUIDS_cuda/external/install/cantera" \
    -DACPP_PATH="/data/gpi/code/XFLUIDS_cuda/external/install/AdaptiveCpp" \
    -DBOOST_ROOT="/data/gpi/code/XFLUIDS_cuda/external/install/boost" \
    -DCOMPILER_PATH="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/clang++" \
    -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib;/data/gpi/code/XFLUIDS_cuda/external/install/cantera/lib;/data/gpi/code/XFLUIDS_cuda/external/install/boost/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    
make -j4
```