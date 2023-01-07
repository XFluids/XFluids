# Euler-SYCL

## sycl for cuda based on intel/llvm/clang++

1. intel oneapi 2023.0.0 && codeplay Solutions for Nvidia
2. clang++ as compiler && some compiling options

## run

make clean && make GPU=1

````
    export SYCL_DEVICE_FILTER=cuda
	export SYCL_PI_TRACE=1
````

needed befor run exec
