##CUDA inc
#find_package(NVHPC)#need add NVHPCConfig.cmake to /usr/lib/
#find_package(CUDA)
# add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda)
# add_compile_options(-Xsycl-target-backend)
# add_compile_options(--cuda-gpu-arch=sm_86)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-gpu-arch=sm_86")