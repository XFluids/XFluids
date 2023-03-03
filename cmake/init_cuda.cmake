# #CUDA inc
add_compile_options(-DPform_id=2)
add_compile_options(-Dnum_GPUs=2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-gpu-arch=sm_75")