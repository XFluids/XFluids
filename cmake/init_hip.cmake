# #HIP inc
add_compile_options(-Ddevice_id=2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx906")