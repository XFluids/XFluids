set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-gpu-arch=${ARCH}")