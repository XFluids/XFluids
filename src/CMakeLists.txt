add_executable(${PROJECT_NAME} "main.cpp")

add_library(cxxsrc
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/fluid_sycl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sycl_solver.cpp
)

add_subdirectory(read_ini) # 添加读取文件、mpi支持等的使用# #add libs
target_link_libraries(${PROJECT_NAME}
  PRIVATE
  cxxsrc # 自编C++源代码库
  readini
)