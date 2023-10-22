target_include_directories(cxxsrc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
) # 将本文件内的库文件链接到上一级父文件夹中，父文件夹只需add_subdirectory
add_library(readgrid "")
target_sources(readgrid
  PRIVATE
  readgrid.cpp
)

IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
  add_sycl_to_target(TARGET readgrid)
ENDIF()