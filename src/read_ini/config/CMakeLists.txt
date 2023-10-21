add_subdirectory(inih)
add_library(config
  STATIC
  ConfigMap.h
  ConfigMap.cpp
)

IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
  add_sycl_to_target(TARGET config)
ENDIF()
  
target_link_libraries(config
  PRIVATE
  inih
)
