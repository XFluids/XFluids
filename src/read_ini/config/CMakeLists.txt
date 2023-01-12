add_subdirectory(inih)
add_library(config 
  STATIC
    ConfigMap.h
    ConfigMap.cpp
)

target_link_libraries(config 
  PRIVATE 
    inih
)

