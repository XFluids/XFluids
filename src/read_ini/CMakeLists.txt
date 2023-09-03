target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
) # 将本文件内的库文件链接到上一级父文件夹中，父文件夹只需add_subdirectory
add_library(readini "")
target_sources (readini
  PRIVATE
   setup_ini.cpp 
)
##add libs
add_subdirectory(config)
target_link_libraries(readini
    PUBLIC
    config
)