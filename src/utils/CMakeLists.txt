add_subdirectory(atttribute)

add_library(utils INTERFACE)

target_link_libraries(utils
INTERFACE
attribute
)
