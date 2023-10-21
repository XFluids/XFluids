add_library(inih
    STATIC
        ini.h
        INIReader.h
        ini.cpp
        INIReader.cpp
)

IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET inih)
ENDIF()
