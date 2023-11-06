find_library(config NAMES libconfig.a HINTS "${LIBRARY_OUTPUT_PATH}")

IF("${config}" STREQUAL "config-NOTFOUND")
  add_library(config
    STATIC
    ConfigMap.h
    ConfigMap.cpp
    ./inih/ini.cpp
    ./inih/INIReader.cpp
  )

  IF(SYCL_COMPILE_SYSTEM STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET config)
  ENDIF()

ELSE()
  message(STATUS "${config} is located at ${LIBRARY_OUTPUT_PATH}")
  add_library(config STATIC IMPORTED GLOBAL)
  set_property(TARGET config
    PROPERTY IMPORTED_LOCATION ${config})
ENDIF()