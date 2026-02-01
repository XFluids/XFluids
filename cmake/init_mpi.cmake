add_compile_options(-DUSE_MPI)

if(DEFINED ENV{MPI_PATH})
    include_directories(BEFORE "$ENV{MPI_PATH}/include")
endif()

if(DEFINED ENV{MPI_PATH})
    message(STATUS "Searching MPI lib in explicit path: $ENV{MPI_PATH}")
    find_library(MPI_CXX
        NAMES libmpi.so libmpicxx.so mpi
        PATHS
            "$ENV{MPI_PATH}/lib/release"
            "$ENV{MPI_PATH}/lib"
            "$ENV{MPI_PATH}/lib64/release"
            "$ENV{MPI_PATH}/lib64"
        NO_DEFAULT_PATH
    )
else()
    find_library(MPI_CXX NAMES libmpi.so libmpicxx.so)
endif()

IF(MPI_CXX)
    message(STATUS "MPI settings: ")

    IF(EXPLICIT_ALLOC)
        add_compile_options(-DEXPLICIT_ALLOC)
        IF(AWARE_MPI)
            add_compile_options(-DAWARE_MPI)
        ENDIF(AWARE_MPI)
        message(STATUS "  AWARE_MPI: ${AWARE_MPI}")
        message(STATUS "  MPI buffer allocate method: explicit")
    ELSE()
        message(STATUS "  MPI buffer allocate method: implicit")
    ENDIF()

    message(STATUS "  MPI_HOME: $ENV{MPI_PATH}")
    if(DEFINED ENV{MPI_PATH})
        message(STATUS "  MPI_INC: $ENV{MPI_PATH}/include added")
    endif()
    message(STATUS "  MPI_CXX lib located: ${MPI_CXX} found")
ELSE()
    message(FATAL_ERROR "Fail to find MPI_CXX library. 
                        \nCurrent MPI_PATH environment is: $ENV{MPI_PATH}. 
                        \nPlease set SYSTEM environment variable MPI_PATH to your MPI installation root (e.g. /opt/intel/.../mpi/2021.11)")
ENDIF()
