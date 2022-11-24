################################################################################
# Parallel solver of Euler equation on Intel GPUs with DPC++
# ################################################################################

# APP Name
EXECUTABLE	:= EulerSYCL

# Add source files here
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
CCFILES := $(call rwildcard,./,*.cpp)

#C++ compiler
CXX := dpcpp#icpc

#C++ compiler flags
ifeq ($(dbg),1)
CCFLAGS   += -g
TARGET    := debug
LIBPARALLEL = -ltbb_debug -ltbbmalloc_debug
else
CCFLAGS   += -O3
TARGET    := release
LIBPARALLEL = -ltbb -ltbbmalloc
endif 
# openmp flags
CCFLAGS   += -fopenmp 

# double precsion flag
ifeq ($(USE_DP),1)
CCFLAGS   += -DUSE_DP
endif

# Extra user flags
CXXFLAGS   := -DUNIX -std=c++20#17

# PREPARE AND COMPILE:
CXXFLAGS += $(CCFLAGS)

LIBS := -L/usr/lib/ -lrt

# Common LD_flags
LD_FLAGS       :=  -lOpenCL -lsycl $(LIBPARALLEL)#$(LIBS) -lrt

# Location of output and obj files
OUTDATADIR      ?= ./outdata
OBJECTDIR       ?= ./obj

# Set up object files
OBJECTS 	 =  $(patsubst %.cpp,$(OBJECTDIR)/%.cpp.o,$(notdir $(CCFILES)))

# Target rules
all: directories $(EXECUTABLE)

# Link commands:
$(OBJECTDIR)/%.cpp.o : %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
$(EXECUTABLE): directories $(OBJECTS) Makefile
	$(CXX) $(CCFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LD_FLAGS)

directories:
	    mkdir -p $(OBJECTDIR)
		mkdir -p $(OUTDATADIR)

clean:
	        rm -rf $(EXECUTABLE) $(OBJECTDIR)/* *.bin
		 
reset:
	        rm -rf $(EXECUTABLE) $(OBJECTDIR)/* *.bin $(OBJECTDIR) $(OUTDATADIR)