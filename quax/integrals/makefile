# NOTE: These paths below need to be edited such that they point to a set of 
# Eigen headers, Python headers, Pybind11 headers, Libint API headers libint2.h libint2.hpp,
# the rest of the Libint2 headers, and the library location of libint2.a
CC      := g++
# Options passed to compiler, add "-fopenmp" if intending to use OpenMP
CFLAGS  := -O3 -fPIC -fopenmp
# Libint prefix location (where /include, /include/libint2, /lib, /share are located) 
LIBINT_PREFIX := $(shell python3-config --prefix)
# Conda prefix location, it is suggested to use conda to install nearly all dependencies
CONDA_PREFIX := $(shell python3-config --prefix)

I1 := $(LIBINT_PREFIX)/include
I2 := $(LIBINT_PREFIX)/include/libint2
L1 := $(LIBINT_PREFIX)/lib
# Eigen headers location 
I3 := $(CONDA_PREFIX)/eigen3
# HDF5 headers, static and shared libraries 
I6 := $(CONDA_PREFIX)/include
L2 := $(CONDA_PREFIX)/lib
# Edit path in quotes to be same location as L2 definition above
RPATH := -Wl,-rpath,"$(CONDA_PREFIX)/lib"

# This 'TARGETS' suffix should be set to whatever is returned by the command `python3-config --extension-suffix` entered on command line.
# and it should match the same python version referenced in the above include path for I4 := (3.7 in this case)
TARGETS := libint_interface$(shell python3-config --extension-suffix)
OBJ     := libint_interface.o

# Rest is boilerplate. Do not edit unless you know what you're doing.
.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(OBJ)

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS) -I $(I1) -I $(I2) -I $(I3) $(shell python3 -m pybind11 --includes) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)
$(TARGETS): $(OBJ)
	$(CC) $^ -o $@ $(CFLAGS) -shared -I $(I1) -I $(I2) -I $(I3) -I $(I4) -I $(I5) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)

