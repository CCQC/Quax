# NOTE: These paths below need to be edited such that they point to a set of 
# Eigen headers, Python headers, Pybind11 headers, Libint API headers libint2.h libint2.hpp, the rest of the Libint2 headers, and the library location of libint2.a and libint2.so
# Note this requires compiling LIBINT with shared libraries, i.e. the flag -DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=ON must be passed when compiling
# this is because Pybind11 requires -fPIC and -shared flags.
# These paths currently use a version of Libint2 which only has up to 2 angular momentum and 4 orders of differentiation, and is edited to enable ERI derivatives
CC      := g++
# Libint prefix location (where /include, /lib, /include/libint2 is) 
LIBINT_PREFIX := /home/adabbott/Git/libint_test1/libint/build/libint-2.7.0-beta.6/PREFIX
I1 := $(LIBINT_PREFIX)/include
I2 := $(LIBINT_PREFIX)/include/libint2
L1 := $(LIBINT_PREFIX)/lib
# Eigen headers location 
I3 := /home/adabbott/anaconda3/envs/psijax/include/eigen3
# Python headers location 
I4 := /home/adabbott/anaconda3/envs/psijax/include/python3.6m
# Pybind11 headers location 
I5 := /home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include
# HDF5 headers, static and shared libraries 
I6 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include
L2 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib
RPATH := -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"

# This 'TARGETS' suffix should be set to whatever is returned by the command `python3-config --extension-suffix` entered on command line
# and it should match the same python version referenced in the above includes I4 (3.6 in this case)
TARGETS := libint_interface.cpython-36m-x86_64-linux-gnu.so
OBJ     := libint_interface.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

#TODO add -O3
$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -O3 -fPIC -std=c++11 -I $(I1) -I $(I2) -I $(I3) -I $(I4) -I $(I5) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)
$(TARGETS): $(OBJ)
	$(CC) $^ -o $@ -O3 -fPIC -shared -std=c++11 -I $(I1) -I $(I2) -I $(I3) -I $(I4) -I $(I5) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)

