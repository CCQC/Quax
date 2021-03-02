# NOTE: These paths below need to be edited such that they point to a set of 
# Eigen headers, Python headers, Pybind11 headers, Libint API headers libint2.h libint2.hpp, the rest of the Libint2 headers, and the library location of libint2.a,
CC      := g++
# Libint prefix location (where /include, /include/libint2, /lib, /share are located) 
LIBINT_PREFIX := /home/adabbott/Git/libint_am3/libint/build/libint-2.7.0-beta.6/PREFIX

I1 := $(LIBINT_PREFIX)/include
I2 := $(LIBINT_PREFIX)/include/libint2
L1 := $(LIBINT_PREFIX)/lib
# Eigen headers location 
I3 := /home/adabbott/anaconda3/envs/quax/include/eigen3
# Python headers location 
I4 := /home/adabbott/anaconda3/envs/quax/include/python3.7m
# Pybind11 headers location 
I5 := /home/adabbott/anaconda3/envs/quax/lib/python3.7/site-packages/pybind11/include
# HDF5 headers, static and shared libraries 
I6 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/include
L2 := /home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib
# Edit path in quotes to be same location as L2 definition above
RPATH := -Wl,-rpath,"/home/adabbott/Git/hdf5/hdf5-1.12.0/PREFIX/lib"

# This 'TARGETS' suffix should be set to whatever is returned by the command `python3-config --extension-suffix` entered on command line.
# and it should match the same python version referenced in the above include path for I4 := (3.7 in this case)
TARGETS := libint_interface.cpython-37m-x86_64-linux-gnu.so
OBJ     := libint_interface.o

# Rest is boilerplate. Do not edit unless you know what you're doing.
.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -O3 -fPIC -std=c++11 -I $(I1) -I $(I2) -I $(I3) -I $(I4) -I $(I5) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)
$(TARGETS): $(OBJ)
	$(CC) $^ -o $@ -O3 -fPIC -shared -std=c++11 -I $(I1) -I $(I2) -I $(I3) -I $(I4) -I $(I5) -I $(I6) -lint2 -L $(L1) -lhdf5 -lhdf5_cpp -L $(L2) $(RPATH)

