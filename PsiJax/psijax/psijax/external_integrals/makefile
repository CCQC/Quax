# NOTE: These paths below need to be edited such that they point to a set of 
# Eigen headers, Python headers, Pybind11 headers, Libint API headers libint2.h libint2.hpp, the rest of the Libint2 headers, and the library location of libint2.a and libint2.so
# Note this requires compiling LIBINT with shared libraries, i.e. the flag -DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=ON must be passed when compiling
# this is because Pybind11 requires -fPIC and -shared flags.
# These paths currently use a version of Libint2 which only has up to 2 angular momentum and 4 orders of differentiation, and is edited to enable ERI derivatives
CC      := g++
INCL1   := /home/adabbott/anaconda3/envs/psijax/include/eigen3
INCL2   := /home/adabbott/anaconda3/envs/psijax/include/python3.6m
INCL3   := /home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include
INCL4   := /home/adabbott/Git/libint_dev/libint-2.7.0-beta.6/PREFIX/include
INCL5   := /home/adabbott/Git/libint_dev/libint-2.7.0-beta.6/PREFIX/include/libint2
LIBS    := /home/adabbott/Git/libint_dev/libint-2.7.0-beta.6/PREFIX/lib
RPATH   := -Wl,-rpath,"/home/adabbott/Git/libint_dev/libint-2.7.0-beta.6/PREFIX/lib"

TARGETS := libint_interface.cpython-36m-x86_64-linux-gnu.so
OBJ     := libint_interface.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

# To have libint2 play nice with Pybind11, we need it compiled with shared libraries *.so, and we need to point to them 
# and encode where they are in the compiled executable with -Wl,-rpath
# We also need fPIC, shared flags, the usual libint includes, pybind11 includes, and library paths, and library name -lint2
# TODO make -Wl,-rpath string a VARIALBE listed above next to includes,libs
$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -O3 -fPIC -shared -std=c++11 -I $(INCL1) -I $(INCL2) -I $(INCL3) -I $(INCL4) -I $(INCL5) -lint2 -L $(LIBS) $(RPATH)

$(TARGETS): $(OBJ)
	$(CC)  $^ -o $@ -O3 -fPIC -shared -std=c++11 -I $(INCL1) -I $(INCL2) -I $(INCL3) -I $(INCL4) -I $(INCL5) -lint2 -L $(LIBS) $(RPATH) 
