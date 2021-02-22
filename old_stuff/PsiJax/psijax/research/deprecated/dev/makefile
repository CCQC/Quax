# NOTE: These paths below need to be edited such that they point to a set of 
# Eigen headers, libint API headers libint2.h libint2.hpp, the rest of the libint2 headers, python headers, pybind11 headers, and the library location of libint2.a and libint2.so
# These paths currently use a version of libint2, which only has up to 2 angular momentum and two orders of differentiation, but I remembered to compile oei derivs this time.
CC      := g++

TARGETS := test 
OBJ     := test.o
DEPS    := 

.PHONY: all clean

all: $(TARGETS)

clean: rm -f $(OBJ)

# To have libint2 play nice with Pybind11, we need it compiled with shared libraries *.so, and we need to point to them 
# and encode where they are in the compiled executable with -Wl,-rpath
# We also need fPIC, shared flags, the usual libint includes, pybind11 includes, and library paths, and library name -lint2
$(OBJ): %.o : %.cc $(DEPS)
	$(CC) -c $< -o $@ -std=c++11 

$(TARGETS): $(OBJ)
	$(CC)  $^ -o $@ -std=c++11 
