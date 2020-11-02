import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations

def generate_lookup(dim, order):
    # Based on order of differentiation (number of dimensions)
    # and using dimension size dim, instantiate lookup array
    # and collect the multidimensional indices
    if order == 1:
      lookup = np.zeros((dim),int)
      combos = []
      for i in range(0,dim):
        combos.append([i])
    if order == 2:        
      lookup = np.zeros((dim,dim),int)
      combos = []
      for i in range(0,dim):
        for j in range(i,dim):
          combos.append([i,j])
    if order == 3:        
      lookup = np.zeros((dim,dim,dim),int)
      combos = []
      for i in range(0,dim):
        for j in range(i,dim):
          for k in range(j,dim):
            combos.append([i,j,k])
    if order == 4:
      lookup = np.zeros((dim,dim,dim,dim),int)
      combos = []
      for i in range(0,dim):
        for j in range(i,dim):
          for k in range(j,dim):
            for l in range(k,dim):
              combos.append([i,j,k,l])

    if order == 5:
      lookup = np.zeros((dim,dim,dim,dim,dim),int)
      combos = []
      for i in range(0,dim):
        for j in range(i,dim):
          for k in range(j,dim):
            for l in range(k,dim):
              for m in range(l,dim):
                combos.append([i,j,k,l,m])

    if order == 6:
      lookup = np.zeros((dim,dim,dim,dim,dim,dim),int)
      combos = []
      for i in range(0,dim):
        for j in range(i,dim):
          for k in range(j,dim):
            for l in range(k,dim):
              for m in range(l,dim):
                for n in range(m,dim):
                  combos.append([i,j,k,l,m,n])

    # Number of elements in generalized upper tri
    # This is the single-dimension index
    size = len(combos)
    for i in range(size):
        # Get multi-dim index 
        multi_idx = combos[i]
        # Loop over all permutations, assign to totally symmetric lookup array
        for perm in permutations(multi_idx):
            lookup[perm] = i 
    return lookup

def generate_lookup_old(dim, order):
    """dim: tuple of dimensions """
    dimensions = (dim,) * order
    buffer_index_lookup = np.zeros(dimensions, dtype=int)
    count = 0
    for idx in cwr(np.arange(dim),order):
        # for all permutations of index, assign to array (totally symmetric)
        for perm in permutations(idx):
            buffer_index_lookup[perm] = count
        count += 1
    return buffer_index_lookup
    

dim = 12 # just do 12 for now, simulates either 4 atom cartesian or 
a = generate_lookup(dim,2)
b = generate_lookup_old(dim,2)
print("2d match", np.allclose(a,b))

a = generate_lookup(dim,3)
b = generate_lookup_old(dim,3)
print("3d match", np.allclose(a,b))

a = generate_lookup(dim,4)
b = generate_lookup_old(dim,4)
print("4d match", np.allclose(a,b))

a = generate_lookup(dim,5)
b = generate_lookup_old(dim,5)
print("5d match", np.allclose(a,b))

a = generate_lookup(dim,6)
b = generate_lookup_old(dim,6)
print("6d match", np.allclose(a,b))

# These functions, generate_lookup*, create the buffer index lookup arrays. 
# When given a set of indices which represent the Shell derivative operator, e.g. 0,0 == d/dx1 d/dx1, 0,1 = d/dx1 d/dx2, etc
# these arrays, when indexed with those indices, give the flattened buffer index according to the order these shell derivatives
# are packed into the buffer by Libint.

# These arrays are always the same for finding the shell derivative maps for overlap, kinetic, potential, and ERI for a given derivative order. 
# The number of dimensions is equal to deriv_order.
# The size of the dimension is equal to the number of centers * 3  
# To support up to 4th order derivatives, we want to form the following lookups, ideally at compile time.
# oei lookup deriv1: dim_size = 6, ndim = 1
# oei lookup deriv2: dim_size = 6, ndim = 2
# oei lookup deriv3: dim_size = 6, ndim = 3
# oei lookup deriv4: dim_size = 6, ndim = 4

# tei lookup deriv1: dim_size = 12, ndim = 1
# tei lookup deriv2: dim_size = 12, ndim = 2
# tei lookup deriv3: dim_size = 12, ndim = 3
# tei lookup deriv4: dim_size = 12, ndim = 4

# For potential integrals, another lookup array needs to be dynamically created according to the number of atoms in the molecule. 
# This is because libint outputs a set of nuclear derivatives in addition to shell derivatives. 
# These nuclear derivatives are also only the unique ones, the generalized upper triangle of a (3*NATOM,3*NATOM,...,3*NATOM) array.
# So every time potential_deriv is called, we must generate
# nuclear_deriv_lookup = generate_lookup(3 * Natom, deriv_order)


