import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations

def generate_lookup(dim_size, ndim):
    # Based on order of differentiation (number of dimensions, ndim)
    # and using dimension size dim_size, instantiate lookup array
    # and collect the multidimensional indices corresponding to generalized upper triangle
    if ndim == 1:
      lookup = np.zeros((dim_size),int)
      combos = []
      for i in range(0,dim_size):
        combos.append([i])
    if ndim == 2:        
      lookup = np.zeros((dim_size,dim_size),int)
      combos = []
      for i in range(0,dim_size):
        for j in range(i,dim_size):
          combos.append([i,j])
    if ndim == 3:        
      lookup = np.zeros((dim_size,dim_size,dim_size),int)
      combos = []
      for i in range(0,dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            combos.append([i,j,k])
    if ndim == 4:
      lookup = np.zeros((dim_size,dim_size,dim_size,dim_size),int)
      combos = []
      for i in range(0,dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            for l in range(k,dim_size):
              combos.append([i,j,k,l])

    if ndim == 5:
      lookup = np.zeros((dim_size,dim_size,dim_size,dim_size,dim_size),int)
      combos = []
      for i in range(0,dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            for l in range(k,dim_size):
              for m in range(l,dim_size):
                combos.append([i,j,k,l,m])

    if ndim == 6:
      lookup = np.zeros((dim_size,dim_size,dim_size,dim_size,dim_size,dim_size),int)
      combos = []
      for i in range(0,dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            for l in range(k,dim_size):
              for m in range(l,dim_size):
                for n in range(m,dim_size):
                  combos.append([i,j,k,l,m,n])

    # Number of elements in generalized upper tri
    # This corresponds to flattened buffer index 
    size = len(combos)
    for i in range(size):
        # Get multi-dim index 
        multi_idx = combos[i]
        # Loop over all permutations, assign to totally symmetric lookup array
        for perm in permutations(multi_idx):
            lookup[perm] = i 
    return lookup

def generate_lookup_old(dim_size, ndim):
    """dim: tuple of dimensions """
    dimensions = (dim_size,) * ndim 
    buffer_index_lookup = np.zeros(dimensions, dtype=int)
    count = 0
    for idx in cwr(np.arange(dim_size),ndim):
        # for all permutations of index, assign to array (totally symmetric)
        for perm in permutations(idx):
            buffer_index_lookup[perm] = count
        count += 1
    return buffer_index_lookup
    

dim_size = 12 # just do 12 for now, simulates either 4 atom cartesian or 
a = generate_lookup(dim_size,2)
b = generate_lookup_old(dim_size,2)
print("2d match", np.allclose(a,b))

a = generate_lookup(dim_size,3)
b = generate_lookup_old(dim_size,3)
print(a)
print("3d match", np.allclose(a,b))

a = generate_lookup(dim_size,4)
b = generate_lookup_old(dim_size,4)
print("4d match", np.allclose(a,b))

a = generate_lookup(dim_size,5)
b = generate_lookup_old(dim_size,5)
print("5d match", np.allclose(a,b))

#a = generate_lookup(dim_size,6)
#b = generate_lookup_old(dim_size,6)
#print("6d match", np.allclose(a,b))

# These functions, generate_lookup*, create the buffer index lookup arrays. 
# When given a set of indices which represent the Shell derivative operator, e.g. 0,0 == d/dx1 d/dx1, 0,1 = d/dx1 d/dx2, etc
# these arrays, when indexed with those indices, give the flattened buffer index according to the order these shell derivatives
# are packed into the buffer by Libint.

# These arrays are always the same for finding the shell derivative mapping for overlap, kinetic, potential, and ERI for a given derivative order. 
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

# 10 atoms, 2nd derivs, do these shapes add to 666?
a = generate_lookup(6, 2)
print(a)
print("Number of nuclear 2nd derivs of 2 atoms should be ", a[-1,-1] + 1)

a = generate_lookup(30, 2)
print(a)

a = generate_lookup(6, 2)
print(a)

a = generate_lookup(6, 2)
print(a)


#def f(n2,i,j):
#    return min(i,j) * (n2 - min(i,j) - 1) // 2 + max(i,j)
## n2 matrix size times 2, i,j unordered indices.
#print(f(60, 5,5))



