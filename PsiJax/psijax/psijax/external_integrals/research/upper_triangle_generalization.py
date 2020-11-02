import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations

dim = 3 # just do 12 for now, simulates either 4 atom cartesian or 

# Remeber your compound index formula

#xidx, yidx = np.triu_indices(dim,0)
#
## With np triu
##print(np.stack((xidx,yidx)).T)
#
## Dimensions = 2 
## With combinations with replacement
#combos = np.asarray(list(cwr(np.arange(dim),2)))
#
## With loops  
#test = []
#for i in range(0,dim):
#    for j in range(i,dim):
#        test.append([i,j])
#
#loops = np.asarray(test)
#print("dimensions 2",np.allclose(loops, combos))
#
## Idea: create array of size which corr to flattened index
#flattened_indices = np.arange(loops.shape[0])
#print(loops)
#print(flattened_indices)
## Now create your lookup array. This can be done dynamically as needed in C++, or just store all of them at compile time
#lookup = np.zeros((dim,dim),int)
#
#for i in range(loops.shape[0]):
#    idx = loops[i]
#    for perm in permutations(idx):
#        lookup[perm] = i 
#print(lookup)

# 

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
    

print(generate_lookup(dim,2))
print(generate_lookup_old(dim,2))

        
# Dimensions = 3 
# With combinations with replacement
#combos = np.asarray(list(cwr(np.arange(dim),3)))
#
## With loops  
#test = []
#for i in range(0,dim):
#    for j in range(i,dim):
#        for k in range(j,dim):
#            test.append([i,j,k])
#
#loops = np.asarray(test)
#
##print(np.allclose(loops, combos))
#
### Dimensions = 4 
### With combinations with replacement
##combos = np.asarray(list(cwr(np.arange(dim),4)))
##
### With loops  
##test = []
##for i in range(0,dim):
##    for j in range(i,dim):
##        for k in range(j,dim):
##            for l in range(k,dim):
##                test.append([i,j,k,l])
##
##loops = np.asarray(test)
##
##print(np.allclose(loops, combos))
##
### How to convert to flattened index?
##
#
#
