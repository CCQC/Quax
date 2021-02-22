import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations

## Test 1
#deriv_vec = np.array([0,2,0,0,0,0,1,0,0])
#deriv_order = np.sum(deriv_vec)
#shell_atom_index_list = [1,2,2,0]
## shell derivative should contain: 10,10, and either 3 or 6

## Test 2
#deriv_vec = np.array([1,0,0,1,0,0,1,0,0])
#deriv_order = np.sum(deriv_vec)
#shell_atom_index_list = [1,1,0,2]
# shell derivative should contain: 6, 9, and either 0 or 3.  

## Test 3
#deriv_vec = np.array([0,0,0,0,0,2,0,1,0])
#deriv_order = np.sum(deriv_vec)
#shell_atom_index_list = [0,0,1,2]
## shell derivative should contain: 8, 8, and 10.  

# Test 4
#deriv_vec = np.array([0,1,0,0,0,1,0,0,0])
#deriv_order = np.sum(deriv_vec)
#shell_atom_index_list = [1,1,0,0]
# shell derivative should contain: (2 or 5) and (7 or 10) 

deriv_vec = np.array([0,0,2,0,0,0])
print("deriv_vec", deriv_vec)
natom = 2
deriv_order = int(np.sum(deriv_vec))
shell_atom_index_list = [0,0]

#k = 2 # number centers
n = deriv_order # derivative order
dim = (18,) * n
buffer_index_lookup = np.zeros(dim, dtype=int)
count = 0
for idx in cwr(np.arange(18),n):
    # for all permutations of index, assign to array since totally symmetric.
    for perm in permutations(idx):
        buffer_index_lookup[perm] = count
    count += 1
print(buffer_index_lookup)

# This can be done outside shell quartet loop
# Convert deriv_vec to a set of (DERIV_LEVEL) atom indices and component indices 0, 1, or 2
desired_atom_indices = []
desired_coordinates = []
for i in range(deriv_vec.shape[0]):
    if deriv_vec[i] > 0:
        for j in range(deriv_vec[i]):
            desired_atom_indices.append(i // 3)
            desired_coordinates.append(i % 3)

#TODO two nested lists
print('desired_atom_indices',desired_atom_indices)
print('desired_coordinates',desired_coordinates)

# Shell derivatives block
indices = []
for j in range(len(desired_atom_indices)):
    tmp_indices = []
    for i in range(2): # this range needs to be extended to 0,1, 2+ncart
        atom_idx = shell_atom_index_list[i]
        desired_atom_idx = desired_atom_indices[j]
        if atom_idx == desired_atom_idx:
            # With potential integrals, we have 6 shell indices, NCART dummy indices, the NCART indices
            tmp_indices.append(3 * i + desired_coordinates[j])
    if tmp_indices != []:
        indices.append(tmp_indices)

# Nuclear derivatives block
#tmp_indices = []
#for j in range(len(desired_atom_indices)):
#    new = (2 + natom) * 3 + 3 * desired_atom_indices[j] + desired_coordinates[j]
#    tmp_indices.append(new)
#indices.append(tmp_indices)

# Instead of looping over 2, we want to collect 0,1,

# Can we transpose the result with a diff loop?
transposed_indices = []
for i in range(2):
    tmp_indices = []
    for j in range(len(desired_atom_indices)):
        atom_idx = shell_atom_index_list[i]
        desired_atom_idx = desired_atom_indices[j]
        if atom_idx == desired_atom_idx:
            tmp_indices.append(3 * i + desired_coordinates[j])
    if tmp_indices != []:
        transposed_indices.append(tmp_indices)
print(transposed_indices)

# Can we add nuclear deriv info to tranposed_indices?
#tmp_indices = []
for j in range(len(desired_atom_indices)):
    new = (2 + natom) * 3 + 3 * desired_atom_indices[j] + desired_coordinates[j]
    transposed_indices[j].append(new)

# Do I have to consider the cross blocks? mixed derivatives involving shell and nuclear?
# probably but its gonna suck

print("indices",indices)
print("transposed_indices",transposed_indices)

# The number of sublists in indices is equal to the number of atoms in the shell quartet which participate in this nuclear derivative according to deriv vec
# The size of each sublists is equal to the order of differentiation
# We require all combinations of deriv_order elements from each of these sublists

# Attempt at rewrite:
# For tranposed indices, the number of sublists is equal to number of differentiated things
# size of each sublist is order of differentiation
#buffer_indices = []
#for i in range(len(transposed_indices)):
#    for j in range(len(transposed_indices)):
#        if deriv_order == 1:
#            idx = transposed_indices[i][0]
#            buffer_indices.append(buffer_index_lookup[idx])
#        elif deriv_order == 2:
#            idx1 = transposed_indices[i][0]
#            idx2 = transposed_indices[j][1]
#            buffer_indices.append(buffer_index_lookup[idx1,idx2])

buffer_indices = []
#for i in range(len(transposed_indices[0])):
#    for j in range(len(transposed_indices[1])):
#        if deriv_order == 1:
#            idx = transposed_indices[0][i]
#            buffer_indices.append(buffer_index_lookup[idx])
#        elif deriv_order == 2:
#            idx1 = transposed_indices[i][0]
#            idx2 = transposed_indices[j][1]
#            buffer_indices.append(buffer_index_lookup[idx1,idx2])

if deriv_order == 2:
    for i in range(len(transposed_indices[0])):
        for j in range(len(transposed_indices[1])):
            print(i,j)
            idx1 = transposed_indices[0][i]
            idx2 = transposed_indices[1][j]
            buffer_indices.append(buffer_index_lookup[idx1,idx2])

    #if deriv_order == 1: 
    #    idx = transposed_indices[i][0]
    #    buffer_indices.append(buffer_index_lookup[idx])
    #elif deriv_order == 2: 
    #    idx1, idx2 = transposed_indices[i][0], transposed_indices[i][1]
    #    buffer_indices.append(buffer_index_lookup[idx1,idx2])
        
print(buffer_indices)
        
buffer_indices = []
if deriv_order == 1:
  for i in range(len(indices[0])):
    idx = indices[0][i]
    buffer_indices.append(buffer_index_lookup[idx])
    
elif deriv_order == 2:
  for i in range(len(indices[0])):
    for j in range(len(indices[1])):
      idx1 = indices[0][i]
      idx2 = indices[1][j]
      buffer_indices.append(buffer_index_lookup[idx1,idx2])

print(buffer_indices)

#from itertools import product
#print(list(product([[1,2]



