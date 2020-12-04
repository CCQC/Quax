import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations

# Define derivative vector, deriv order, shell atoms
deriv_vec = np.array([0,0,1,0,0,0])
deriv_order = int(np.sum(deriv_vec))

natom = int(deriv_vec.shape[0] / 2)
#shell_atom_index_list = [0,1,1,1]
shell_atom_index_list = [0,1]
ncart = deriv_vec.shape[0]
print("Deriv vec ", deriv_vec)
print("atoms ", shell_atom_index_list)

ncenters = 2
#k = ncenters * 3 # Dimension size = number centers * 3
k = ncenters * 3 + ncart * 2 # if potential, k is ncenters * 3 + ncart * 2
n = 1 * deriv_order # derivative order
dim = (k,) * n
buffer_index_lookup = np.zeros(dim, dtype=int)
count = 0
for idx in cwr(np.arange(k),n):
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


print('desired_atom_indices',desired_atom_indices)
print('desired_coordinates',desired_coordinates)

# Collect all indices along axis (which is 12 for eris, representing 4 shells xyz components, 6 for overlap/kinetic, 2 shells xyz components)
# which have atom indices which match the desired atoms to be differentiated according to deriv_vec
#indices = []
#for i in range(4):
#    atom_idx = shell_atom_index_list[i]
#    tmp = []
#    for j in range(len(desired_atom_indices)):
#        desired_atom_idx = desired_atom_indices[j]
#        if atom_idx == desired_atom_idx:
#            #indices.append(3 * i + desired_coordinates[j])
#            tmp.append(3 * i + desired_coordinates[j])
#    if tmp != []:
#        indices.append(tmp)

indices = []
for j in range(len(desired_atom_indices)):
    desired_atom_idx = desired_atom_indices[j]
    tmp = []
    for i in range(ncenters):
        atom_idx = shell_atom_index_list[i]
        if atom_idx == desired_atom_idx:
            tmp.append(3 * i + desired_coordinates[j])
    if tmp != []:
        indices.append(tmp)

# For potential, just do another iteration over 6 + ncart, 6 + 2 * ncart instead of 4
for j in range(len(desired_atom_indices)):
    desired_atom_idx = desired_atom_indices[j]
    tmp = []
    for i in range(natom): #6 + ncart, 6 + 2 * ncart):
        if i == desired_atom_idx:
            # if this is an atom we want, get coordinate, add offset
            offset = 6 + ncart
            tmp.append(offset + 3 * i + desired_coordinates[j])
    if tmp != []:
        indices.append(tmp)

print("indices",indices)

from itertools import product
print(list(product(*indices)))
print('ALL CART PRODUCTS')
for i in list(product(*indices)):
    print(list(i))

#def recurs(indices, mutli_idx, 

# I want to pick EXACTLY ONE element from each sublist of indices to form a multi_index, and then use that to index buffer_indices_lookup


# Unique combinations with replacement
#print(list(set(cwr(indices, deriv_order))))
#final_indices = list(set(cwr(indices, deriv_order)))

# What to do with these indices? Well, find their buffer indices!
#buffer_indices = []
#
#for i in range(len(final_indices)):
#    if deriv_order == 1:
#        idx1 = final_indices[i]
#        buffer_indices.append(buffer_index_lookup[idx1])
#    if deriv_order == 2:
#        idx1, idx2 = final_indices[i]
#        buffer_indices.append(buffer_index_lookup[idx1,idx2])
#
## Now loop over all buffer indices and fill shell quartet values
#print(buffer_indices)
    
    


    
    
    


