import numpy as np

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

deriv_vec = np.array([0,0,1,0,0,0])
shell_atom_index_list = [1,1,0,0]

# This can be done outside shell quartet loop
# Convert deriv_vec to a set of (DERIV_LEVEL) atom indices and component indices 0, 1, or 2
desired_atom_indices = []
desired_coordinates = []
for i in range(deriv_vec.shape[0]):
    if deriv_vec[i] > 0:
        for j in range(deriv_vec[i]):
            desired_atom_indices.append(i // 3)
            desired_coordinates.append(i % 3)

# Now we are in the shell quartet loop, before computing integrals
# Every shell quartet has 4 atom indices. 
# We can check if EVERY desired atom is contained in this set of 4 atom indices
# This will ensure the derivative we want is in the buffer
desired_shell_atoms = []
for desired_atom in desired_atom_indices:
    if shell_atom_index_list[0] == desired_atom:
        desired_shell_atoms.append(0)
    elif shell_atom_index_list[1] == desired_atom:
        desired_shell_atoms.append(1)
    elif shell_atom_index_list[2] == desired_atom:
        desired_shell_atoms.append(2)
    elif shell_atom_index_list[3] == desired_atom:
        desired_shell_atoms.append(3)

# If the length of desired_shell_atoms is not == DERIV_LEVEL, 
# this shell quartet can be SKIPPED, since it does not contain the desired derivative

# We can now convert the desired shell atom indices
# to a shell derivative.
# Find shell derivative [0-11, 0-11, ....]  
shell_derivative = []
for i in range(len(desired_shell_atoms)):
    shell_derivative.append(3 * desired_shell_atoms[i] + desired_coordinates[i])

print(deriv_vec)
print(shell_derivative)

# For n'th order derivative of k center integral, we can make a buffer_index_lookup array
# which takes in shell derivative index and outputs buffer index 
# This can be stored and then indexed once shell_derivative is found
import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import permutations
k = 2 # number centers
n = 2 # derivative order
dim = (3*k,) * n
buffer_index_lookup = np.zeros(dim, dtype=int)
count = 0
for idx in cwr(np.arange(3*k),n):
    # for all permutations of index, assign to array since totally symmetric.
    for perm in permutations(idx):
        buffer_index_lookup[perm] = count
    count += 1
print(buffer_index_lookup)


