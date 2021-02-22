# Testbed for writing C++ code
# Here we try to map deriv_vec ---> one buffer idx for a particular shell quartet

import numpy as np

# 3 atoms
deriv_vec = np.array([0,2,0,0,0,0,1,0,0])
deriv_order = np.sum(deriv_vec)

# Shell quartet atom indices
atom1 = 1
atom2 = 2
atom3 = 2
atom4 = 0

atom_index_list = np.asarray([atom1,atom2,atom3,atom4])

###################################################################################
##### This block is only dependent on deriv_vec, can be computed at beginning #####
nuclear_coords = []                                                               #
for i in range(deriv_vec.shape[0]):                                               #
    if deriv_vec[i] > 0:                                                          #
        for j in range(deriv_vec[i]):                                             #
            nuclear_coords.append(i)                                              #
#all of this is length deriv_level                                                # 
nuclear_coords = np.asarray(nuclear_coords)                                       #
#print(nuclear_coords)                                                             #
desired_atom_indices = nuclear_coords // 3                                        #
#print(desired_atom_indices)                                                       #
desired_coordinates = nuclear_coords % 3                                          #
#print(desired_coordinates)                                                        #
##### This block is only dependent on deriv_vec, can be computed at beginning #####
###################################################################################

# Create shell derivative 
#blah = []
#while len(blah) <= deriv_order:
#    # For every shell atom index
#    for i in range(atom_index_list.shape[0]):
#        # for every differentiated atom index
#        for j in range(desired_atom_indices.shape[0]):
#            # if there is a match 
#            if atom_index_list[i] == desired_atom_indices[j]:
                

shell_atom_index_list = [atom1,atom2,atom3,atom4]

# collections indices: shell0 shell1 shell2 or shell3
collection = []
#shell_derivative = []
# Collect the first N shell atom indices which are needed according to desired atom indices
for i,shell_atom_idx in enumerate(shell_atom_index_list):
        for j,atom_idx in enumerate(desired_atom_indices):
            if shell_atom_idx == atom_idx and len(collection) < deriv_order:
                #collection.append(i*3) # this appears to be broken sometimes, mabye you ahve to add correct desired cooridnates
                #collection.append(i*3 + desired_coordinates[j])
                collection.append(shell_atom_idx*3 + desired_coordinates[j])

                #collection.append( (i * 3 + desired_coordinates[j])  ) 
                #collection.append(i) 
                #shell_derivative.append(shell_atom_index_list[i] * 3)
        #if atom_idx in desired_atom_indices:

#shell_derivative = np.asarray(collection) + desired_coordinates

shell_derivative = np.asarray(collection)
#print(shell_derivative)

# ABOVE ALGO WORKS nevermind it doesnt


# new algo: make copy of deriv_vec. loop through shell atom indices 
#   
#print(desired_atom_indices)

# Okay. lets simplify things. First just check atom 1, is it in desired_atom_inndices? 
#for i,desired_atom in enumerate(desired_atom_indices):
#    if shell_atom_index_list[0] == desired_atom:
#        #print(shell_atom_index_list[0], desired_atom)
#    if shell_atom_index_list[1] == desired_atom:
#        print(shell_atom_index_list[1], desired_atom)
#    if shell_atom_index_list[2] == desired_atom:
#        print(shell_atom_index_list[2], desired_atom)
        
        
nuclear_coords = []                                                              
desired_atom_indices = []                                                              
desired_coordinates = []
for i in range(deriv_vec.shape[0]):                                               
    if deriv_vec[i] > 0:                                                          
        for j in range(deriv_vec[i]):                                             
            nuclear_coords.append(i)                                              
            desired_atom_indices.append(i // 3)                                              
            desired_coordinates.append(i % 3)

#print("break")
#print(deriv_vec)
#print(nuclear_coords)
#print(desired_atom_indices)
#print(desired_coordinates)
        
desired_shell_atoms = []
for i,desired_atom in enumerate(desired_atom_indices):
    if shell_atom_index_list[0] == desired_atom:
        #desired_shell_atoms.append(shell_atom_index_list[0])
        desired_shell_atoms.append(0)
    elif shell_atom_index_list[1] == desired_atom:
        #desired_shell_atoms.append(shell_atom_index_list[1])
        desired_shell_atoms.append(1)
    elif shell_atom_index_list[2] == desired_atom:
        #desired_shell_atoms.append(shell_atom_index_list[2])
        desired_shell_atoms.append(2)
    elif shell_atom_index_list[3] == desired_atom:
        #desired_shell_atoms.append(shell_atom_index_list[3])
        desired_shell_atoms.append(3)

# here you want the index of that atom index, 0, 1, 2, or 3
print(desired_shell_atoms)
# Take 3 * desired_shell_atoms + desired_coordinates

shell_derivative = []
for i in range(len(desired_shell_atoms)):
    shell_derivative.append(3 * desired_shell_atoms[i] + desired_coordinates[i])

print(deriv_vec)
print("shell atom indices", shell_atom_index_list)
print("shell derivative", shell_derivative)




def get_atoms_involved(deriv_vec):
    b = []
    for i in range(deriv_vec.shape[0]):
        if deriv_vec[i] > 0:
            b.append(i // 3) 
    return np.asarray(b)


#atoms_involved = get_atoms_involved(deriv_vec)
#print(atoms_involved)

    


    


