import numpy as np
import math
from itertools import combinations_with_replacement
from itertools import permutations
np.set_printoptions(threshold=100000)

def how_many_derivs(k, n):
    """k is number centers, n is deriv order, no potential integrals"""
    val = 1
    for i in range(n):
        val *= (3 * k + i)
    return int((1 / math.factorial(n)) * val)

def how_many_nuc_derivs(k, n, natoms):
    val = 1
    for i in range(n): 
        val *= (3 * (k + natoms) + i)
    val /= math.factorial(n)
    return int(val)

def cartesian_product(*arrays):
    '''Cartesian product of a series of arrays'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    return np.asarray(tmp)

# Create array which maps multi index to 
# 1d buffer index which is sequential along the flattened generalized upper triangle
# combinations with replacement is essentially a generalization of upper triangle forloops
# for i, for j>=i, for k>=j , etc 
def generate_buffer_lookup(nparams, deriv_order):
    """dim: tuple of dimensions """
    dimensions = (nparams,) * deriv_order 
    buffer_index_lookup = np.zeros(dimensions, dtype=int)
    count = 0
    for idx in combinations_with_replacement(np.arange(nparams),deriv_order):
        # for all permutations of index, assign to array (totally symmetric)
        for perm in permutations(idx):
            buffer_index_lookup[perm] = count
        count += 1
    return buffer_index_lookup

# This is the inverse mapping of above function, generate_buffer_lookup
# Create array which is of size [nderivs, deriv_order]
# which maps 1d buffer index (the flattened generalized upper triangle index)
# to the corresponding to multidimensional index tuple. 
# This was tested against og func
# for 4center  deriv_order=1 it is of shape (12, 1)
# for 4center  deriv_order=2 it is of shape (78, 2)
# for 4center  deriv_order=3 it is of shape (364, 3)
# for 4center  deriv_order=4 it is of shape (1365, 4)
def generate_multi_index_lookup(nparams, deriv_order, nderivs):
    lookup = np.zeros((nderivs, deriv_order),int)
    idx = 0
    # DUMB you literally just want the result of cwr
    for indices in combinations_with_replacement(np.arange(nparams), deriv_order):
      for i in range(len(indices)):
        lookup[idx, i] = indices[i]
      idx += 1
    return lookup

def generate_deriv_index_map(deriv_order, ncenters):
    """
    Rewrite so that it always creates the same size array and returns it?
    """
    # Number of possible derivatives
    nderivs = how_many_derivs(ncenters, deriv_order) # e.g. for 4center: 12, 78, 364, 1365
    # Number of differentiable parameters in a shell set (assumes 3 cartesian components for each center)
    nparams = ncenters * 3
    # The BraKet type determines what the base permutation affects are.
    if ncenters == 4:
        swap_braket_perm = [6,7,8,9,10,11,0,1,2,3,4,5]
        swap_bra_perm = [3,4,5,0,1,2,6,7,8,9,10,11]
        swap_ket_perm = [0,1,2,3,4,5,9,10,11,6,7,8]
        # All possible on/off combinations of swap_braket, swap_bra, and swap_ket 
        # gathered into an array of indices 0 or 1
        switch = np.array([0,1])
        possibilities = cartesian_product(switch, switch, switch)
    # If 3 centers, BraKet::xs_xx, can only swap the ket 
    if ncenters == 3:
        #possibilities = [0,1]
        possibilities = [[0,0,0],[0,0,1]]
        # braket and bra: do nothing
        swap_braket_perm = [0,1,2,3,4,5,6,7,8]
        swap_bra_perm = [0,1,2,3,4,5,6,7,8]
        swap_ket_perm = [0,1,2,6,7,8,3,4,5]
        # Construct array for xsxx case?
    # TODO should I add BraKet::xx_xs?
    mapDerivIndex = np.zeros((2,2,2, nderivs), dtype=np.int16)
    
    # Get lookup which maps flattened upper triangle index to the multidimensional index 
    # in terms of full array axes. Each axis of this multidimensional array represents
    # a different partial derivative.
    lookup_forward = generate_multi_index_lookup(nparams, deriv_order, nderivs)
    # Get lookup which maps multi-index back to flattened upper triangle index
    lookup_backward = generate_buffer_lookup(nparams, deriv_order)
    
    for case in possibilities:
        # each swap_* value is 0 or 1 for swapping braket, bra centers, or ket centers
        swap_braket, swap_bra, swap_ket = case 

        # For every single derivative index 0-11, 0-78, 0-364, etc,
        # lookup its multidimensional index, then apply the permutation rules for this BraKet::* 
        # based on whether the parameters swap_braket, swap_bra, swap_ket are true
        for z in range(nderivs):
            multi_idx = lookup_forward[z]
            new_indices = []
            for idx in multi_idx:
                if swap_braket == 1: 
                    idx = swap_braket_perm[idx]
                if swap_bra == 1: 
                    idx = swap_bra_perm[idx]
                if swap_ket == 1: 
                    idx = swap_ket_perm[idx]
                new_indices.append(idx)
            # Sort new_indices so it is in the order of upper triangle indices, i <= j <= k...
            new_indices.sort()

            #new_indices = np.asarray(new_indices)

            # There are two easy ways to map back to the 1d index here.
            # The first is to construct a (deriv_order)-dimensional array, where each element is a 1d buffer index.
            # Then we can just index the array with new_indices to find the 1d index
            # This is very fast, but has large memory requirement at higher orders.
            # Instead, we can loop through lookup_forward until new_indices matches an entry, and then that entry is the desired 1d buffer index.
            # Instead of looking up in an array, loop through the lookup_forward until you find the match
            # The number of loops is, at most,  the 1d generalized flattened upper triangle buffer index 
            # An alternative to this is constructing a multi-dim tensor storing the 1d values at each multi index coordinate, but this gets very large at higher deriv_order
            new_idx = 0
            for k in range(nderivs):
                if new_indices == list(lookup_forward[k]):
                #if np.allclose(new_indices, lookup_forward[k]): 
                    new_idx = k
                    break
            mapDerivIndex[swap_braket, swap_bra, swap_ket, z] = new_idx
    # If BraKet::xx_xx, use whole thing. If BraKet::xs_xx, only need [0][0][:][:] slice
    # If BraKet::xx_sx, need [0][:][0][:] slice
    return mapDerivIndex

#lookup_backward = generate_buffer_lookup(3, 1)
#print(lookup_backward)
#lookup_backward = generate_buffer_lookup(3, 4)
#print(lookup_backward)

# Okay, this is important. Any multidimensional buffer_lookup CONTAINS all lower dimensional ones.
# so you only need to generate the highest order one...
# for example, take the 6th order derivative tensor of something involving 6 parameters
#lookup_backward = generate_buffer_lookup(6, 6)
# note that the 1d slice is just the 1d buffer lookup
#print(lookup_backward[0,0,0,0,0,:])
# this 2d slice is 2d buffer lookup
#print(lookup_backward[0,0,0,0,:,:])
# this 3d slice is 3d buffer lookup
#print(np.allclose(lookup_backward[0,0,0,:,:,:], generate_buffer_lookup(6,3)))
# this 4d slice is 4d buffer lookup
#print(np.allclose(lookup_backward[0,0,0,:,:,:], generate_buffer_lookup(6,3)))

# How bad does it get? Well... suppose you go to 6th order derivs... and you have ERI's
#test = np.max(generate_buffer_lookup(12, 6))
#print(test)
#print(12**6)

#mapDerivIndex = generate_deriv_index_map(1, 3)
#print(repr(mapDerivIndex))
#mapDerivIndex = generate_deriv_index_map(1, 4)
#print(repr(mapDerivIndex))

#mapDerivIndex = generate_deriv_index_map(2, 4)
#print(repr(mapDerivIndex))

#print(len(list(combinations_with_replacement(np.arange(12), 1))))
#print(len(list(combinations_with_replacement(np.arange(12), 2))))
#print(len(list(combinations_with_replacement(np.arange(12), 3))))
#print(len(list(combinations_with_replacement(np.arange(12), 4))))
#print(len(list(combinations_with_replacement(np.arange(12), 5))))
#print(len(list(combinations_with_replacement(np.arange(12), 6))))

mapDerivIndex = generate_deriv_index_map(2, 4)
print(repr(mapDerivIndex))



#mapDerivIndex = generate_deriv_index_map(2, 4)
#print(repr(mapDerivIndex))
#mapDerivIndex = generate_deriv_index_map(2, 3)
#print(repr(mapDerivIndex))


#switch = np.array([0,1])
#possibilities = cartesian_product(switch, switch, switch)
#print(possibilities)        
                            

