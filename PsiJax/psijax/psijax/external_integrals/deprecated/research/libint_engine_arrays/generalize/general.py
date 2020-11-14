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
    # Number of possible derivatives
    nderivs = how_many_derivs(ncenters, deriv_order) # e.g. for 4center: 12, 78, 364, 1365
    # Number of differentiable parameters in a shell set (assumes 3 cartesian components for each center)
    nparams = ncenters * 3
    # Based on the number of centers, the possible permutations and size of mapDerivIndex changes 
    if ncenters == 4:
        swap_braket_perm = [6,7,8,9,10,11,0,1,2,3,4,5]
        swap_bra_perm = [3,4,5,0,1,2,6,7,8,9,10,11]
        swap_ket_perm = [0,1,2,3,4,5,9,10,11,6,7,8]
        # All possible on/off combinations of swap_braket, swap_bra, and swap_ket 
        # gathered into an array of indices 0 or 1
        switch = np.array([0,1])
        possibilities = cartesian_product(switch, switch, switch)
        # Construct array for xxxx case
        mapDerivIndex = np.zeros((2,2,2, nderivs), dtype=int)
    # If 3 centers, BraKet::xs_xx, can only swap the ket 
    if ncenters == 3:
        possibilities = [0,1]
        swap_ket_perm = [0,1,2,6,7,8,3,4,5]
        # Construct array for xsxx case
        mapDerivIndex = np.zeros((2, nderivs), dtype=int)
    # TODO should I add BraKet::xx_xs?
    
    # Get lookup which maps flattened upper triangle index to the multidimensional index 
    # in terms of full array axes. Each axis of this multidimensional array represents
    # a different partial derivative.
    lookup_forward = generate_multi_index_lookup(nparams, deriv_order, nderivs)
    # Get lookup which maps multi-index back to flattened upper triangle index
    lookup_backward = generate_buffer_lookup(nparams, deriv_order)
    
    for case in possibilities:
        # each swap_* value is 0 or 1 for swapping braket, bra centers, or ket centers
        if ncenters == 4:
            swap_braket, swap_bra, swap_ket = case 
        if ncenters == 3:
            swap_braket = 0
            swap_bra = 0
            swap_ket = case
    
        # For every single derivative index 0-11, 0-78, 0-364, etc,
        # lookup its multi_idx, then apply the permutation rules for this BraKet::* 
        # based on whether the parameters swap_braket, swap_bra, swap_ket are true
        for i in range(nderivs):
            multi_idx = lookup_forward[i]
            new_indices = []
            for idx in multi_idx:
                if swap_braket == 1: 
                    idx = swap_braket_perm[idx]
                if swap_bra == 1: 
                    idx = swap_bra_perm[idx]
                if swap_ket == 1: 
                    idx = swap_ket_perm[idx]
                new_indices.append(idx)
            # Now lookup the other direction and determine flattened single index from this new multi index
            # and assign it to mapDerivIndex
            if deriv_order == 1:
                idx1, = new_indices
                new_idx = lookup_backward[idx1]
            elif deriv_order == 2:
                idx1, idx2 = new_indices
                new_idx = lookup_backward[idx1, idx2]
            elif deriv_order == 3:
                idx1, idx2, idx3 = new_indices
                new_idx = lookup_backward[idx1, idx2, idx3]
            elif deriv_order == 4:
                idx1, idx2, idx3, idx4 = new_indices
                new_idx = lookup_backward[idx1, idx2, idx3, idx4]
            # The dimensions of mapDerivIndex changes based on how many centers
            if ncenters == 4:
                mapDerivIndex[swap_braket, swap_bra, swap_ket, i] = new_idx
            if ncenters == 3:
                mapDerivIndex[swap_ket, i] = new_idx
    return mapDerivIndex

#lookup_backward = generate_buffer_lookup(3, 1)
#print(lookup_backward)
#lookup_backward = generate_buffer_lookup(3, 4)
#print(lookup_backward)

# Okay, this is important. Any multidimensional buffer_lookup CONTAINS all lower dimensional ones.
# so you only need to generate the highest order one...
# for example, take the 6th order derivative tensor of something involving 6 parameters
lookup_backward = generate_buffer_lookup(6, 6)
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


#mapDerivIndex = generate_deriv_index_map(1, 4)
#print(repr(mapDerivIndex))
#mapDerivIndex = generate_deriv_index_map(1, 3)
#print(repr(mapDerivIndex))
#
#mapDerivIndex = generate_deriv_index_map(2, 4)
#print(repr(mapDerivIndex))
#mapDerivIndex = generate_deriv_index_map(2, 3)
#print(repr(mapDerivIndex))




