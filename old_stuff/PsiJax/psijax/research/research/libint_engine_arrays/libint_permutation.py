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

# Create array which maps multi index to 1d buffer index
# combinations with replacement is essentially a generalization of upper triangle forloops
# for i, for j<=i, for k<=j , etc 
def generate_buffer_lookup(dim_size, ndim):
    """dim: tuple of dimensions """
    dimensions = (dim_size,) * ndim 
    buffer_index_lookup = np.zeros(dimensions, dtype=int)
    count = 0
    for idx in combinations_with_replacement(np.arange(dim_size),ndim):
        # for all permutations of index, assign to array (totally symmetric)
        for perm in permutations(idx):
            buffer_index_lookup[perm] = count
        count += 1
    return buffer_index_lookup

# Create array which is of size [buffer, multi_indices]
# which maps 1d buffer index to multi_index tuple. 
# This is the inverse mapping of above function, generate_buffer_lookup
# for 4center  deriv1 it is (12, 1)
# for 4center  deriv2 it is (78, 2)
# for 4center  deriv3 it is (364, 3)
# for 4center  deriv4 it is (1365, 4)
def generate_multi_index_lookup(dim_size, ndim, nderivs):
    # dim_size=total differentiable parameters, ndim=deriv_order
    if ndim == 1:
      lookup = np.zeros((dim_size, 1),int)
      idx = 0
      for i in range(0, dim_size):
        lookup[idx, 0] = i
        idx += 1
    if ndim == 2:
      lookup = np.zeros((nderivs, 2),int)
      idx = 0
      for i in range(0, dim_size):
        for j in range(i,dim_size):
          lookup[idx, 0] = i
          lookup[idx, 1] = j
          idx += 1
    if ndim == 3:
      lookup = np.zeros((nderivs, 3),int)
      idx = 0
      for i in range(0, dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            lookup[idx, 0] = i
            lookup[idx, 1] = j
            lookup[idx, 2] = k
            idx += 1
    if ndim == 4:
      lookup = np.zeros((nderivs, 4),int)
      idx = 0
      for i in range(0, dim_size):
        for j in range(i,dim_size):
          for k in range(j,dim_size):
            for l in range(k,dim_size):
              lookup[idx, 0] = i
              lookup[idx, 1] = j
              lookup[idx, 2] = k
              lookup[idx, 3] = l
              idx += 1
    return lookup

def generate_eri_deriv_index(deriv_order):
    ncenters = 4
    nderivs = how_many_derivs(ncenters, deriv_order) # 12, 78, 364
    
    # all possiblities of swap_braket, swap_bra, and swap_ket 
    # gathered into an array of indices 0 or 1
    switch = np.array([0,1])
    possibilities = cartesian_product(switch, switch, switch)
    
    # Get lookup which maps flattened upper triangle index to multi-index in terms of full array axes 
    lookup_forward = generate_multi_index_lookup(12, deriv_order, nderivs)
    # Get lookup which maps multi-index back to flattened upper triangle index
    lookup_backward = generate_buffer_lookup(12, deriv_order)
    mapDerivIndex_xxxx= np.zeros((2,2,2, nderivs), dtype=int)
    
    for case in possibilities:
        swap_braket, swap_bra, swap_ket = case # each is 0 or 1 for swapping braket, bra centers, or ket centers
    
        # For every single derivative index 0-11, 0-78, 0-364, etc,
        # lookup its multi_idx, then apply the permutation rules for this BraKet::xx_xx
        # based on whether the parameters swap_braket, swap_bra, swap_ket are true
        for i in range(nderivs):
            multi_idx = lookup_forward[i]
            new_indices = []
            for idx in multi_idx:
                # If braket swap is on, all indices (0,1,2,3,4,5) ---> (6,7,8,9,10,11) and vice versa
                if swap_braket == 1: 
                    perm = [6,7,8,9,10,11,0,1,2,3,4,5]
                    idx = perm[idx]
                # If bra swap is on, all indices (0,1,2,3,4,5) ---> (3,4,5,0,1,2) and vice versa
                if swap_bra == 1: 
                    if idx < 6:
                        perm = [3,4,5,0,1,2,6,7,8,9,10,11]
                        idx = perm[idx]
                # If ket swap is on, all indices (6,7,8,9,10,11) ---> (9,10,11,6,7,8) and vice versa
                if swap_ket == 1: 
                    if idx > 5:
                        perm = [0,1,2,3,4,5,9,10,11,6,7,8]
                        idx = perm[idx]
                new_indices.append(idx)
            # Now lookup the other direction and determine flattened single index from this new multi index
            # and assign it to mapDerivIndex_xxxx
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

            mapDerivIndex_xxxx[swap_braket, swap_bra, swap_ket, i] = new_idx
    return mapDerivIndex_xxxx

#mapDerivIndex1_xxxx = generate_eri_deriv_index(1)
#mapDerivIndex2_xxxx = generate_eri_deriv_index(2)
#mapDerivIndex3_xxxx = generate_eri_deriv_index(3)
mapDerivIndex4_xxxx = generate_eri_deriv_index(4)

# All you needs to do is take these are repalce [] with {}
#print(repr(mapDerivIndex1_xxxx))
#print(repr(mapDerivIndex2_xxxx))
#print(repr(mapDerivIndex3_xxxx))
print(repr(mapDerivIndex4_xxxx))


