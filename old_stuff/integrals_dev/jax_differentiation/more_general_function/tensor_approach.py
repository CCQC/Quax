import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)

def cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def jax_cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function'''
    tmp = np.meshgrid(*arrays, indexing='ij')
    flattened = []
    for arr in tmp:
        flattened.append(arr.reshape(-1))
    return np.stack(flattened).T
        
# Function definitions. We always return a vector of all primitive values.
# NOTES:
# The base function just computes a single primitive. 
# The vectorized versions can compute many primitives with the same centers at the same time.
# All functions return a vector of primitives, the number of which is dependent on the angular momentum
# (s|p) creates 3 primitives. (p|p) creates 9 (redundant for now)

# investigate shapes of each function output
A = np.array([0.0, 0.0, -0.849220457955])
#B = np.array([0.0, 0.0, -0.859220457955])
B = np.array([0.0, 0.0,  0.849220457955])
alpha_bra = 0.5
alpha_ket = 0.5

c1_d = 0.489335770373359
c1_f = 0.3094831149945914
c1_g = 0.1654256833287603
c1 = c1_f
c2 = 0.4237772081237576

@jax.jit
def overlap_ss(A, B, aa, bb, c1=1, c2=1):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-B, A-B)) / (aa + bb)))
    return ss * c1 * c2

@jax.jit
def overlap_ps(A, B, alpha_bra, alpha_ket,c1,c2):
    '''Returns the (3,) vector (px|s) (py|s) (pz|s)'''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket,c1,c2)
    result = (oot_alpha_bra * first_term)
    return result 

@jax.jit
def overlap_ds(A, B, alpha_bra, alpha_ket,c1,c2):
    '''
    Returns (3,3) Rank 2 Tensor
        (dxx,s) (dxy,s) (dxz,s) 
        (dyx,s) (dyy,s) (dyz,s) 
        (dzx,s) (dzy,s) (dzz,s) 
    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2)
    ai = np.zeros((3,3))
    ai = jax.ops.index_update(ai, jax.ops.index[0,0], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[1,1], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[2,2], 1)
    second_term = ai * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2)
    result = oot_alpha_bra * (first_term + second_term)
    return result 

def overlap_fs(A, B, alpha_bra, alpha_ket,c1,c2):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # Issue: 4th order derivatives dont work with jacfwd... hmmm have to use jacrev
#    print(overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2))
    first_term = jax.jacfwd(overlap_ds, 0)(A,B,alpha_bra,alpha_ket,c1,c2)
#    print(np.transpose(first_term, (1,2,0)))

    # my best guess at what second_term is supposed to be
    px, py, pz = overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2)
    second_term =  np.array(
                    [[[2*px,py, 0],
                      [  py, 0, 0],
                      [  pz, 0, 0]],
                     [[0,   px, 0],
                      [px,2*py, 0], 
                      [0,   pz, 0]],
                     [[pz, 0,  px],
                      [0, pz,  py],  
                      [px,py,2*pz]]])
 

    #second_term =  np.einsum('ijk,i->ijk',ai,overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2))

    #TEMP TODO NOTE
    #tmp = overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2)
    #print(tmp)
    #second_term = ai * np.tile(tmp, (3,3,1))

    #second_term =  np.einsum('ijk,i->ijk',ai,overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2))
    
    #second_term =  np.einsum('ijk,i,j,k->ijk',ai,tmp,tmp,tmp)

    #print(first_term)
    #print(second_term)
    result = oot_alpha_bra * (first_term + second_term)
    print('FIRST')
    print(first_term)
    print('SECOND')
    print(second_term)
    print(result)
    return result

def overlap_gs(A, B, alpha_bra, alpha_ket,c1,c2):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # Issue: 4th order derivatives dont work with jacfwd... hmmm have to use jacrev
    first_term = jax.jacfwd(overlap_fs, 0)(A,B,alpha_bra,alpha_ket,c1,c2)
    ai = np.zeros((3,3,3,3))
    #ai = jax.ops.index_update(ai, jax.ops.index[0,0,0,0], 3)
    #ai = jax.ops.index_update(ai, jax.ops.index[0,0,1,1], 1) # TEMP TODO NOTE
    #ai = jax.ops.index_update(ai, jax.ops.index[1,1,1,1], 3)
    #ai = jax.ops.index_update(ai, jax.ops.index[2,2,2,2], 3)

    # this fixes the corr. value  # for a given tensor address (i,j,k,l...n), how many indices to the left of n equal it?
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,0,0], 3)
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,0,1], 0)
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,0,2], 0)
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,1,1], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,1,2], 0)
    ai = jax.ops.index_update(ai, jax.ops.index[0,0,2,2], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[0,1,1,1], 2)
    ai = jax.ops.index_update(ai, jax.ops.index[0,1,1,2], 0)
    ai = jax.ops.index_update(ai, jax.ops.index[0,1,2,2], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[0,2,2,2], 2)
    ai = jax.ops.index_update(ai, jax.ops.index[1,1,1,1], 3)
    ai = jax.ops.index_update(ai, jax.ops.index[1,1,1,2], 0)
    ai = jax.ops.index_update(ai, jax.ops.index[1,1,2,2], 1)
    ai = jax.ops.index_update(ai, jax.ops.index[1,2,2,2], 2)
    ai = jax.ops.index_update(ai, jax.ops.index[2,2,2,2], 3)
    

    second_term =  ai * overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2)

    result = oot_alpha_bra * (first_term + second_term)
    return result


print("(f|s)")
fs = overlap_fs(A,B,alpha_bra,alpha_ket,c1,c2)
print(fs)

#
## try full upper triangle, then smaller, then smaller
# Generalized lower triangle indices
##offset  k = 0 full block
#print(fs[0,0,0])
#print(fs[0,0,1])
#print(fs[0,0,2])
#print(fs[0,1,1])
#print(fs[0,1,2])
#print(fs[0,2,2])
##offset k = 0, but kick off first row 
#print(fs[1,1,1])
#print(fs[1,1,2])
#print(fs[1,2,2])
##offset k = 0, but kick off first and second row 
#print(fs[2,2,2])
#
#
#print("(g|s)")
#gs = overlap_gs(A,B,alpha_bra,alpha_ket,c1,c2)
#print(gs)
#print(gs[0,0,0,0])
#print(gs[0,0,0,1])
#print(gs[0,0,0,2])
#print(gs[0,0,1,1])
#print(gs[0,0,1,2])
#print(gs[0,0,2,2])
#
#print(gs[0,1,1,1])
#print(gs[0,1,1,2])
#print(gs[0,1,2,2])
#
#
#print(gs[0,2,2,2])
#
#print(gs[1,1,1,1])
#print(gs[1,1,1,2])
#print(gs[1,1,2,2])
#print(gs[1,2,2,2])
#print(gs[2,2,2,2])
#
#
#
