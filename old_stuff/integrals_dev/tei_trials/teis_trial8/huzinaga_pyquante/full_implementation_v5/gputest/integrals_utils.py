import jax
import jax.numpy as np
import numpy as onp
from functools import partial
#from jax.config import config; config.update("jax_enable_x64", True)
 
@jax.jit
def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    c = np.exp(np.dot(A-C,A-C) * (-alpha_bra * alpha_ket / (alpha_bra + alpha_ket)))
    return R,c

def find_unique_shells(nshells):
    '''Find shell quartets which correspond to corresponding to unique two-electron integrals, i>=j, k>=l, IJ>=KL'''
    v = onp.arange(nshells,dtype=np.int16) 
    indices = old_cartesian_product(v,v,v,v)
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3]) 
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2] * (indices[:,2] + 1)/2 + indices[:,3]
    mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

#@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    #tmp = np.meshgrid(*arrays, indexing='ij')
    return np.asarray(tmp)

def old_cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def am_vectors(am, length=3):
    '''
    Builds up all possible angular momentum component vectors of with total angular momentum 'am'
    am = 2 ---> [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    Returns a generator which must be converted to an iterable,
    for example, call the following: [list(i) for i in am_vectors(2)]

    Works by building up each possibility :
    For a given value in reversed(range(am+1)), find all other possible values for other entries in length 3 vector
     value     am_vectors(am-value,length-1)    (value,) + permutation
       2 --->         [0,0]                 ---> [2,0,0] ---> dxx
       1 --->         [1,0]                 ---> [1,1,0] ---> dxy
         --->         [0,1]                 ---> [1,0,1] ---> dxz
       0 --->         [2,0]                 ---> [0,2,0] ---> dyy
         --->         [1,1]                 ---> [0,1,1] ---> dyz
         --->         [0,2]                 ---> [0,0,2] ---> dzz
    '''
    if length == 1:
        yield (am,)
    else:
        # reverse so angular momentum order is canonical, e.g., dxx dxy dxz dyy dyz dzz
        for value in reversed(range(am + 1)):
            for permutation in am_vectors(am - value,length - 1):
                yield (value,) + permutation

