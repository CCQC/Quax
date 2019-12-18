import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
 
xgrid_array = np.asarray(onp.arange(0, 30, 1e-5))
# Load boys function values, defined in the range x=0,30, at 1e-5 intervals
# NOTE: The factorial pre-factors and minus signs are appropriately fused into the boys function values
boys = np.asarray(onp.load('boys/boys_F0_F10_grid_0_30_1e5.npy'))

#def boys0(x):
#    interval = 1e-5 # The interval of the precomputed Boys function grid
#    i = jax.lax.convert_element_type(jax.lax.round(x / interval), np.int64) # index of gridpoint nearest to x
#    xgrid = xgrid_array[i] # grid x-value
#    xx = x - xgrid
#
#    #Assume fused factorial factors/minus signs, preslice, convert to lax ops 
#    s = boys[:,i]
#    F = jax.lax.add(s[0], 
#        jax.lax.add(jax.lax.mul(xx,s[1]),  
#        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,2.),s[2]),
#        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,3.),s[3]),
#        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,4.),s[4]), 
#                    jax.lax.mul(jax.lax.pow(xx,5.),s[5]))))))
#    return F

def boys0(x):
    interval = 1e-5 # The interval of the precomputed Boys function grid
    i = jax.lax.convert_element_type(jax.lax.round(x / interval), np.int64) # index of gridpoint nearest to x
    xgrid = xgrid_array[i] # grid x-value
    xx = x - xgrid
    # when x>= 30, the grid breaks down. See how expensive this is.

    # We either have to make the grid huge, or use np.where and the border expression F0(x>=30) = sqrt(pi) / 2 * sqrt(x)
    # TODO causes huge memory blow ups in np.where?
    s = boys[:,i]
    F = np.where(x<= 30, jax.lax.add(s[0], 
                         jax.lax.add(jax.lax.mul(xx,s[1]),  
                         jax.lax.add(jax.lax.mul(jax.lax.pow(xx,2.),s[2]),
                         jax.lax.add(jax.lax.mul(jax.lax.pow(xx,3.),s[3]),
                         jax.lax.add(jax.lax.mul(jax.lax.pow(xx,4.),s[4]), 
                         jax.lax.mul(jax.lax.pow(xx,5.),s[5])))))),
                         np.sqrt(np.pi) / (2 * np.sqrt(x)))
    return F

#NOTE: the below custom jvp definition works, but doesnt affect performance of ERI hessians at all, oddly enough
#@jax.custom_transforms
#def boys0(x):
#    interval = 1e-5 # The interval of the precomputed Boys function grid
#    i = jax.lax.convert_element_type(jax.lax.round(x / interval), np.int64) # index of gridpoint nearest to x
#    xgrid = xgrid_array[i] # grid x-value
#    xx = x - xgrid
#
#    #Assume fused factorial factors/minus signs, preslice, convert to lax ops 
#    s = boys[:,i]
#    F = np.array([s[0], jax.lax.mul(xx,s[1]), jax.lax.mul(jax.lax.pow(xx,2.),s[2]),
#                  jax.lax.mul(jax.lax.pow(xx,3.),s[3]),jax.lax.mul(jax.lax.pow(xx,4.),s[4]), 
#                  jax.lax.mul(jax.lax.pow(xx,5.),s[5])])
#    return F
#

@jax.jit
def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    c = np.exp(np.dot(A-C,A-C) * (-alpha_bra * alpha_ket / (alpha_bra + alpha_ket)))
    return R,c

#@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    #tmp = np.meshgrid(*arrays, indexing='ij')
    return np.asarray(tmp)

def find_unique_shells(nshells):
    '''Find shell quartets which correspond to corresponding to unique two-electron integrals, i>=j, k>=l, IJ>=KL'''
    v = onp.arange(nshells,dtype=np.int16) 
    indices = old_cartesian_product(v,v,v,v)
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3]) 
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2] * (indices[:,2] + 1)/2 + indices[:,3]
    mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

def old_cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


@partial(jax.jit, static_argnums=(0,))
def lower_take_mask(ai):
    """
    Gives an array of same size as ai or bi for below equations. Each value in this array indicates
    which primitive (a-1i|b) or (a|b-1i) to take from the lower angular momentum function when evaluating the 
    second term in 
    (a + 1i | b) = 1/2alpha * (d/dAi (a|b) + ai (a - 1i | b))
    or
    (a | b + 1i) = 1/2beta  * (d/dBi (a|b) + bi (a | b - 1i))
    """
    num_nonzero = np.count_nonzero(ai)
    take = np.zeros_like(ai) 
    take = jax.ops.index_update(take, ai!=0, np.arange(num_nonzero) + 1)
    return take

