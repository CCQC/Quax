import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

#@jax.jit
def boys0(arg):
    """
    JAX-compatible function for evaluating F0(x) (boys function for 0 angular momentum):
    F0(x) = sqrt(pi/(4x)) * erf(sqrt(x)). 
    For small x, denominator blows up, so a taylor expansion is typically used:   
    F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
    """
    # First option: just force it to be 1 when 'arg' is small (
    # This is relatively fast, but inaccurate for small arg, i.e. gives 1.0 instead of 0.999999999333367) or w/e
    #TODO what cutoff can you get away with???
    #sqrtarg = np.sqrt(arg)
    #result = np.where(arg < 1e-10, 1.0, jax.scipy.special.erf(sqrtarg) * np.sqrt(np.pi) / (2 * sqrtarg))
    # alternative, use expansion, however the expansion must go to same order as angular momentum, otherwise
    #potential/eri integrals are wrong. 
    # It is unknown whether nuclear derivatives on top of integral derivatives will cause numerical issues
    # This currently just supports up to g functions. (arg**4 term)
    # Could just add a fudge factor arg**10 / 1e10 to cover higher order derivatives dimensional cases or some other function which is better? 
    #result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
    #result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
    result = jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg))
    return result



#@jax.jit
#def boys0(arg):
#    sqrtarg = arg**(0.5)
#    result = np.where(sqrtarg < 1e-5, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(sqrtarg) * np.sqrt(np.pi) / (2 * sqrtarg))
#    #TEMP TODO TODO
#    #result = 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216)
#    #result = np.where(sqrtarg < 1e-5, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(sqrtarg) * np.sqrt(np.pi) / (2 * sqrtarg))
#    #result = np.where(arg < 1e-5, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), np.reciprocal(arg))
#    return result 
#
#

"""
Boys function taylor expansion and upward recurrence
Fn(x) = sum_k=0   (-x)^k / (k! (2n + 2k + 1))

F_(n+1)(x) = [(2n+1) F_n(x) - exp(-x) ] / [2x]
"""

def boys2(arg):
    return np.where(arg<1e-8, 1/5 + (arg / 7) - (arg**2/18), (3*boys1(arg) - np.exp(-arg)) / (2*arg))


def boys1(arg):
    """
    JAX-compatible function for evaluating F0(x) (boys function for v=1 angular momentum):
    F1(x) = [F0(x) - exp(-x)] / 2x
    For small x, denominator blows up, so we use a taylor expansion
    which is just the derivative of boys0 taylor expansion
    F1(x) ~= 1/3 - x/5 + x/14
    """
    return np.where(arg<1e-8, 1/3 + (arg / 5) - (arg**2/14), (boys0(arg) - np.exp(-arg)) / (2*arg))


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

