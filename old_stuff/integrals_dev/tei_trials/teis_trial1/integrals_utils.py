import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

#def boys0(arg):
#    """
#    JAX-compatible function for evaluating F0(x) (boys function for 0 angular momentum):
#    F0(x) = sqrt(pi/(4x)) * erf(sqrt(x)). 
#    For small x, denominator blows up, so a taylor expansion is typically used:   
#    F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
#    """
#    #result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
#    #result = np.where(arg < 1e-8, taylor(arg), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
#    # using rsqrt is much more efficient, lax.select bypasses an np.where checks, multiply by 0.5 better than div by sqrt(4). 
#    #result = np.where(arg < 1e-8, taylor(arg), 0.5 * np.sqrt(np.pi) * np.reciprocal(np.sqrt(arg)) * jax.scipy.special.erf(np.sqrt(arg)))
#    #result = jax.lax.select(arg < 1e-8, taylor(arg), 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
#    result = jax.lax.select(arg < 1e-8, taylor(arg),  0.88622692545275798 * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
#    
#    #result = jax.lax.select(arg < 0.1, taylor(arg), 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
#    return result
#
def taylor(arg):
    #return 1.0 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216) - (arg**5 / 1320) + (arg**6 / 9360)
    return 1.0 + (-arg * 0.3333333333333333333) + ((-arg)**2 * 0.1) + ((-arg)**3 * 0.023809523809523808) \
           + ((-arg)**4 * 0.004629629629629629) + ((-arg)**5 * 0.0007575757575757576) + ((-arg)**6 * 0.0001068376068376068)


# New boys function: define custom Jacobian-Vector product with stability epsilons to avoid
# branching in the computational graph due to np.where

@jax.custom_transforms
def boys0(x):
    ''' F0(x) boys function '''
    #return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-10) * jax.lax.erf(jax.lax.sqrt(x + 1e-12))
    #return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-10) * jax.lax.erf(jax.lax.sqrt(x + 1e-10))
    return 0.88622692545275798 * jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))


def boys0_jvp_rule(g, ans, x):
    #result = jax.lax.select(x < 1e-8, g * ((-0.3333333333333333333) + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629)) + -(5*(x)**4 * 0.0007575757575757576) + (6*(x)**5 * 0.0001068376068376068),
    #                                  g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
    # This gives instabilities
    #x =  x + 1e-12
    result = g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x))
    #result =  g * ((ans - jax.lax.exp(-x)) / (2*(x)))
    # Try to re-express in a more stable way
    #result = g * (ans - jax.lax.exp(-x))# / (2 * x) 
    #print(result)
    return result 
jax.defjvp(boys0, boys0_jvp_rule)



#           + ((-arg)**7 * 0.004629629629629629) + ((-arg)**8 * 0.0007575757575757576) + ((-arg)**9 * 0.0001068376068376068)

#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys0)))))(1e-5)
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(taylor)))))(1e-5)
# jaxpr of boys function blows up at high order.
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys0))))))))(1e-5)
#print(jaxpr)



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
#def boys1(arg):
#    """
#    JAX-compatible function for evaluating F0(x) (boys function for v=1 angular momentum):
#    F1(x) = [F0(x) - exp(-x)] / 2x
#    For small x, denominator blows up, so we use a taylor expansion
#    which is just the derivative of boys0 taylor expansion
#    F1(x) ~= 1/3 - x/5 + x/14
#    """
#    return np.where(arg<1e-10, 1/3, (boys0(arg) - np.exp(-arg)) / (2*arg)


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

