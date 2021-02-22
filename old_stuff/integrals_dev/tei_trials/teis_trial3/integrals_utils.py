import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
 
xgrid_array = np.asarray(onp.arange(0, 30, 1e-5))
# Load boys function values, defined in the range x=0,30, at 1e-5 intervals
# NOTE: The factorial pre-factors and minus signs are appropriately fused into the boys function values
boys = np.asarray(onp.load('boys/boys_F0_F10_grid_0_30_1e5.npy'))

def boys0(x):
    interval = 1e-5 # The interval of the precomputed Boys function grid
    i = jax.lax.convert_element_type(jax.lax.round(x / interval), np.int64) # index of gridpoint nearest to x
    xgrid = xgrid_array[i] # grid x-value
    xx = x - xgrid

    #Assume fused factorial factors/minus signs, preslice, convert to lax ops 
    s = boys[:,i]
    F = jax.lax.add(s[0], 
        jax.lax.add(jax.lax.mul(xx,s[1]),  
        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,2.),s[2]),
        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,3.),s[3]),
        jax.lax.add(jax.lax.mul(jax.lax.pow(xx,4.),s[4]), 
                    jax.lax.mul(jax.lax.pow(xx,5.),s[5]))))))
    return F

def boys1(x):
    return -jax.grad(boys0)(x)

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
#def boys0_jvp_rule(g, ans, x):
#    """
#    We use a Boys function implementation given by eqn 33 of Weiss, Ochsenfeld, J. Comp. Chem. 2015, 36, 1390-1398
#    Noting that the Boys0 function and its derivatives are, for
#    xx = (x- xgrid), F0, F1, F2.. are precomputed Boys function values at xgrid, 
#    F  = F0 - xx F1 + xx^2 F2 - xx^3 F3 + xx^4 F4 + xx^5 F5
#    dF =  0 - F1 + 2xx F2 - 3xx^2 + 4xx^3 F4 + 5xx^4 F5
#    ddF =  0 - 0 + 2 F2 - 6xx + 12xx^2 F4 + 15xx^3 F5
#
#    You are really just dividing the previous by pad(arange(n_nonzero), (size_previous - size_arange_nonzero, 0)) / xx
#    """
#    interval = 1e-5 # The interval of the precomputed Boys function grid
#    i = jax.lax.convert_element_type(jax.lax.round(x / interval), np.int64) # index of gridpoint nearest to x
#    xgrid = xgrid_array[i] # grid x-value
#    xx = x - xgrid
#
#    # If a value in boys vector is 0, this function has already been differentiated, or its just 0 anyway
#    # pad the beginning of this array with 0's to match length of 'ans'
#    tmp1 = np.arange(np.count_nonzero(ans)) / (xx + 1e-12)
#    tmp = np.pad(tmp1, (ans.shape[0] - tmp1.shape[0], 0))
#    return g * ans * tmp

#
#    #Assume fused factorial factors/minus signs, preslice, convert to lax ops 
#    s = boys[:,i]
#    F = g * jax.lax.add(s[1],  
#        jax.lax.add(jax.lax.mul(2*xx,s[2]),
#        jax.lax.add(jax.lax.mul(jax.lax.pow(3*xx,2.),s[3]),
#        jax.lax.add(jax.lax.mul(jax.lax.pow(4*xx,3.),s[4]), 
#                    jax.lax.mul(jax.lax.pow(5*xx,4.),s[5])))))
#
#    #result = g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x))
#    return F 
#jax.defjvp(boys0, boys0_jvp_rule)


def old(boys_arg):
    F = jax.lax.select(boys_arg < 1e-12, taylor(boys_arg), boys_old(boys_arg)) 
    return F

def boys_old(x):
    return 0.88622692545275798 * jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))

def taylor(arg):
    return 1.0 + (-arg * 0.3333333333333333333) + ((-arg)**2 * 0.1) + ((-arg)**3 * 0.023809523809523808) \
           + ((-arg)**4 * 0.004629629629629629) + ((-arg)**5 * 0.0007575757575757576) + ((-arg)**6 * 0.0001068376068376068)


#
## New boys function: define custom Jacobian-Vector product with stability epsilons to avoid
## branching in the computational graph due to np.where
#@jax.custom_transforms
#def boys0(x):
#    ''' F0(x) boys function '''
#    return 0.88622692545275798 * jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))
#
#def boys0_jvp_rule(g, ans, x):
#    result = g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x))
#    return result 
#jax.defjvp(boys0, boys0_jvp_rule)

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

