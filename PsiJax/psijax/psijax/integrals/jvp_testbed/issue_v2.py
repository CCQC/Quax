# Dear Jax team,
# In an external library, I have a high-performance implementation of a function and its arbitrary-order derivatives. 
# I would like to implement this function as a `jax.core.Primitive`, and define a JVP rule which does not explictly evaluate the operations for computing the derivative,
# but instead refers to the derivative implementation in the external library. 
# Ultimately, I would like to support nested derivatives `jacfwd(jacfwd(...))` of this primitive, within a larger computation which mostly uses native JAX primitives.
# I would like to know if this is possible. 
# To illustrate what I'm trying to do, I have come up with a simple analogue for my rather complicated use case. 
# f(x,y,z) = exp(1 * x + 2 * y + 3 * z)
# Any partial derivative  

import jax
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as onp
import jax.numpy as np

# "External library" implementation
def func(vec):
    coef = onp.array([1.,2.,3.])
    return onp.exp(onp.sum(coef * vec))

def func_deriv(vec, deriv_vec):
    """
    Parameters
    ----------
    vec: function argument
    deriv_vec : vector which indicates which components of 'vec' to differentiate with respect to, and how many times
                e.g. [1,0,0...,0] means take the first derivative wrt argument 1, 
                [1,1,0...,0] means take the second derivative wrt argument 1 and 2
                [1,1,1]      means take the third derivative wrt argument 1,2, and 3
    Returns 
    ----------
    A single partial derivative of func
    """
    coef = onp.array([1.,2.,3.])
    return onp.prod(onp.power(coef, deriv_vec)) * func(vec)  

# JAX implementation
def jax_func(vec):
    coef = np.array([1.,2.,3.])
    return np.exp(np.sum(coef * vec))

vec = np.array([1.0,2.0,3.0])
gradient = jax.jacfwd(jax_func)(vec)
#print(gradient)
hessian = jax.jacfwd(jax.jacfwd(jax_func))(vec)
print(hessian)

cubic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax_func)))(vec)
print(cubic)

# Check derivatives
#print(gradient)
#print(func_deriv(vec, onp.array([1,0,0])),func_deriv(vec, onp.array([0,1,0])), func_deriv(vec, onp.array([0,0,1])))
#print(hessian)
#print(func_deriv(vec, onp.array([2,0,0])))
#print(func_deriv(vec, onp.array([1,1,0])))
#print(func_deriv(vec, onp.array([1,0,1])))
#
#print(func_deriv(vec, onp.array([1,1,0])))
#print(func_deriv(vec, onp.array([0,2,0])))
#print(func_deriv(vec, onp.array([0,1,1])))
#
#print(func_deriv(vec, onp.array([1,0,1])))
#print(func_deriv(vec, onp.array([0,1,1])))
#print(func_deriv(vec, onp.array([0,0,2])))

# How do nested JVP's interact with jax_func?
#what = jax.jvp(jax_func, (vec,), (np.array([1.,0.,0.]),))
#print(what)
#huh = jax.jvp(jax.jvp(jax_func, (vec,), (np.array([1.,0.,0.]),)), (vec,), (np.array([1.,0.,0.])))
#print(huh)

#huh = jax.api._std_basis([1,0,0])
#print(huh)

def my_jacfwd(f):
    """A basic version of jax.jacfwd, assumes only one argument, no static args, etc"""
    def jacfun(x):
        # create little function that grabs tangents
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis
        Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        return np.transpose(Jt)
    return jacfun

# Nice. Don't have to worry about batching rules in your tests now!
def my_jacfwd_novmap(f):
    """A basic version of jax.jacfwd, with no vmap. assumes only one argument, no static args, etc"""
    def jacfun(x):
        # create little function that grabs tangents (second arg returned, hence [1])
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis. Note we are only mapping over tangents arg of jvp
        #Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        Jt = np.asarray([_jvp(i) for i in np.eye(len(x))])
        return np.transpose(Jt)
    return jacfun



# We can now  Define JAX primitives, and JVP's
func_p = jax.core.Primitive("func")
func_deriv_p = jax.core.Primitive("func_deriv")

def func(vec):
    return func_p.bind(vec)

def func_deriv(vec, deriv_vec):
    return func_deriv_p.bind(vec, deriv_vec)

def func_impl(vec):
    coef = onp.array([1.,2.,3.])
    return onp.exp(onp.sum(coef * vec))

def func_deriv_impl(vec, deriv_vec):
    """
    Parameters
    ----------
    vec: function argument
    deriv_vec : vector which indicates which components of 'vec' to differentiate with respect to, and how many times
                e.g. [1,0,0...,0] means take the first derivative wrt argument 1, 
                [1,1,0...,0] means take the second derivative wrt argument 1 and 2
                [1,1,1]      means take the third derivative wrt argument 1,2, and 3
    Returns 
    ----------
    A single partial derivative of func
    """
    coef = onp.array([1.,2.,3.])
    return onp.prod(onp.power(coef, deriv_vec)) * func(vec)  

func_p.def_impl(func_impl)
func_deriv_p.def_impl(func_deriv_impl)

def func_jvp(primals, tangents):
    vec, = primals
    out_primals = func(vec)
    out_tangents = func_deriv(vec, tangents[0])
    return out_primals, out_tangents

def func_deriv_jvp(primals, tangents):
    vec, deriv_vec = primals
    out_primals = func_deriv(vec, deriv_vec)
    out_tangents = func_deriv(vec, tangents[0] + deriv_vec)
    return out_primals, out_tangents

jax.ad.primitive_jvps[func_p] = func_jvp 
jax.ad.primitive_jvps[func_deriv_p] = func_deriv_jvp 

vec = np.array([1.0,2.0,3.0])

gradient = my_jacfwd_novmap(func)(vec)
print(gradient)

hessian = my_jacfwd_novmap(my_jacfwd_novmap(func))(vec)
print(hessian)

cubic = my_jacfwd_novmap(my_jacfwd_novmap(my_jacfwd_novmap(func)))(vec)
print(cubic)

# Define JVP rules

#def func_jvp(primals, tangents):
#    vec, = primals
#    out_primals = func(vec)
#
#    for v_dot in tangents:
#        d = func_deriv(vec, v_dot)
#        #out_tangents = func_deriv(vec, tangents)
#
#    return out_primals, out_tangents





