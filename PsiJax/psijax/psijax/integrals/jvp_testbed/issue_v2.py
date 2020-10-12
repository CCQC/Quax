# In an external library, I have a high-performance implementation of a function and its arbitrary-order derivatives. 
# I would like to implement this function as a `jax.core.Primitive`, and define a Jacobian-vector product rule which 
# does not explictly evaluate the operations for computing the derivative, but instead refers to the derivative 
# implementation in the external library. Ultimately, I would like to support nested derivatives `jacfwd(jacfwd(...))` 
# of this primitive, within a larger computation which mostly uses native JAX primitives. Is this possible? 
# To illustrate what I'm trying to do, I have come up with a simple analogue for my rather complicated use case. 
# Consider the function R^3 --> R^1:  f(x,y,z) = exp(1 * x + 2 * y + 3 * z). 
# Any order partial derivative of this function is just the same function times a coefficient. 
# That coefficient is equal to Product([1,2,3]^[l,m,n]) where l,m,n are the orders of differentiation w.r.t. arguments x,y, and z, respectively.
# We can make functions which compute this functions evaluation and all of its partial derivatives which reference external libraries but compute them within JAX.

import jax
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as onp
import jax.numpy as np

# "External library" (NumPy) implementation of function and its derivative.
def func(vec):
    coef = onp.array([1.,2.,3.])
    return onp.exp(onp.sum(coef * vec))

def func_deriv(vec, deriv_vec):
    """
    Parameters
    ----------
    vec: function argument
    deriv_vec : vector which indicates which components of 'vec' to differentiate with respect to, and how many times
                e.g. [1,0,0] means take the first derivative wrt component 0 of the vector
                     [2,0,0] means take the second derivative wrt component 0 of the vector
                     [1,1,0] means differentiate once wrt component 0 and once wrt component 1 
                     [1,1,1]      means take the third derivative wrt component 0,1, and 2
    Returns 
    ----------
    A single partial derivative of function 'func'
    """
    coef = onp.array([1.,2.,3.])
    return onp.prod(onp.power(coef, deriv_vec)) * func(vec)  

# JAX-differentiable implementation, for reference and checking
def jax_func(vec):
    coef = np.array([1.,2.,3.])
    return np.exp(np.sum(coef * vec))

vec = np.array([1.0,2.0,3.0])
gradient = jax.jacfwd(jax_func)(vec)
hessian = jax.jacfwd(jax.jacfwd(jax_func))(vec)

# Check derivatives of jax_func versus our implementation of func_deriv
print("Checking derivatives")
print(gradient)
print(func_deriv(vec, onp.array([1,0,0])),func_deriv(vec, onp.array([0,1,0])), func_deriv(vec, onp.array([0,0,1])))
print(hessian)
print(func_deriv(vec, onp.array([2,0,0])))
print(func_deriv(vec, onp.array([1,1,0])))
print(func_deriv(vec, onp.array([1,0,1])))

print(func_deriv(vec, onp.array([1,1,0])))
print(func_deriv(vec, onp.array([0,2,0])))
print(func_deriv(vec, onp.array([0,1,1])))

print(func_deriv(vec, onp.array([1,0,1])))
print(func_deriv(vec, onp.array([0,1,1])))
print(func_deriv(vec, onp.array([0,0,2])))

# We can now define new JAX primitives which implement 'func' and 'func_deriv'.  
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
    out_tangents = func_deriv(vec, tangents[0]) # NOTE what happens here: we pass a basis vector [1,0,0...] to func_deriv
    return out_primals, out_tangents

def func_deriv_jvp(primals, tangents):
    vec, deriv_vec = primals
    out_primals = func_deriv(vec, deriv_vec)
    out_tangents = func_deriv(vec, tangents[0] + deriv_vec) # NOTE and then here, that basis vector is now deriv_vec, and we are adding the current tangent to it
    return out_primals, out_tangents

jax.ad.primitive_jvps[func_p] = func_jvp 
jax.ad.primitive_jvps[func_deriv_p] = func_deriv_jvp 

# At this point, we have not implemented a batching rule transformation for func and func_deriv, so we cannot use jax.vmap, which means jax.jacfwd, which depends on vmap, cannot be used
# directly with these functions. We can define our own version of jax.jacfwd, and then infer how to remove vmap dependence.
# We can use this to compare JAX's jax_func derivatives to ours, without worrying about implementing a batching rule.
def my_jacfwd(f):
    """A basic version of jax.jacfwd, assumes only one argument, no static args, etc"""
    def jacfun(x):
        # create little function that grabs tangents
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis
        Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        return np.transpose(Jt)
    return jacfun

# This works. Don't have to worry about batching rules in your tests now!
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

# Let's test our functions now. Here we will all-JAX utilities to compute the gradient, Hessian, and cubic derivative tensor of our function
jax_gradient = jax.jacfwd(jax_func)(vec)
jax_hessian = jax.jacfwd(jax.jacfwd(jax_func))(vec)
jax_cubic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax_func)))(vec)

# Now lets use our primitive function, 'func' and our cheeky version of jacfwd.
vec = np.array([1.0,2.0,3.0])
gradient = my_jacfwd_novmap(func)(vec)
hessian = my_jacfwd_novmap(my_jacfwd_novmap(func))(vec)
cubic = my_jacfwd_novmap(my_jacfwd_novmap(my_jacfwd_novmap(func)))(vec)

# Did we get the same results? 
print(np.allclose(jax_gradient, gradient))
print(np.allclose(jax_hessian, hessian))
print(np.allclose(jax_cubic, cubic))

# Yes! Great. TODO how to JIT? how to batch? 

# Summary notes: we did it.
# Above we have defined a function  R^n -> R^1, and a function which can compute arbitrary order partial derivatives wrt each vector component
# We used an external library (regular NumPy) to call these functions.
# We registered them as JAX primtives, defined their evaluation rules, defined their jacobian-vector product rules, and used a custom version of 
# jacfwd which does not depend on vmap (so I don't have to think about batching rules)
# and then compared nested jacfwds to JAX's nested jacfwd's, and they agree perfectly. AWESOME!

# Next on the todo list is simulate the above for some simple version of TEI derivatives, such as the partial derivative of G is just the differentiated coordinates times G
# If this works, then we can swap out that dummy derivative computation for the real deal, referencing Psi4. Boom!

# Still not clear is how to deal with weird args in your function, like basis and molecule specification. Maybe use a JAX issue our scour the source code.

