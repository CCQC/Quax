import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

## Using custom transforms:
#@jax.custom_transforms
#def boys(x):
#    ''' F0(x) boys function '''
#    #return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-16) * jax.lax.erf(jax.lax.sqrt(x))
#    #x = x + 1e-12
#    return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-12) * jax.lax.erf(jax.lax.sqrt(x + 1e-12))
#
## Function that tells how to modify the function's Jacobian-vector product
## We want the primal to just be the input, and the primal out to be the evaluation of the original function
## We want the input tangent 'vector' to just be 1.0 (scalar) since this function is scalar -> scalar,
## and the output tangent 'vector' to be the derivative definition of the F0 boys function given by Helgaker: d/dx F0(x) = -F1(x)
## for numerical stability at x near 0, we add a small epsilon. Hopefully this doesnt screw up higher order derivatives.
#
#def boys_jvp_rule(primals, tangents):
#    primal, = primals
#    tangent, = tangents
#    #primal_out = boys(primal)
#    #tangent_out = np.where(primal < 1e-10, -(0.3333333333333333333) + (2* primal * 0.1) + -(3*primal**2 * 0.023809523809523808) + (4*primal**3 * 0.004629629629629629) , -(boys(primal) - np.exp(-primal)) / (2 * (primal)))
#    #primal_out = boys(primal)
#    tmp = 0.88622692545275798 * jax.lax.rsqrt(primal + 1e-12) * jax.lax.erf(jax.lax.sqrt(primal + 1e-12))
#    primal_out = tmp
#    tangent_out = np.where(primal < 1e-10, -(0.3333333333333333333) + (2* primal * 0.1) + -(3*primal**2 * 0.023809523809523808) + (4*primal**3 * 0.004629629629629629) , -(tmp - np.exp(-primal)) / (2 * (primal)))
#    return primal_out, tangent_out
#
#jax.defjvp_all(boys, boys_jvp_rule)

#def boys_vjp_rule(arg):
#    output = boys(arg)
#    def vjp_map(output_cotangents):
#        input_cotangents = 0.0
#        return input_cotangents
#    return output, vjp_map 

#jax.defvjp_all(boys, boys_vjp_rule)



#@jax.custom_transforms
#def boys0(arg):
#    result = jax.lax.select(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10), 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
#    return result
#
#jax.defjvp_all(boys0, 
# Try to use jax.core.Primitive to make a boys function 0 primitive

# primitive definition
def boys(x):
    return boys_p.bind(x)
boys_p = jax.core.Primitive('boys')

# evalutation rule of primitive
def boys_eval(x):
    x = x + 1e-12
    return 0.88622692545275798 * jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))
boys_p.def_impl(boys_eval)

# Jacobian-vector product rule of primitive
#def boys_jvp_rule(primal, tangent):
#    #NOTE: this may just be pushing the work down one more differentiation order, since youre still using np.where
#    #primal, = primals
#    #tangent, = tangents
#    primal_out = 0.88622692545275798 * jax.lax.rsqrt(primal + 1e-12) * jax.lax.erf(jax.lax.sqrt(primal + 1e-12))
#    tangent_out = np.where(primal < 1e-10, -(0.3333333333333333333) + (2* primal * 0.1) + -(3*primal**2 * 0.023809523809523808) + (4*primal**3 * 0.004629629629629629) , -(primal_out - np.exp(-primal)) / (2 * (primal)))
#    return primal_out, tangent_out

def boys_jvp_rule(g, x):
    #NOTE: this may just be pushing the work down one more differentiation order, since youre still using np.where
    #primal, = primals
    #tangent, = tangents
    tmp = boys(x)
    result = jax.lax.select(x < 1e-8, (-0.3333333333333333333) + + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629),
                                        jax.lax.div(-jax.lax.sub(tmp, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
    #primal_out = 0.88622692545275798 * jax.lax.rsqrt(primal + 1e-12) * jax.lax.erf(jax.lax.sqrt(primal + 1e-12))
    #tangent_out = np.where(primal < 1e-10, -(0.3333333333333333333) + (2 * primal * 0.1) + -(3 * primal**2 * 0.023809523809523808) + (4 * primal**3 * 0.004629629629629629) , -(primal_out - np.exp(-primal)) / (2 * (primal)))
    return result 

jax.interpreters.ad.defvjp(boys_p, lambda g,x : 2)
jax.interpreters.ad.defjvp(boys_p, boys_jvp_rule)

# Translation rule of primitive (for jit)
#def boys_translation_rule(c, x):
#    Use c.XlaOp's??  https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

# Batching (vectorization) rule of primitive (for vmap)
def boys_batching_rule(vector_arg_values, batch_axes):
  assert batch_axes[0] == batch_axes[1]
  assert batch_axes[0] == batch_axes[2]
  res = boys_p(*vector_arg_values)
  return res, batch_axes[0]

jax.interpreters.batching.primitive_batchers[boys_p] = boys_batching_rule

def control(arg):
    result = jax.lax.select(arg < 1e-8, 1.0 + (-arg * 0.3333333333333333333) + ((-arg)**2 * 0.1) + ((-arg)**3 * 0.023809523809523808) + ((-arg)**4 * 0.004629629629629629) + ((-arg)**5 * 0.0007575757575757576) + ((-arg)**6 * 0.0001068376068376068), 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
    return result


#print(boys(0.5))
#print(control(0.5))
#
#print(boys(0.0))
#print(control(0.0))

print("Checking raw boys function values for 0, 0.1, 1.1, 10.1")
print(control(0.),boys(0.))
print(control(0.1),boys(0.1))
print(control(1.1),boys(1.1))
print(control(10.1),boys(10.1))

print("Checking jacfwd values")
print(jax.jacfwd(control)(0.),jax.jacfwd(boys)(0.))
print(jax.jacfwd(control)(0.5),jax.jacfwd(boys)(0.5))
print(jax.jacfwd(control)(10.5),jax.jacfwd(boys)(10.5))


print("Checking jacfwd jacfwd values")
print(jax.jacfwd(jax.jacfwd(control))(0.), jax.jacfwd(jax.jacfwd(boys))(0.))
print(jax.jacfwd(jax.jacfwd(control))(0.5),jax.jacfwd(jax.jacfwd(boys))(0.5))
print(jax.jacfwd(jax.jacfwd(control))(10.5),jax.jacfwd(jax.jacfwd(boys))(10.5))

print("Checking jacfwd jacfwd jacfwd values")
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(0.), jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(0.))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(0.5),jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(0.5))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(10.5),jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(10.5))


print("Checking jacfwd jacfwd jacfwd jacfwd values")
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(0.), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(0.))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(0.5), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(0.5))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(10.5), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(10.5))

print("Checking jacrev values")
print(jax.jacrev(control)(0.),jax.jacrev(boys)(0.))
print(jax.jacrev(control)(0.5),jax.jacrev(boys)(0.5))
print(jax.jacrev(control)(10.5),jax.jacrev(boys)(10.5))


print("Checking jacrev jacrev values")
print(jax.jacrev(jax.jacrev(control))(0.), jax.jacrev(jax.jacrev(boys))(0.))
print(jax.jacrev(jax.jacrev(control))(0.5),jax.jacrev(jax.jacrev(boys))(0.5))
print(jax.jacrev(jax.jacrev(control))(10.5),jax.jacrev(jax.jacrev(boys))(10.5))

print("Checking jacrev jacrev jacrev values")
print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(0.), jax.jacrev(jax.jacrev(jax.jacrev(boys)))(0.))
print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(0.5),jax.jacrev(jax.jacrev(jax.jacrev(boys)))(0.5))
print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(10.5),jax.jacrev(jax.jacrev(jax.jacrev(boys)))(10.5))


print("Checking jacrev jacrev jacrev jacrev values")
print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(0.), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(0.))
print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(0.5), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(0.5))
print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(10.5), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(10.5))



#def boys0_impl(x):
#    return jax.interpreters.xla.apply_primitive(boys0_p, x)
#jax.interpreters.xla


"""
d/dx F0(x) = -F1(x)



Boys function taylor expansion and upward recurrence
Fn(x) = sum_k=0   (-x)^k / (k! (2n + 2k + 1))

F_(n+1)(x) = [(2n+1) F_n(x) - exp(-x) ] / [2x]
"""

## in all denominators, add a small fudge factor 1e-10
#@jax.jarrett
#def new_boys0(arg):
#    return 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg + 1e-12) * jax.lax.erf(jax.lax.sqrt(arg))
#
#@jax.jarrett
#def new_boys1(arg):
#    #return np.where(arg<1e-8, 1/3 + (arg / 5) - (arg**2/14), (boys0(arg) - np.exp(-arg)) / (2*arg))
#    return (new_boys0(arg) - np.exp(-arg)) / (2*(arg + 1e-12))
#
#@jax.jarrett
#def new_boys2(arg):
#    #return np.where(arg<1e-8, 1/5 + (arg / 7) - (arg**2/18), (3*boys1(arg) - np.exp(-arg)) / (2*arg))
#    return (3*new_boys1(arg) - np.exp(-arg)) / (2*(arg + 1e-12))
#
#
#
