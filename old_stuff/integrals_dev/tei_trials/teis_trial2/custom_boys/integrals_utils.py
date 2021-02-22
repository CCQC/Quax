import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

# Using custom transforms:
@jax.custom_transforms
def boys(x):
    ''' F0(x) boys function '''
    return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-12) * jax.lax.erf(jax.lax.sqrt(x + 1e-12))

# Function that tells how to modify the function's Jacobian-vector product
# We want the primal to just be the input, and the primal out to be the evaluation of the original function
# We want the input tangent 'vector' to just be 1.0 (scalar) since this function is scalar -> scalar,
# and the output tangent 'vector' to be the derivative definition of the F0 boys function given by Helgaker: d/dx F0(x) = -F1(x)
# for numerical stability at x near 0, we add a small epsilon. Hopefully this doesnt screw up higher order derivatives.
def boys_jvp_rule(g, ans, x):
    #result = jax.lax.select(x < 1e-8, g * ((-0.3333333333333333333) + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629)),
    #                                  g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
    x =  x + 1e-12
    result = g * jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x))
    return result 

jax.defjvp(boys, boys_jvp_rule)

def control(arg):
    result = jax.lax.select(arg < 1e-8, 1.0 + (-arg * 0.3333333333333333333) + ((-arg)**2 * 0.1) + ((-arg)**3 * 0.023809523809523808) + ((-arg)**4 * 0.004629629629629629) + ((-arg)**5 * 0.0007575757575757576) + ((-arg)**6 * 0.0001068376068376068), 0.5 * jax.lax.sqrt(np.pi) * jax.lax.rsqrt(arg) * jax.lax.erf(jax.lax.sqrt(arg)))
    return result

#print("Checking raw boys function values for 0, 0.1, 1.1, 10.1")
#print(control(0.),boys(0.))
#print(control(0.1),boys(0.1))
#print(control(1.1),boys(1.1))
#print(control(10.1),boys(10.1))
#
#print("Checking jacfwd values")
#print(jax.jacfwd(control, argnums=0)(0.),jax.jacfwd(boys  , argnums=0)(0.))
#print(jax.jacfwd(control, argnums=0)(0.5),jax.jacfwd(boys , argnums=0)(0.5))
#print(jax.jacfwd(control, argnums=0)(10.5),jax.jacfwd(boys, argnums=0)(10.5))
#
#
#print("Checking jacfwd jacfwd values")
#print(jax.jacfwd(jax.jacfwd(control))(0.), jax.jacfwd(jax.jacfwd(boys))(0.))
#print(jax.jacfwd(jax.jacfwd(control))(0.5),jax.jacfwd(jax.jacfwd(boys))(0.5))
#print(jax.jacfwd(jax.jacfwd(control))(10.5),jax.jacfwd(jax.jacfwd(boys))(10.5))
#
#print("Checking jacfwd jacfwd jacfwd values")
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(0.), jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(0.))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(0.5),jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(0.5))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(control)))(10.5),jax.jacfwd(jax.jacfwd(jax.jacfwd(boys)))(10.5))
#
#
#print("Checking jacfwd jacfwd jacfwd jacfwd values")
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(0.), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(0.))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(0.5), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(0.5))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(control))))(10.5), jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys))))(10.5))
#
#print("Checking jacrev values")
#print(jax.jacrev(control)(0.),jax.jacrev(boys)(0.))
#print(jax.jacrev(control)(0.5),jax.jacrev(boys)(0.5))
#print(jax.jacrev(control)(10.5),jax.jacrev(boys)(10.5))
#
#
#print("Checking jacrev jacrev values")
#print(jax.jacrev(jax.jacrev(control))(0.), jax.jacrev(jax.jacrev(boys))(0.))
#print(jax.jacrev(jax.jacrev(control))(0.5),jax.jacrev(jax.jacrev(boys))(0.5))
#print(jax.jacrev(jax.jacrev(control))(10.5),jax.jacrev(jax.jacrev(boys))(10.5))
#
#print("Checking jacrev jacrev jacrev values")
#print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(0.), jax.jacrev(jax.jacrev(jax.jacrev(boys)))(0.))
#print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(0.5),jax.jacrev(jax.jacrev(jax.jacrev(boys)))(0.5))
#print(jax.jacrev(jax.jacrev(jax.jacrev(control)))(10.5),jax.jacrev(jax.jacrev(jax.jacrev(boys)))(10.5))
#
#
#print("Checking jacrev jacrev jacrev jacrev values")
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(0.), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(0.))
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(0.5), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(0.5))
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(control))))(10.5), jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))(10.5))


