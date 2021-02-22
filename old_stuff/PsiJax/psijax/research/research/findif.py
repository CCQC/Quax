# pip install fdm

import numpy as np
import jax.numpy as jnp
import fdm
import jax

# Function:
#f(x,y,z) = exp(1 * x + 2 * y + 3 * z)
def func(vec):
    coef = jnp.array([1.,2.,3.])
    return jnp.exp(jnp.sum(coef * vec))

findif_gradient = fdm.gradient(func)
jax_gradient = jax.jacfwd(func, 0)


inp1 =  np.array([0.1,0.2,0.3]) 
inp2 = jnp.array([0.1,0.2,0.3]) 

print(findif_gradient(inp1))
print(jax_gradient(inp2))

findif_hessian = fdm.jacobian(jax_gradient)
jax_hessian = jax.jacfwd(jax_gradient)

print(findif_hessian(inp1))
print(jax_hessian(inp2))

findif_cubic = fdm.jacobian(jax_hessian)
jax_cubic = jax.jacfwd(jax_hessian)

print(findif_cubic(inp1))
print(jax_cubic(inp2))
