import jax.numpy as np
import jax
from jax.config import config; config.update("jax_enable_x64", True)



def f(x):
    return np.linalg.norm(np.power(x + x, 2))

grad_calculator = jax.grad(f)
hessian_calculator = jax.hessian(f)
cubic_calculator = jax.jacfwd(jax.jacfwd(jax.jacrev(f)))
slow_cubic_calculator = jax.jacrev(jax.jacrev(jax.jacrev(f)))


a = np.array([1.0,2.0,3.0])
#b = np.tile(a, 50)


#cubic_calculator(b)
#slow_cubic_calculator(b)

#print("function eval")
#print(f(a))
#print("gradient eval")
#print(grad_calculator(a))
#print("hessian eval")
#print(hessian_calculator(a))
#print("cubic eval")
#print(cubic_calculator(a))
#
#print(slow_cubic_calculator(a))

