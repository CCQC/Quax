import jax
import jax.numpy as np

def f(x,y):
    return 2 * x + 3 * y

inp = np.array([1.0,2.0,3.0,4.0,5.0])

def partialf(i):
    grad_result = jax.grad(f,argnums=(0,1))(inp[i],inp[i])
    return grad_result

# computes all gradients
grad_results = jax.lax.map(partialf, np.arange(5))

# grab particular partial derivatives later
# ...


