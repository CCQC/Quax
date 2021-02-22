import jax
import jax.numpy as np

@jax.jit
def f(lA, lB):
    lp = lA + lB
    return np.zeros((lp,lp))
 
f(2,2)

