
import jax.numpy as np
import jax
def mul(a, b):
  return a @ b

a = np.ones((8, 128, 128))
b = np.ones((4, 128, 128))

jax.pmap(mul)(a, b)
