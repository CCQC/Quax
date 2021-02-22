import jax.numpy as np
import jax

@jax.jit
def jitted_function(i):
    return np.sqrt(i ** i * i + i)

N = 100000
seed = jax.random.PRNGKey(0)
vec = jax.random.uniform(seed, (N,))
print(vec.shape)

# about 10 seconds
#for i in vec:
#    res = jitted_function(i)

# 0.57 sec for N = 100,000
# 0.66 sec for N = 1 billion
res = jax.lax.map(jitted_function, vec)
print(res.shape)


